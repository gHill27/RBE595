# This is necessary to find the main code
from calendar import c
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from sensed_world import SensedWorld
from colorama import Fore, Back
import heapq
import time
import json
from enum import Enum
import numpy as np
from collections import deque
import os 
import random
import math
from events import Event
import sys

class State(Enum):
    Free = 1
    Monster = 2
    FarFromMonster = 3
    Bomb = 4


class TestCharacter(CharacterEntity):
    def __init__(self, name, avatar, x, y):
        super().__init__(name, avatar, x, y)
        self.wrld = None  # Initialize world reference
        self.path = []  # Initialize path list
        self.time = 0  # Initialize time variable
        self.score = 0 # Initialize score
        self.state: State = State.Free #starts assuming it has an empty environment
        self.GAMMA = 0.9
        self.LEARNING_RATE = 0.001
        if len(sys.argv) > 1:
            self.EXPLORATION_PROB = float(sys.argv[1])
        else:
            self.EXPLORATION_PROB = 0.0
        self.MONSTER_AVOID_RADUIS = 3
        self.dist = None
        self.pos_history = deque(maxlen=11)
        self.no_path_ticks = 0      # consecutive turns A* has failed 
        self.NO_PATH_THRESHOLD = 3  
        
       

        # load weights from q_learning_weights.json
        with open("q_learning_weights.json") as f:
            q_weights = json.load(f)["weights"]
            self.bomb_weights = q_weights["Bomb"]
            self.monster_weights = q_weights["Monster"]
            self.far_weights = q_weights["Far"]

        
        # feature for walls destroyed
        self.walls_destroyed = 0
        # current weight set
        self.weights = None
    
    def _is_bomb_active(self, wrld):
        return len(wrld.bombs) > 0

    def define_state(self, wrld, distances_matrix):
        old_state = self.state
        
        bomberman_dist_to_exit = distances_matrix[self.y][self.x] #row col
        all_monsters = [
            (m.x, m.y) for mlist in wrld.monsters.values() for m in mlist
        ]
        if not all_monsters:
            if self._is_bomb_active(wrld):
                self.state = State.Bomb
                print(f'old state {old_state} --> {self.state}')
                return 
            self.state = State.Free if bomberman_dist_to_exit > 0 else State.FarFromMonster
            print(f'old state {old_state} --> {self.state}')
            return
        
        monster_distances = []
        mdist = self.get_min_monster_dist(wrld)
        for monster in all_monsters:
            monster_distances.append(distances_matrix[monster[1]][monster[0]]) #row col

        winning_race = bomberman_dist_to_exit > 0 and (not monster_distances or bomberman_dist_to_exit < min(monster_distances))

        if winning_race:
            self.state = State.Free
        elif self._is_bomb_active(wrld):
            self.state = State.Bomb            
        elif mdist <= self.MONSTER_AVOID_RADUIS: #close to monster!
            self.state = State.Monster
        else:
            self.state = State.FarFromMonster
        

        print(f'old state {old_state} --> {self.state}')

    def do(self, wrld):
        # Your code here
        self.wrld = wrld  # Store the world reference for use in A* algorithm
        self.record_position()
        distances_matrix = self.compute_distances(wrld.exitcell)
        self.define_state(wrld,distances_matrix)
        pos = (self.x, self.y)
        goal = wrld.exitcell
        distances_matrix = self.compute_distances(wrld.exitcell)
        self.dist = distances_matrix
        name: str = None
        weight: dict = None
        #state logic:
        match self.state:
            case State.Free:
                self.path = self.Astar(pos,goal)[1:]
                if self.path:
                    best_move = self.path.pop(0)
                    # no need to look at Q-values if we have a free path to the exit
                    next_step = best_move
                    dx = next_step[0] - self.x
                    dy = next_step[1] - self.y
                    self.move(dx,dy)  # skip analysis and just move
                    return
                else:
                    self.state = State.FarFromMonster #TODO actually make this smarter maybe
                    print('debug to far state ')
                    weight = self.far_weights
                    name = "Far"
                    self.weights = self.far_weights

            case State.Bomb:
                weight = self.bomb_weights 
                name = "Bomb"
                self.weights = self.bomb_weights

            case State.FarFromMonster:
                weight = self.far_weights
                name = "Far"
                self.weights = self.far_weights

            case State.Monster:
                weight = self.monster_weights
                name = "Monster"
                self.weights = self.monster_weights

        #restructure do: 
        #should check states DONE
        #should then get the weights for that state DONE
        #should then go argmax of Q(s,a) = w1f1(s,a) + w2f2(s,a) + ... DONE
        #should then go delta <- real reward + gamma * argmax Q(s',a') - Q(s,a) DONE
        #should then update weight <- weight + learning factor * f1(s,a) DONE
        
        Q_s_a, best_move, sim_s_prime = self.argmaxQ(wrld)
        # print(f'Q(s,a) = {Q_s_a} \n best move is {best_move}')
        terminal = sim_s_prime.me(self) is None
        exited = any(e.tpe == Event.CHARACTER_FOUND_EXIT and e.character.name == self.name
             for e in sim_s_prime.events)
        died = terminal and not exited

        
        reward = -1*sim_s_prime.scores[self.name] - -1*wrld.scores[self.name]
        if exited:
            reward = 10000
        if died:
            reward -= 1000
        reward = reward/100.0 # scaling so it doesnt get massive weights 
        # print(f'reward is {reward}')
        if terminal:
            delta = reward - Q_s_a

        else:
            q_sp_ap = self.argmaxQ(sim_s_prime)[0]
            # print(f"Q(s',a') is {q_sp_ap}")
            delta = reward + q_sp_ap*self.GAMMA - Q_s_a
        # print(f'delta is {delta}')

        features_s_a = self.computefeatures(sim_s_prime) 
        
        for index, weight in enumerate(self.weights):
            self.weights[index] = weight + self.LEARNING_RATE * delta * features_s_a[index]

        self.update_weight_category(name, self.weights)
        if self.should_bomb(wrld,best_move):
            self.place_bomb()

        self.move(*best_move)



    def get_blocked_move(self,pos,goal):
        distance_to_goal = math.dist(pos,goal)
        best = pos
        best_dist = distance_to_goal
        for neighbor in self.get_neighbors(pos):
            if math.dist(neighbor,goal) < best_dist:
                best = neighbor
                best_dist = math.dist(neighbor,goal)
        
        return best
    
    def argmaxQ(self,wrld, explore = False):
        pos = (wrld.me(self).x,wrld.me(self).y)
        moves = self.get_valid_moves(wrld) #returns dx dy
        bestQ = -math.inf
        bestrank = -math.inf
        bestmove = (0,0)
        bestwrld = wrld
        
        if explore and random.random() < self.EXPLORATION_PROB:
            random_move = random.choice(moves)
            bestmove = random_move
            img_wrld = SensedWorld.from_world(wrld)   
            bestwrld = self.move_entities(random_move,img_wrld)
            bestQ = self.computeQfunction(bestwrld)
        
        else:
            for move in moves:
                img_wrld = SensedWorld.from_world(wrld)   
                nxt = self.move_entities(move,img_wrld)
                Q = self.computeQfunction(nxt)
                rank = Q# used exclusively for picking the best move
                if nxt.me(self) is None:
                    exited = any(e.tpe == Event.CHARACTER_FOUND_EXIT and e.character.name == self.name
                                for e in nxt.events)
                    rank = 1e9 if exited else -1e9

                if rank > bestrank:
                    bestrank = rank
                    bestQ = Q
                    bestmove = move
                    bestwrld = nxt
        return (bestQ, bestmove, bestwrld)

    def computeQfunction(self,wrld):
        features = self.computefeatures(wrld)
        p1 = self.weights[0] * features[0]
        p2 = self.weights[1] * features[1]
        p3 = self.weights[2] * features[2]
        p4 = self.weights[3] * features[3]
        return p1 + p2 + p3 + p4

    def computefeatures(self,wrld):
        #get weights here and compute
        if wrld.me(self) is None:
            exited = any(e.tpe == Event.CHARACTER_FOUND_EXIT and e.character.name == self.name
                     for e in wrld.events)
            return (0.0, 1.0, 0.0, 0.0, 1.0) if exited else (1.0, 0.0, 1.0, 0.0, 1.0)
        
        f1 = 1/(1+self.get_min_monster_dist(wrld))
        #f2
        threat = False
        for monsters in wrld.monsters.values():
            for monster in monsters:
                if self.Astar((wrld.me(self).x,wrld.me(self).y),(monster.x,monster.y)):
                    threat = True        
        if threat:
            maxd = np.max(self.dist)
            d = self.dist[wrld.me(self).y][wrld.me(self).x]
            if d <= 0:
                d = self.chebyshev_distance((wrld.me(self).x,wrld.me(self).y),wrld.exitcell)
                # print(f'no ideal path using chebyshevs distance {d}')
                f2 = -d/max(wrld.width(),wrld.height())
            else:
                f2 = -d/maxd
        else:
            f2 = 0  
        # print(threat)

        if (wrld.me(self).x, wrld.me(self).y) in self.simulate_explosion_danger_cells(wrld):
            f3 = 1.0         
        else: 
            f3 = 0.0
    
        #added number of neighbors weight to incentivise moves in the open.
        num_of_neighbors = len(self.get_neighbors((wrld.me(self).x, wrld.me(self).y)))
        f4 = 1/(10 - num_of_neighbors)
        # print(
        #     f"f1 = {f1} \nf2 = {f2} \nf3 = {f3} \nf4 = {f4}"
        # )
        return [f1, f2, f3, f4]

    def get_valid_moves(self,wrld):
        x = wrld.me(self).x
        y = wrld.me(self).y

        blocked = {(b.x, b.y) for b in wrld.bombs.values()}
        blocked |= {(e.x, e.y) for e in wrld.explosions.values()}
        blocked |= {(m.x, m.y) for ml in wrld.monsters.values() for m in ml}

        valid_moves = []
        directions = [(1,0),(-1,0),(0,1),(0,-1),(0,0),(1,1),(-1,-1),(1,-1),(-1,1)]
        # Inline checks to avoid extra function call overhead
        for direction in directions:
            nx = direction[0] + x
            ny = direction[1] + y
            if not (0 <= nx < wrld.width() and 0 <= ny < wrld.height()):
                continue
            if (nx,ny) in blocked:
                continue 
            if wrld.wall_at(nx, ny):
                continue
            else:
                valid_moves.append((direction[0], direction[1]))
        if valid_moves:
            # print(f"valid moves are {valid_moves}")
            return valid_moves
        else:
            return [(0,0)]

    def move_entities(self,move,img_wrld):
        img_wrld.me(self).move(*move)
        # assume worst case that monster is moving toward us 100% of the time
        for index, monster_list in img_wrld.monsters.items():
            for monster in monster_list:
                xdist = img_wrld.me(self).x - monster.x 
                ydist = img_wrld.me(self).y - monster.y
                #dx calculation
                if xdist < 0: dx = -1
                elif xdist > 0: dx = 1
                else: dx = 0
                #dy calculation
                if ydist < 0: dy = -1
                elif ydist > 0: dy = 1
                else: dy = 0
                
                monster.move(dx,dy)
        new_wrld, events = img_wrld.next()
        return new_wrld 


    def Astar(self, pos, goal):
        start_time = time.perf_counter()
        
        # open_set stores tuples of (f_score, pos)
        open_set = []
        heapq.heappush(open_set, (self.heuristic(pos, goal), pos))
        
        came_from = {}
        g_score = {pos: 0}
        closed_set = set()

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current == goal:
                execution_time = time.perf_counter() - start_time
                self.time = execution_time
                return self.reconstruct_path(came_from, current)

            # Skip if we have already expanded this node
            if current in closed_set:
                continue
            closed_set.add(current)

            current_g = g_score[current]

            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g = current_g + 1

                # If this path to neighbor is strictly better than any previous one:
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_val = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_val, neighbor))

        return []  # No path found

    def heuristic(self, a, b):
        # Using Manhattan distance as heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, pos):
        x, y = pos
        neighbors = []
        explosion_cells = {(e.x, e.y) for e in self.wrld.explosions.values()}
        # Inline checks to avoid extra function call overhead
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x, y),
                       (x + 1, y + 1), (x - 1, y - 1),(x+1,y-1),(x-1,y+1)):
            if 0 <= nx < self.wrld.width() and 0 <= ny < self.wrld.height() and not self.wrld.wall_at(nx, ny) and not (nx,ny) in explosion_cells:
                neighbors.append((nx, ny))
        return neighbors
    
    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path

    def chebyshev_distance(self, point1, point2):
        return max(abs(a - b) for a, b in zip(point1, point2))

    #BOMBING CODE:
    def should_bomb(self, wrld,move) -> bool:
        # Never stack bombs
        if wrld.bombs:
            return False

        x, y = wrld.me(self).x, wrld.me(self).y

        # Reason 1: no path to the exit, and a wall is right next to me
        if self.Astar((x,y),wrld.exitcell):
            self.no_path_ticks = 0
        else:
            self.no_path_ticks += 1

        wall_adjacent = any(
            wrld.wall_at(x + dx, y + dy)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= x + dx < wrld.width() and 0 <= y + dy < wrld.height()
        )
        # Reason 1: blocked for several turns in a row AND touching a wall
        if self.no_path_ticks >= self.NO_PATH_THRESHOLD and wall_adjacent:
            return self._arm_bomb()


        # Reason 2: a monster is very close
        for mlist in wrld.monsters.values():
            for m in mlist:
                if max(abs(m.x - x), abs(m.y - y)) <= 2:
                    return self._arm_bomb()

        #reason 3: best move is to stay still, maybe try bombing to fix it? 
        displacement = self.net_displacement()
        if displacement < 2 and self.no_path_ticks > 0 and wall_adjacent:
            return True

        return False
    
    def _arm_bomb(self):
        self.no_path_ticks = 0
        return True
    
    def update_weight_category(self, category_name, new_weights, filename="q_learning_weights.json"):
        # 1. Load existing file if it exists, otherwise start with a fresh structure
        if os.path.exists(filename):
            with open(filename, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {"weights": {}}
        else:
            data = {"weights": {}}

        # 2. Make sure the "weights" dictionary exists inside the JSON
        if "weights" not in data:
            data["weights"] = {}

        # 3. Update only the specific category (e.g., "Monster")
        data["weights"][category_name] = new_weights

        # 4. Save the updated data back to the file with nice formatting (indent=2)
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def get_min_monster_dist(self,wrld, default=1000):
        """Chebyshev distance from me to the nearest monster in `wrld`.
        Measures from the position in `wrld`, so it works on simulated worlds."""
        me = wrld.me(self)
        if me is None:                  # character is dead/gone in this world
            return 0
        dists = [
            self.chebyshev_distance((m.x, m.y), (me.x, me.y))
            for mlist in wrld.monsters.values()
            for m in mlist
        ]
        return min(dists, default=default)
    
    def get_min_bomb_dist(self, wrld, default=1000):
        me = wrld.me(self)
        if me is None:
            return default
        mx, my = me.x, me.y
        dists = []
        for b in list(wrld.bombs.values()):
            dists.append(max(abs(b.x - mx), abs(b.y - my)))
        return min(dists, default=default)

    def get_min_explosion_dist(self, wrld, default=1000):
        """Chebyshev distance from me to the nearest explosion cell in `wrld`.
        Returns 0 only if I'm standing in an explosion (or dead);
        returns `default` (far away) when there is no explosion at all."""
        me = wrld.me(self)
        for event in wrld.events:
            if event.tpe == event.BOMB_HIT_CHARACTER:
                return 0

        dists = [
            self.manhattan_distance((e.x, e.y), (me.x, me.y))
            for e in wrld.explosions.values()
            if 0 <= e.x < wrld.width() and 0 <= e.y < wrld.height()
        ]
        return min(dists, default=default)

    def compute_distances(self, start_cell):
        """
        Computes shortest path distances from a start cell using existing class helpers.
        :param start_cell: Tuple of (x, y)
        :return: 2D NumPy array of distances (shape: height x width)
        """
        width = self.wrld.width()
        height = self.wrld.height()
        
        # Initialize distance array with -1 (unreachable/walls)
        # Shape is (height, width) so we index as [y, x]
        distances = np.full((height, width), -1, dtype=int)
        
        # Validate start position using your existing helper
        if not self.is_valid_move(start_cell):
            return distances  # Start is out of bounds or a wall

        sx, sy = start_cell
        queue = deque([start_cell])
        distances[sy, sx] = 0  # y = row, x = col
        
        while queue:
            curr_pos = queue.popleft()
            cx, cy = curr_pos
            curr_dist = distances[cy, cx]
            
            # Leverage your optimized get_neighbors method
            for nx, ny in self.get_neighbors(curr_pos):
                # Check if unvisited in our distance map
                if distances[ny, nx] == -1:
                    distances[ny, nx] = curr_dist + 1
                    queue.append((nx, ny))
                    
        return distances

    def simulate_explosion_danger_cells(self,wrld,blast_range=4,timer_threshold=3):
        cells = {(e.x, e.y) for e in wrld.explosions.values()}
        for b in wrld.bombs.values():
            if b.timer > timer_threshold:
                continue 
            cells.add((b.x, b.y))
            for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
                for i in range(1, blast_range + 1):
                    x, y = b.x + dx*i, b.y + dy*i
                    if not (0 <= x < wrld.width() and 0 <= y < wrld.height()):
                        break
                    cells.add((x, y))
                    if wrld.wall_at(x, y):   # blast hits the wall and stops there
                        break
        return cells


    def is_valid_move(self, pos):
        # Check if the position is within bounds and not a wall
        x, y = pos
        # Assuming wrld is accessible and has methods to check bounds and walls
        return (0 <= x < self.wrld.width()) and (0 <= y < self.wrld.height()) and not self.wrld.wall_at(pos[0], pos[1])

    def manhattan_distance(self,point1, point2):
        return sum(abs(p1 - p2) for p1, p2 in zip(point1, point2))

    def record_position(self):
        self.pos_history.append((self.x, self.y))

    def net_displacement(self):
        """How far I am from where I was 10 moves ago (Chebyshev)."""
        h = self.pos_history
        if len(h) < 2:
            return 0
        return max(abs(h[-1][0] - h[0][0]), abs(h[-1][1] - h[0][1]))