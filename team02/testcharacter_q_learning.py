# This is necessary to find the main code
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
        self.LEARNING_RATE = 0.01
        self.EXPLORATION_PROB = 0.0
        self.MONSTER_AVOID_RADUIS = 2
        self.dist = None

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
        if self._is_bomb_active(wrld):
            self.state = State.Bomb
            print(f'old state {old_state} --> {self.state}')
            return
        
        bomberman_dist_to_exit = distances_matrix[self.y][self.x] #row col
        all_monsters = [
            (m.x, m.y) for mlist in wrld.monsters.values() for m in mlist
        ]
        monster_distances = []
        
        for monster in all_monsters:
            monster_distances.append(distances_matrix[monster[1]][monster[0]]) #row col
        
        mdist = self.get_min_monster_dist(self.wrld)
        if mdist:
            if bomberman_dist_to_exit - mdist < 0 and bomberman_dist_to_exit > 0: 
                self.state = State.Free
            elif abs(mdist - bomberman_dist_to_exit) <= self.MONSTER_AVOID_RADUIS: #close to monster!
                self.state = State.Monster
            elif mdist > 0:
                self.state = State.FarFromMonster

        print(f'old state {old_state} --> {self.state}')

    def do(self, wrld):
        # Your code here
        self.wrld = wrld  # Store the world reference for use in A* algorithm
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
                else:
                    self.state = State.FarFromMonster #TODO actually make this smarter maybe
                return

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
        terminal = sim_s_prime.me(self) is None
        exited = any(e.tpe == Event.CHARACTER_FOUND_EXIT and e.character.name == self.name
             for e in sim_s_prime.events)
        died = terminal and not exited

        
        reward = -1*sim_s_prime.scores[self.name] - -1*wrld.scores[self.name]

        if died:
            reward -= 1000
        # reward = reward/100.0 # scaling so it doesnt get massive weights 
        if terminal:
            delta = reward - Q_s_a

        else:
            delta = reward + self.argmaxQ(sim_s_prime)[0]*self.GAMMA - Q_s_a

        features_s_a = self.computefeatures(sim_s_prime) 
        
        print(f'reward is {reward}')
        for index, weight in enumerate(self.weights):
            self.weights[index] = weight + self.LEARNING_RATE * delta * features_s_a[index]

        self.update_weight_category(name, self.weights)
        if self.should_bomb(wrld,pos):
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
   
    # def get_move(self,wrld):
    #     Qval, move = self.argmaxQ(wrld)
    #     return move


    # def update_weights(self, name, wrld):
    #     qvals = self.computefeatures(wrld)
    #     delta = self.calcDelta(wrld)
    #     for index, weight in enumerate(self.weights):
    #         self.weights[index] = weight + self.LEARNING_RATE*delta*qvals[index]
    #     self.update_weight_category(name,self.weights) #updates the respective name in the json
        

    # def calcDelta(self,wrld):
    #     Qprimemax, Qprimemove, bestwrld = self.argmaxQ(wrld)
    #     Qcurr = self.computeQfunction(wrld)
    #     reward = wrld.scores["me"] - self.score
    #     delta = reward + self.GAMMA*Qprimemax - Qcurr
    #     return delta
    
    def argmaxQ(self,wrld):
        pos = (wrld.me(self).x,wrld.me(self).y)
        moves = self.get_valid_moves(wrld) #returns dx dy
        bestQ = -math.inf
        bestmove = (0,0)
        bestwrld = wrld
        for move in moves:
            img_wrld = SensedWorld.from_world(wrld)   
            nxt = self.move_entities(move,img_wrld)
            if nxt.me(self) is None:
                exited = any(e.tpe == Event.CHARACTER_FOUND_EXIT and e.character.name == self.name
                            for e in nxt.events)
                currQ = 1e6 if exited else -1e6
            currQ = self.computeQfunction(nxt)
            if currQ > bestQ:
                bestQ = currQ
                bestmove = move
                bestwrld = nxt

        return (bestQ, bestmove, bestwrld)

    def computeQfunction(self,wrld):
        features = self.computefeatures(wrld)
        p1 = self.weights[0] * features[0]
        p2 = self.weights[1] * features[1]
        p3 = self.weights[2] * features[2]
        return p1 + p2 + p3

    def computefeatures(self,wrld):
        #get weights here and compute
        if wrld.me(self) is None:
            exited = any(e.tpe == Event.CHARACTER_FOUND_EXIT and e.character.name == self.name
                     for e in wrld.events)
            return (0.0, 1.0, 0.0) if exited else (1.0, 0.0, 1.0)
        
        f1 = 1/(1+self.get_min_monster_dist(wrld))
        #f2
        d = self.dist[wrld.me(self).y][wrld.me(self).x]
        if d <= 0:
            d = self.chebyshev_distance((wrld.me(self).x,wrld.me(self).y),wrld.exitcell)
        f2 = 1/(1+d)              
        f3 = 1/(1+self.get_min_explosion_dist(wrld))
        return (f1, f2, f3)

    def get_valid_moves(self,wrld):
        x = wrld.me(self).x
        y = wrld.me(self).y
        valid_moves = []
        directions = [(1,0),(-1,0),(0,1),(0,-1),(0,0),(1,1),(-1,-1),(1,-1),(-1,1)]
        # Inline checks to avoid extra function call overhead
        for direction in directions:
            nx = direction[0] + x
            ny = direction[1] + y
            if 0 <= nx < wrld.width() and 0 <= ny < wrld.height() and not wrld.wall_at(nx, ny):
                valid_moves.append((direction[0], direction[1]))
        return valid_moves

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
        # Inline checks to avoid extra function call overhead
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x, y),
                       (x + 1, y + 1), (x - 1, y - 1),(x+1,y-1),(x-1,y+1)):
            if 0 <= nx < self.wrld.width() and 0 <= ny < self.wrld.height() and not self.wrld.wall_at(nx, ny):
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
    def should_bomb(self, wrld, pos) -> bool:
        # Never stack bombs
        if wrld.bombs:
            return False

        x, y = pos

        # Reason 1: no path to the exit, and a wall is right next to me
        if not self.Astar(pos, wrld.exitcell):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    if (dx, dy) != (0, 0) and 0 <= nx < wrld.width() and 0 <= ny < wrld.height():
                        if wrld.wall_at(nx, ny):
                            return True

        # Reason 2: a monster is very close
        for mlist in wrld.monsters.values():
            for m in mlist:
                if max(abs(m.x - x), abs(m.y - y)) <= 2:
                    return True

        return False

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

    def get_min_monster_dist(self,wrld, default=100):
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
    
    # def get_min_bomb_dist(self,wrld):
    #     bomb_dists = []
    #     for bomb in wrld.bombs.values():
    #         timer = getattr(bomb, 'timer', 2)
    #         if timer <= 5:
    #             bomb_dists.append(self.chebyshev_distance([bomb.x,bomb.y], [wrld.me(self).x, wrld.me(self).y]))
        
    #     return min(bomb_dists,default=0)

    def get_min_explosion_dist(self, wrld, default=100):
        """Chebyshev distance from me to the nearest explosion cell in `wrld`.
        Returns 0 only if I'm standing in an explosion (or dead);
        returns `default` (far away) when there is no explosion at all."""
        me = wrld.me(self)
        for event in wrld.events:
            if event.tpe == self.BOMB_HIT_CHARACTER:
                return 0

        dists = [
            self.chebyshev_distance((e.x, e.y), (me.x, me.y))
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

    def is_valid_move(self, pos):
        # Check if the position is within bounds and not a wall
        x, y = pos
        # Assuming wrld is accessible and has methods to check bounds and walls
        return (0 <= x < self.wrld.width()) and (0 <= y < self.wrld.height()) and not self.wrld.wall_at(pos[0], pos[1])
