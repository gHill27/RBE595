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

        # load weights from q_learning_weights.json
        with open("q_learning_weights.json") as f:
            q_weights = json.load(f)["weights"]
            self.bomb_weights = q_weights["Bomb"]
            self.monster_weights = q_weights["Monster"]
            self.far_weights = q_weights["Far"]

        self.learning_rate = 0.01 # for Q-learning update step
        self.gamma = 0.9 # future rewards discount factor
        # probability for trying unexplored moves
        self.exploration_prob = 0.0
        # when to start considering monsters
        self.monster_avoid_distance = 5
        # feature for walls destroyed
        self.walls_destroyed = 0
        # current weight set
        self.weights = None
    
    def _is_bomb_active(self, wrld):
        return bool(wrld.bombs)

    def define_state(self, wrld, distances_matrix):
        old_state = self.state
        if self._is_bomb_active(wrld):
            self.state = State.Bomb
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
            if bomberman_dist_to_exit - mdist < 0: 
                self.state = State.Free
            elif abs(mdist - bomberman_dist_to_exit) <= 4: #close to monster!
                self.state = State.Monster
            elif mdist > 0:
                self.state = State.FarFromMonster

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
    


    def do(self, wrld):
        # Your code here
        self.wrld = wrld  # Store the world reference for use in A* algorithm
        distances_matrix = self.compute_distances(wrld.exitcell)
        self.define_state(wrld,distances_matrix)
        pos = (self.x, self.y)
        goal = wrld.exitcell
        self.score = wrld.scores["me"]
        self.danger_mask = self._compute_danger_mask(wrld, self.wrld.width(), self.wrld.height())
       
        name: str = None
        weight: dict = None
        #state logic:
        match self.state:
            case State.Free:
                self.path = self.Astar(pos,goal)[1:]
                if self.path:
                    self.move(*self.get_move())  # skip analysis and just move
                else:
                    print("ERROR NO PATH!!!")
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

        if not self.path or self.path[-1] != goal:  # Recalculate path if it's empty or goal has changed
            Astar_path = self.Astar(pos, goal)
            self.path = Astar_path[1:]  # Skip the first position since it's the current position
        else:
            print("self.should bomb: " + str(self.should_bomb(wrld,pos)))
            if self.should_bomb(wrld,pos):
                self.place_bomb()
            else:
                move = self.get_move()  # Move according to the next step in the path
                # if move == (0, 0) and self.should_bomb(wrld,pos):
                self.move(*move)
                self.update_weights(name)
            self.set_cell_color(pos[0], pos[1], Fore.GREEN)  # Set the color of the cell to green
            print(f"Current position: {pos}, Next position: {self.path[0] if self.path else 'None'}, Time taken for A*: {self.time:.6f} seconds")

    def get_move(self):
        print("self.state: " + str(self.state))
        if self.path:
            best_move = self.path.pop(0)
            if self.state == State.Free:
                # no need to look at Q-values if we have a free path to the exit
                next_step = best_move
                dx = next_step[0] - self.x
                dy = next_step[1] - self.y
                return dx, dy
        else:
            best_move = self.get_blocked_move((self.x, self.y),self.wrld.exitcell)
        Qs = []
        neighbors = self.get_neighbors((self.x, self.y))
        unexplored = []
        for move in neighbors:
            # compute Q-value
            features = self.get_state_features(self.wrld)
            Q = 0.0
            for idx, weight in enumerate(self.weights):
                Q += weight*(1/(features[idx]+1))
            
            if Q > max(Qs,default=0) and Q > 0:
                Qs.append(Q)
                print("taking best Q move")
                best_move = move
            # else:
            #     # add first Q value if Qs is empty
            #     Qs.append(Q)
            #     best_move = move
        # explore areas with no known Q-value with some probability exploration_prob
        if random.random() < self.exploration_prob:
            best_move = (self.x, self.y)
        if not best_move:
            best_move = (self.x, self.y)
        next_step = best_move
        dx = next_step[0] - self.x
        dy = next_step[1] - self.y
        return dx, dy

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
                       (x + 1, y + 1), (x - 1, y - 1)):
            if 0 <= nx < self.wrld.width() and 0 <= ny < self.wrld.height() and not self.wrld.wall_at(nx, ny):
                neighbors.append((nx, ny))
        return neighbors
    
    def is_valid_move(self, pos):
        # Check if the position is within bounds and not a wall
        x, y = pos
        # Assuming wrld is accessible and has methods to check bounds and walls
        return (0 <= x < self.wrld.width()) and (0 <= y < self.wrld.height()) and not self.wrld.wall_at(pos[0], pos[1])

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path

    def manhattan_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_state_features(self, wrld):
        current_world = SensedWorld.from_world(wrld)
        next_world, events = current_world.next()

        feature_monster_dist = self.get_min_monster_dist(self.wrld)

        bomb_distances = []
        # feature for distance to closest bomb within 2 timesteps of explosion
        for bomb in wrld.bombs.values():
            timer = getattr(bomb, 'timer', 2)
            if timer <= 5:
                bomb_distances.append(self.manhattan_distance([bomb.x,bomb.y], [self.x,self.y]))

        feature_min_bomb_dist = min(bomb_distances,default=0)

        expl_distances = []
        # feature for distance to explosion
        for expl in wrld.explosions.values():
            if 0 <= expl.x < self.wrld.width() and 0 <= expl.y < self.wrld.height():
                expl_distances.append(self.manhattan_distance([expl.x,expl.y], [self.x,self.y]))

        feature_min_expl_dist = min(expl_distances,default=0)

        adj_walls = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < wrld.width() and 0 <= ny < wrld.height() and wrld.wall_at(nx, ny):
                    adj_walls.append((nx, ny))
        # not using this feature anymore\
        # feature_adjacent_walls = len(adj_walls)

        # not using this feature anymore\
        # feature_walls_destroyed = self.walls_destroyed

        feature_danger_mask = int(self.danger_mask[(self.y),(self.x)])
        exit_location = self.wrld.exitcell
        exit_dist = self.manhattan_distance([exit_location[0],exit_location[1]], [self.x,self.y])
        feature_exit_dist = exit_dist
        features = [feature_exit_dist, feature_min_bomb_dist, feature_monster_dist]
        # print("feature_exit_dist: " + str(feature_exit_dist))
        # print("feature_min_bomb_dist: " + str(feature_min_bomb_dist))
        # print("feature_min_expl_dist: " + str(feature_min_expl_dist))
        # print("feature_monster_dist: " + str(feature_monster_dist))
        # print("feature_danger_mask: " + str(feature_danger_mask))

        # return a list of the features
        return features

    #BOMBING CODE:
    def should_bomb(self,wrld,pos) -> bool:
        """ Places a bomb if there is no path to the exit, a bomb would clear a new path to the exit that avoids the monster, or a monster is blocking the path"""
        retval = False
        if not self.path:
            retval = True #should place bomb no path to the exit  
        monsters = []
        for m in wrld.monsters.values():
            monsters.append(m[0].x)
            monsters.append(m[0].y)

        if self.get_blocked_move(pos,wrld.exitcell) == (0,0) : #CHANGE TO BOMB A NEARBY WALL IF ITS IN THE WAY
            retval = True
        monster_current_distances = []
        if wrld.monsters:
            # compute list of monster distances for this step of the game
            for m_current in wrld.monsters.values():
                monster_current_distances.append(self.manhattan_distance([m_current[0].x,m_current[0].y], [self.x,self.y]))
            min_monster_distance = 0
            for dist in monster_current_distances:
                if not min_monster_distance:
                    # initialize first monster
                    min_monster_distance = dist
                if dist < min_monster_distance:
                    min_monster_distance = dist
            if min_monster_distance < self.monster_avoid_distance:
                retval = True

        if retval:
            print('placing bomb!')
        
        return retval

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

    def _compute_danger_mask(self, wrld, width, height):
            danger = np.zeros((height, width), dtype=bool)

            for expl in wrld.explosions.values():
                if 0 <= expl.x < width and 0 <= expl.y < height:
                    danger[expl.y, expl.x] = True

            for bomb in wrld.bombs.values():
                timer = getattr(bomb, 'timer', 2)
                if timer <= 2:
                    self._mask_blast(wrld, bomb.x, bomb.y, width, height, danger)

            return danger

    def _mask_blast(self, wrld, bx, by, width, height, danger):
        danger[by, bx] = True
        expl_range = getattr(wrld, 'expl_range', 4)
        for dx, dy in ((-1,0), (1,0), (0,-1), (0,1)):
            for r in range(1, expl_range + 1):
                nx, ny = bx + dx * r, by + dy * r
                if not (0 <= nx < width and 0 <= ny < height):
                    break
                danger[ny, nx] = True
                if wrld.wall_at(nx, ny):
                    break

    def update_weights(self, name):
        # calculate reward gained from the move
        reward = self.score - self.wrld.scores["me"]
        self.score = self.wrld.scores["me"]

        # sample Q-values from the list of possible moves
        # and choose the action with the highest value
        features = self.get_state_features(self.wrld)
        Q = 0.0
        for idx, weight in enumerate(self.weights):
            Q += weight*(1/(features[idx]+1))

        # Update weights
        current_world = SensedWorld.from_world(self.wrld)
        next_world, _ = current_world.next()
        next_reward = next_world.scores["me"] - self.score
        delta = (reward + self.gamma*next_reward) - Q
        print("reward: " + str(reward))
        print("next_reward: " + str(next_reward))
        print("delta: " +str(delta))
        for idx, weight in enumerate(self.weights):
            self.weights[idx] = weight+self.learning_rate*delta*features[idx]
        self.update_weight_category(name,self.weights) #updates the respective name in the json

    def get_blocked_move(self,pos,goal):
        distance_to_goal = math.dist(pos,goal)
        best = pos
        best_dist = distance_to_goal
        for neighbor in self.get_neighbors(pos):
            if math.dist(neighbor,goal) < best_dist:
                best = neighbor
                best_dist = math.dist(neighbor,goal)
        
        return best

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

    def get_min_monster_dist(self,current_world):

        monster_current_distances = []
        if current_world.monsters:
            # compute list of monster distances for this step of the game
            for m_current in current_world.monsters.values():
                monster_current_distances.append(self.manhattan_distance([m_current[0].x,m_current[0].y], [self.x,self.y]))
        min_monster_distance = 1000
        for dist in monster_current_distances:
            if not min_monster_distance:
                # initialize first monster
                min_monster_distance = dist
            if dist < min_monster_distance:
                min_monster_distance = dist
        # if min_monster_distance > self.monster_avoid_distance:
        #     min_monster_distance = 0
        return min_monster_distance