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
        self.state: State = State.FarFromMonster #starts assuming it has a monster in the way far away.

        # load weights from q_learning_weights.json
        with open("q_learning_weights.json") as f:
            q_weights = json.load(f)["weights"]
            self.bomb_weights = q_weights["Bomb"]
            self.monster_weights = q_weights["Monster"]
            self.far_weights = q_weights["Far"]

        self.learning_rate = 0.9 # for Q-learning update step
        self.gamma = 0.9 # future rewards discount factor

        # load q-table from q_table.json
        with open("q_table.json") as f:
            self.q_table = json.load(f)
    
    def _is_bomb_active(self, wrld):
        return bool(wrld.bombs)

    def define_state(self, wrld, distances_matrix):
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
        
        for mdist in monster_distances: #TODO: Fix implementation to actually work with multiple monsters!!
            if bomberman_dist_to_exit - mdist < 0: 
                self.state = State.Free
            elif abs(mdist - bomberman_dist_to_exit) <= 4: #close to monster!
                self.state = State.Monster
            else:
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
    
    def get_blocked_move(self,pos,goal):
        distance_to_goal = self.manhattan_distance(pos,goal)
        for neighbor in self.get_neighbors(pos):
            if self.manhattan_distance(neighbor,goal) < distance_to_goal:
                return self.get_blocked_move()

    def do(self, wrld):
        # Your code here
        self.wrld = wrld  # Store the world reference for use in A* algorithm
        distances_matrix = self.compute_distances(wrld.exitcell)
        self.define_state(wrld,distances_matrix)
        pos = (self.x, self.y)
        goal = wrld.exitcell
        self.score = wrld.scores["me"]

       
        name: str = None
        weight: dict = None
        #state logic:
        match self.state:
            case State.Free:
                self.path = self.Astar(pos,goal)[1:]
                if self.path:
                    self.move(*self.get_move())  # skip analysis and just move
                else:
                    self.move(*self.get_blocked_move())
                    # print("ERROR NO PATH!!!")
                return

            case State.Bomb:
                weight = self.bomb_weights 
                name = "Bomb"

            case State.FarFromMonster:
                weight = self.far_weights
                name = "Far"

            case State.Monster:
                weight = self.monster_weights
                name = "Monster"
            
        if not self.path or self.path[-1] != goal:  # Recalculate path if it's empty or goal has changed
            Astar_path = self.Astar(pos, goal)
            self.path = Astar_path[1:]  # Skip the first position since it's the current position
        if self.path:
            self.move(*self.get_move())  # Move according to the next step in the path

            # calculate reward gained from the move
            reward = wrld.scores["me"] - self.score
            self.score = wrld.scores["me"]

            # sample Q-values from the list of possible moves
            # and choose the action with the highest value
            features = self.get_state_features(wrld)
            Q = 0
            for weight in self.q_weights:
                for feature_value in features:
                    Q += weight*feature_value

            state_action_pair = self.get_state(wrld)
            action = self.get_move()
            # add action to stat
            state_action_pair.append(action[0])
            state_action_pair.append(action[1])
            # q-table is a dictionary indexed by state-action pairs
            # convert state_action_list to a string
            state_action_pair = "".join(map(str,state_action_pair))
            # update q-table with state-action pair and its Q-value
            self.q_table[state_action_pair] = Q

            self.Q_value_update(reward, self.learning_rate, weight, wrld, self.gamma)
            self.update_weight_category(name,weight) #updates the respective name in the json
            # update q-table
            with open("q_table.json", "w") as f:
                json.dump(self.q_table, f)

            self.set_cell_color(pos[0], pos[1], Fore.GREEN)  # Set the color of the cell to green
            print(f"Current position: {pos}, Next position: {self.path[0] if self.path else 'None'}, Time taken for A*: {self.time:.6f} seconds")

            
    def get_move(self):
        if self.path:
            next_step = self.path.pop(0)
            dx = next_step[0] - self.x
            dy = next_step[1] - self.y
            return dx, dy
        else:
            return 0, 0  # No movement if path is empty

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
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
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

    def get_state_features(self, wrld, weights):
        current_world = SensedWorld.from_world(wrld)
        next_world, events = current_world.next()

        monster_current_distances = []
        monster_next_distances = []
        if current_world.monsters and next_world.monsters:
            # compute list of monster distances for this and the next step of the game
            for m_current in current_world.monsters.values():
                for m_next in next_world.monsters.values():
                    monster_current_distances.append(self.manhattan_distance([m_current[0].x,m_current[0].y], [self.x,self.y]))
                    monster_next_distances.append(self.manhattan_distance([m_next[0].x,m_next[0].y], [self.x,self.y]))

        # if the feature vector is not the same length as the q_weights vector then pad it with zeros
        # this will happen when there are variable number of monsters between game variants
        features = monster_current_distances+monster_next_distances
        if len(features) < len(weights):
            features = features + [0]*(len(weights)-len(features))
        # return a list of the features
        return features

    def get_state(self, wrld):
        # state is the position of all of the entities in the game
        state = []
        for m in wrld.monsters.values():
            state.append(m[0].x)
            state.append(m[0].y)
        return state + [self.x,self.y]

    def Q_value_update(self, reward, learning_rate, *weights, wrld, gamma):
        features = self.get_state_features(wrld, weights)
        if features:
            Q = 0
            for weight in weights:
                for feature_value in features:
                    Q += weight*feature_value
            current_world = SensedWorld.from_world(wrld)
            next_world, _ = current_world.next()
            next_reward = next_world.scores["me"] - self.score
            delta = (reward + gamma*next_reward) - Q
            for idx, weight in enumerate(weights):
                weights[idx] = weight+learning_rate*delta*features[idx]
        
    #BOMBING CODE:
    def should_bomb(self,wrld) -> bool:
        """ Places a bomb if there is no path to the exit, a bomb would clear a new path to the exit that avoids the monster, or a monster is blocking the path"""
        if not self.path:
            return True #should place bomb no path to the exit  
        monsters = []
        for m in wrld.monsters.values():
            monsters.append(m[0].x)
            monsters.append(m[0].y)

        if True: #CHANGE TO BOMB A NEARBY WALL IF ITS IN THE WAY
            pass


        for value in self.path[:5]:
            if value in monsters:
                return True

        
        pass




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

    
