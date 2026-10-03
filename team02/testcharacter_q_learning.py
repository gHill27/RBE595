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

class TestCharacter(CharacterEntity):

    def do(self, wrld):
        # Your code here
        self.wrld = wrld  # Store the world reference for use in A* algorithm
        pos = (self.x, self.y)
        goal = wrld.exitcell
        self.score = wrld.scores["me"]

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

            self.Q_value_update(reward, self.learning_rate, self.q_weights, wrld, self.gamma)

            # update q-table
            with open("q_table.json", "w") as f:
                json.dump(self.q_table, f)

            self.set_cell_color(pos[0], pos[1], Fore.GREEN)  # Set the color of the cell to green
            print(f"Current position: {pos}, Next position: {self.path[0] if self.path else 'None'}, Time taken for A*: {self.time:.6f} seconds")

    def __init__(self, name, avatar, x, y):
        super().__init__(name, avatar, x, y)
        self.wrld = None  # Initialize world reference
        self.path = []  # Initialize path list
        self.time = 0  # Initialize time variable
        self.score = 0 # Initialize score
        # load weights from q_learning_weights.json
        with open("q_learning_weights.json") as f:
            self.q_weights = json.load(f)["weights"]
        self.learning_rate = 0.9 # for Q-learning update step
        self.gamma = 0.9 # future rewards discount factor
        # load q-table from q_table.json
        with open("q_table.json") as f:
            self.q_table = json.load(f)
            
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

    def get_state_features(self, wrld):
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
        if len(features) < len(self.q_weights):
            features = features + [0]*(len(self.q_weights)-len(features))
        # return a list of the features
        return features

    def get_state(self, wrld):
        # state is the position of all of the entities in the game
        state = []
        for m in wrld.monsters.values():
            state.append(m[0].x)
            state.append(m[0].y)
        return state + [self.x,self.y]

    def Q_value_update(self, reward, learning_rate, weights, wrld, gamma):
        features = self.get_state_features(wrld)
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

        # Update weights
        self.q_weights = weights

        

        # update weights in json
        update = dict()
        update["weights"] = self.q_weights
        with open("q_learning_weights.json", "w") as f:
            json.dump(update, f)