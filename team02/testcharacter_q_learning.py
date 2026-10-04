# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from sensed_world import SensedWorld
from colorama import Fore, Back
from events import Event
from collections import deque
import heapq
import time
import json
import numpy as np
import random


class TestCharacter(CharacterEntity):
    EXIT_BONUS = 10000.0
    DEATH_PENALTY = -10000.0
    DESTROY_WALL_REWARD = 10.0
    W_EXIT_DIST = -8.0

    def do(self, wrld):
        # Your code here
        self.wrld = wrld  # Store the world reference for use in A* algorithm
        pos = (self.x, self.y)
        goal = wrld.exitcell
        self.score = wrld.scores["me"]
        self.exit_dist_grid = self._compute_exit_grid(wrld, wrld.exitcell, self.wrld.width(), self.wrld.height())
        self.danger_mask = self._compute_danger_mask(wrld, self.wrld.width(), self.wrld.height())
        
        move = self.get_move()  # Move according to the next step in the path
        if move == (0, 0):
            self.place_bomb()
        else:
            self.move(*move)

        # calculate reward gained from the move
        reward = wrld.scores["me"] - self.score
        self.score = wrld.scores["me"]
        print("wrld.scores[me]: " + str(wrld.scores["me"]))
        print("self.score: " + str(self.score))
        print("reward: " + str(reward))
        # Update Q-values with the latest move's reward
        features = self.get_state_features(wrld)
        Q = 0
        for idx, weight in enumerate(self.q_weights):
            Q += weight*(1/(features[idx]+1))
        print("exit_dist: " + str(features[0]))
        # print("feature walls destroyed: " + str(features[1]))
        print("feature_min_bomb_dist: " + str(features[1]))
        print("feature_min_expl_dist: " + str(features[2]))
        print("feature_monster_dist: " + str(features[3]))
        # print("feature_adjacent_walls: " + str(features[4]))

        state_action_pair = self.get_state_features(wrld)
        action = move
        # add action to state
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
        self.learning_rate = 0.01 # for Q-learning update step
        self.gamma = 0.9 # future rewards discount factor
        # load q-table from q_table.json
        with open("q_table.json") as f:
            self.q_table = json.load(f)
        # probability for trying unexplored moves
        self.exploration_prob = 0.05
        # when to start avoiding monsters
        self.monster_avoid_distance = 10
        # feature for walls destroyed
        self.walls_destroyed = 0

    def get_move(self):
        best_move = None
        Qs = []
        neighbors = self.get_neighbors((self.x, self.y))
        unexplored = []
        for move in neighbors:
            # look up Q-value via state-action pair
            state_action_pair = self.get_state_features(self.wrld)
            action = move
            # add action to state
            state_action_pair.append(action[0])
            state_action_pair.append(action[1])
            state_action_pair = "".join(map(str,state_action_pair))

            try:
                # find Q-value
                if self.q_table[state_action_pair]:
                    Q = self.q_table[state_action_pair]
                    print("chose best Q-valued move")
                    if Q > max(Qs,default=0) and Q > 0:
                        Qs.append(Q)
                        best_move = move
                    # else:
                    #     # add first Q value if Qs is empty
                    #     Qs.append(Q)
                    #     best_move = move
            except:
                # key error means the Q-table doesn't have a value for that state-action pair
                unexplored.append(move)
                continue
        # explore areas with no known Q-value with some probability exploration_prob
        if random.random() < self.exploration_prob:
            if unexplored:
                print("random unexplored move")
                best_move = unexplored[int(random.random()*(len(unexplored)))]
        if not best_move:
            print("random move")
            best_move = neighbors[int(random.random()*(len(neighbors)))]
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
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x, y)):
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
        if current_world.monsters:
            # compute list of monster distances for this step of the game
            for m_current in current_world.monsters.values():
                monster_current_distances.append(self.manhattan_distance([m_current[0].x,m_current[0].y], [self.x,self.y]))
        min_monster_distance = 0
        for dist in monster_current_distances:
            if not min_monster_distance:
                # initialize first monster
                min_monster_distance = dist
            if dist < min_monster_distance:
                min_monster_distance = dist
        if min_monster_distance > self.monster_avoid_distance:
            min_monster_distance = 0


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

        feature_danger_mask = int(self.danger_mask[self.x,self.y])
        feature_monster_dist = min_monster_distance
        exit_location = self.wrld.exitcell
        exit_dist = self.manhattan_distance([exit_location[0],exit_location[1]], [self.x,self.y])
        feature_exit_dist = exit_dist
        features = monster_current_distances
        features = [feature_exit_dist, feature_min_bomb_dist, feature_min_expl_dist, feature_monster_dist, feature_danger_mask]

        # return a list of the features
        return features

# Event Evaluation & Ownership Checks
    def _evaluate_events(self, next_world, events):
        """
        Scans all events in the tick with ownership verification.
        Returns: (event_score, is_terminal)
        """
        total = 100.0
        for e in events:
            owner_name = getattr(e.character, 'name', None)
            mine = (owner_name == self.name)

            if e.tpe == Event.CHARACTER_FOUND_EXIT and mine:
                return self.EXIT_BONUS, True

            if e.tpe == Event.CHARACTER_KILLED_BY_MONSTER and mine:
                return self.DEATH_PENALTY, True

            if e.tpe == Event.BOMB_HIT_CHARACTER:
                victim_name = getattr(e.other, 'name', None)
                if victim_name == self.name:
                    return self.DEATH_PENALTY, True
                elif mine:
                    total += 5000.0

            elif e.tpe == Event.BOMB_HIT_MONSTER and mine:
                total += 50.0

            elif e.tpe == Event.BOMB_HIT_WALL and mine:
                self.walls_destroyed += 1
                total += self.DESTROY_WALL_REWARD

        return total, False

    def Q_value_update(self, reward, learning_rate, weights, wrld, gamma):
        features = self.get_state_features(wrld)
        if features:
            Q = 0
            for idx, weight in enumerate(self.q_weights):
                Q += weight*(1/(features[idx]+1))
            current_world = SensedWorld.from_world(wrld)
            next_world, events = current_world.next()
            event_score, terminal = self._evaluate_events(next_world, events)
            next_reward = 0.0
            if terminal:
                next_reward = event_score
            if next_world.me(self) is None:
                next_reward = self.DEATH_PENALTY
            # next_reward = float(self._heuristic(next_world, self.wrld.exitcell))
            print("heuristic: " + str(self.heuristic))
            delta = (reward + gamma*next_reward) - Q
            for idx, weight in enumerate(weights):
                weights[idx] = weight+learning_rate*delta*features[idx]
        print("reward: " + str(reward))
        print("next_reward: " + str(next_reward))
        print("Q-value: " + str(Q))
        # Update weights
        self.q_weights = weights

        

        # update weights in json
        update = dict()
        update["weights"] = self.q_weights
        with open("q_learning_weights.json", "w") as f:
            json.dump(update, f)

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

    # Optimistic Bomb Logic
    def _is_bomb_active(self, wrld):
        return bool(wrld.bombs)
    
    def _heuristic(self, wrld, exit_pos):
        me = wrld.me(self)
        if me is None:
            return self.DEATH_PENALTY
        
        exit_dist = self.exit_dist_grid[me.y, me.x]
        score = self.W_EXIT_DIST * exit_dist

        mx, my = me.x, me.y
        for mlist in wrld.monsters.values():
            for m in mlist:
                step_dist = max(abs(mx - m.x), abs(my - m.y))
                if step_dist <= 1:
                    score -= 50000.0
                elif step_dist == 2:
                    score -= 8000.0
                elif step_dist == 3:
                    score -= 1000.0

        if self.danger_mask[me.y, me.x]:
            score -= 20000.0

        return score

    def _compute_exit_grid(self, wrld, exit_pos, width, height):
        grid = np.full((height, width), 999.0, dtype=np.float32)
        if exit_pos is None:
            return grid

        grid[exit_pos[1], exit_pos[0]] = 0.0
        queue = deque([exit_pos])

        while queue:
            cx, cy = queue.popleft()
            d = grid[cy, cx]

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if not wrld.wall_at(nx, ny) and grid[ny, nx] == 999.0:
                            grid[ny, nx] = d + 1.0
                            queue.append((nx, ny))
        return grid