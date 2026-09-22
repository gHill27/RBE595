# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
from sensed_world import SensedWorld
import heapq
import random
import time

class TestCharacter(CharacterEntity):

    def do(self, wrld):
        # Your code here
        self.wrld = wrld  # Store the world reference for use in A* algorithm
        pos = (self.x, self.y)
        goal = wrld.exitcell
        if not self.path or self.path[-1] != goal:  # Recalculate path if it's empty or goal has changed
            Astar_path = self.Astar(pos, goal)
            self.path = Astar_path[1:]  # Skip the first position since it's the current position
        if self.path:
            (best_action, reward) = self.find_n_depth_best_next_move_recursive(None, 0, 10, wrld)
            if best_action:
                print("best move: " + str((self.x+best_action[0], self.y+best_action[1])))
                print("reward: " + str(reward))
                if best_action == (0,0):
                    self.place_bomb()
                    move = (0,0)
                else:
                    move = best_action
                    self.move(move[0], move[1])
                    self.set_cell_color(pos[0], pos[1], Fore.GREEN)  # Set the color of the cell to green
                # recompute A* now that we have deviated from the original path
                Astar_path = self.Astar((self.x+best_action[0], self.y+best_action[1]), goal)
                self.path = Astar_path[1:]
            else:
                move = self.get_move()
                self.move(move[0], move[1])  # Move according to the next step in the path
                self.set_cell_color(pos[0], pos[1], Fore.GREEN)  # Set the color of the cell to green
            print(f"Current position: {pos}, Next position: {(self.x+move[0], self.y+move[1]) if self.path else 'None'}, Time taken for A*: {self.time:.6f} seconds")

    def __init__(self, name, avatar, x, y):
        super().__init__(name, avatar, x, y)
        self.wrld = None  # Initialize world reference
        self.path = []  # Initialize path list
        self.time = 0  # Initialize time variable

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

    def find_next_turn_move_rewards(self, wrld):
        # copy current world
        world_copy = SensedWorld.from_world(wrld)
        # view moves on copy of self to determine best move to take
        self_copy = world_copy.me(self)
        moves = [(1,1), (1,0), (1,-1), (0,1), (0,0),
                    (0,-1), (-1,1), (-1,0), (-1,-1)]
        move_rewards = []
        for move in moves:
            if not self_copy:
                next
            elif not self.is_valid_move((self_copy.x+move[0], self_copy.y+move[1])):
                next
            elif move[0] == 0 and move[1] == 0:
                print("placing")
                self_copy.place_bomb()
            else:
                self_copy.move(move[0],move[1])
            (next_world_copy, events) = world_copy.next()
            reward = next_world_copy.scores["me"] - world_copy.scores["me"]
            # staying on the optimal A* path is ideal
            if (self_copy.x+move[0], self_copy.y+move[1]) in self.path:
                # keep following path
                reward += 1
            move_rewards.append((move, reward))
        # return a list of moves with associated rewards
        return move_rewards

    def find_n_depth_best_next_move(self, n, wrld):
        # copy current world
        world_copy = SensedWorld.from_world(wrld)
        # view moves on copy of self to determine best move to take
        self_copy = world_copy.me(self)
        n_depth_move_rewards = []
        moves = [(1,1), (1,0), (1,-1), (0,1),
                (0,-1), (-1,1), (-1,0), (-1,-1)]
        for (move_1, reward_1) in self.find_next_turn_move_rewards(world_copy):
            if move_1 == (0,0):
                self_copy.place_bomb()
            else:
                self_copy.move(move_1[0],move_1[1])
            (world_copy, events) = world_copy.next()
            for _ in range(n):
                for move_n in moves:
                    if not self_copy:
                        next
                    elif not self.is_valid_move((self_copy.x+move_n[0], self_copy.y+move_n[1])):
                        next
                    elif move_n == (0,0):
                        self_copy.place_bomb()
                    else:
                        self_copy.move(move_n[0],move_n[1])
                    (next_world_copy, events) = world_copy.next()

                    new_reward = next_world_copy.scores["me"] - world_copy.scores["me"]
                    if new_reward == 0:
                        reward_1 -= 1000
                        break
                    else:
                        reward_1 += new_reward
                    # step the world by another action
                    world_copy = next_world_copy

            n_depth_move_rewards.append((move_1, reward_1))
        best_reward = 0
        best_move = None
        for (move, reward) in n_depth_move_rewards:
            if reward > best_reward:
                best_reward = reward
                best_move = move
        return (best_move, best_reward)

    def find_n_depth_best_next_move_recursive(self, current_best_move, accumulative_reward, n, wrld):
        if n == 1:
            best_reward = 0
            best_move = None
            for (move, reward) in self.find_next_turn_move_rewards(wrld):
                if reward > best_reward:
                    best_reward = reward
                    best_move = move
                    return (best_move, best_reward+accumulative_reward)
        else:
            # copy current world
            world_copy = SensedWorld.from_world(wrld)
            # view moves on copy of self to determine best move to take
            self_copy = world_copy.me(self)
            moves = [(1,1), (1,0), (1,-1), (0,1), (0,0),
                     (0,-1), (-1,1), (-1,0), (-1,-1)]
            for move in moves:
                if not self_copy:
                    return (current_best_move, accumulative_reward)
                elif not self.is_valid_move((self_copy.x+move[0], self_copy.y+move[1])):
                    return (current_best_move, accumulative_reward)
                elif move[0] == 0 and move[1] == 0:
                    print("placing")
                    self_copy.place_bomb()
                else:
                    self_copy.move(move[0],move[1])
                (next_world_copy, events) = world_copy.next()
                reward = next_world_copy.scores["me"] - world_copy.scores["me"]
                if reward == 0:
                    accumulative_reward = 0
                # staying on the optimal A* path is ideal
                if (self_copy.x+move[0], self_copy.y+move[1]) in self.path:
                    # keep following path
                    reward += 1
                accumulative_reward = reward
                next_best = self.find_n_depth_best_next_move_recursive(move, accumulative_reward, n-1, next_world_copy)
                current_best = accumulative_reward
                if current_best > next_best[1]:
                    return (move, current_best)
                else:
                    return next_best