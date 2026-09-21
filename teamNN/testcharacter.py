# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
import heapq
import random
import time
import typing

class state:
    def __init(self, monster_pos, bomberman_pos, wrld):
        
        


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
            self.move(*self.get_move())  # Move according to the next step in the path
            self.set_cell_color(pos[0], pos[1], Fore.GREEN)  # Set the color of the cell to green
            print(f"Current position: {pos}, Next position: {self.path[0] if self.path else 'None'}, Time taken for A*: {self.time:.6f} seconds")

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


# def Expectimax(state): -> Action
#     return argmax(ExpVal(Result(state,action)))
# end
# # ------------------------------- 
# function Exp-value(state) returns a utility value
# if Terminal-Test(state) then return Utility(state)
# v ← 0
# for each a in Actions(state) do
# p ← Probability(a)
# v ← v + p · Max-value(Result(state, a))
# end for
# return v
# end function
# function Max-value(state) returns a utility value
# if Terminal-Test(state) then return Utility(state)
# v ← −∞
# for each a in Actions(state) do
# v ← Max(v, Exp-value(Result(state,a)))
# end for
# return v
# end function


    # def expectimax(state, depth, agent):
    #        # agent: 0 = (MAX), 1 = opponent (CHANCE)
    #         if depth == 0 or state.is_terminal():
    #             return evaluate(state)
            
    #         actions = state.get_legal_actions(agent)
    #         if not actions:
    #             return evaluate(state)
    
    #         if agent == 0:  # MAX node 
    #             best = float('-inf')
    #             for a in actions:
    #                 successor = state.generate_successor(agent, a)
    #                 value = expectimax(successor, depth, next_agent(agent))
    #                 best = max(best, value)
    #             return best
    
    #         else:  # CHANCE node — opponent's turn
    #             total = 0
    #             prob = 1 / len(actions)   # uniform random opponent
    #             for a in actions:
    #                 successor = state.generate_successor(agent, a)
    #                 value = expectimax(successor, depth - 1, next_agent(agent))
    #                 total += prob * value
    #             return total
    
    # def next_agent(agent):
    #     return 1 - agent  # alternate between MAX and CHANCE

    # def get_action(state, depth=3):
    #     best_action, best_value = None, float('-inf')
    #     for a in state.get_legal_actions(0):
    #         successor = state.generate_successor(0, a)
    #         value = expectimax(successor, depth, agent=1)
    #         if value > best_value:
    #             best_value = value
    #             best_action = a
    #     return best_action