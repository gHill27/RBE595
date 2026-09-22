import sys
sys.path.insert(0, '../bomberman')

import math
import random
from collections import deque
from itertools import product

import numpy as np

from entity import CharacterEntity
from sensed_world import SensedWorld
from events import Event
from colorama import Fore, Back


class TestCharacter(CharacterEntity):
    SEARCH_DEPTH = 3
    ENGAGE_RADIUS = 4
    HUNT_RADIUS = 3

    W_EXIT_DIST = -8.0
    DEATH_PENALTY = -100000.0
    EXIT_BONUS = 100000.0

    def do(self, wrld):
        me = wrld.me(self)
        if me is None:
            return

        w, h = wrld.width(), wrld.height()
        exit_pos = self._find_exit(wrld)
        pos = (me.x, me.y)

        # -----------------------------------------------------------
        # 1. VECTORIZED PRECOMPUTATION (Runs once per tick)
        # -----------------------------------------------------------
        # Exact shortest-path grid to exit: O(1) lookup anywhere in tree
        self.exit_dist_grid = self._compute_exit_grid(wrld, exit_pos, w, h)
        # Boolean mask of lethal cells (active explosions + imminent bomb blasts)
        self.danger_mask = self._compute_danger_mask(wrld, w, h)

        # -----------------------------------------------------------
        # 2. IMMEDIATE WIN CHECK
        # -----------------------------------------------------------
        for dx, dy in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)):
            nx, ny = me.x + dx, me.y + dy
            if exit_pos and (nx, ny) == exit_pos:
                self.move(dx, dy)
                return

        # -----------------------------------------------------------
        # 3. OPTIMISTIC BOMB PLACEMENT
        # -----------------------------------------------------------
        if not self._is_bomb_active(wrld):
            if self._should_place_bomb_optimistic(wrld, me, exit_pos):
                escape_step = self._find_immediate_escape_step(wrld, pos)
                if escape_step is not None:
                    self.place_bomb()
                    self.move(*escape_step)
                    nx = max(0, min(w - 1, me.x + escape_step[0]))
                    ny = max(0, min(h - 1, me.y + escape_step[1]))
                    self.set_cell_color(nx, ny, Fore.MAGENTA + Back.RED)
                    return

        # -----------------------------------------------------------
        # 4. EXPECTIMAX LOOKAHEAD
        # -----------------------------------------------------------
        legal_moves = self._legal_character_moves(wrld, me.x, me.y)
        # Avoid stepping into immediate lethal danger
        safe_moves = [m for m in legal_moves if not self.danger_mask[me.y + m[1], me.x + m[0]]]
        candidate_moves = safe_moves if safe_moves else legal_moves

        best_value = -math.inf
        best_moves = []

        for move in candidate_moves:
            sim_branch = SensedWorld.from_world(wrld)
            sim_me = sim_branch.me(self)
            sim_me.move(*move)

            value = self._chance(sim_branch, self.SEARCH_DEPTH, exit_pos)
            if value > best_value + 1e-5:
                best_value = value
                best_moves = [move]
            elif abs(value - best_value) <= 1e-5:
                best_moves.append(move)

        best_move = random.choice(best_moves) if best_moves else (0, 0)
        self.move(*best_move)

        nx = max(0, min(w - 1, me.x + best_move[0]))
        ny = max(0, min(h - 1, me.y + best_move[1]))
        self.set_cell_color(nx, ny, Fore.WHITE + Back.BLUE)

    # ---------------------------------------------------------------
    # NumPy Distance & Hazard Precomputation
    # ---------------------------------------------------------------
    def _compute_exit_grid(self, wrld, exit_pos, width, height):
        """Single BFS flood-fill generating an exact distance array for the entire map."""
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

    def _compute_danger_mask(self, wrld, width, height):
        """Returns a 2D boolean NumPy array where True indicates lethal blast lines."""
        danger = np.zeros((height, width), dtype=bool)

        for x in range(width):
            for y in range(height):
                # 1. Active fire
                if wrld.explosion_at(x, y) is not None:
                    danger[y, x] = True

                # 2. Imminent bomb blasts (only lethal when fuse <= 2)
                bomb = wrld.bomb_at(x, y)
                if bomb is not None:
                    timer = getattr(bomb, 'timer', 2)
                    if timer <= 2:
                        self._mask_blast(wrld, x, y, width, height, danger)
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

    # ---------------------------------------------------------------
    # Optimistic Bomb Logic
    # ---------------------------------------------------------------
    def _is_bomb_active(self, wrld):
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.bomb_at(x, y) is not None:
                    return True
        return False

    def _should_place_bomb_optimistic(self, wrld, me, exit_pos):
        pos = (me.x, me.y)

        # 1. Do not bomb if a monster is already on top of us (dist <= 1)
        all_monsters = [
            (m.x, m.y) for mlist in wrld.monsters.values() for m in mlist
        ]
        min_m_dist = min((self._chebyshev(pos, m) for m in all_monsters), default=999)
        if min_m_dist <= 1:
            return False

        # 2. Must be adjacent to at least one soft wall
        adj_walls = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = me.x + dx, me.y + dy
                if 0 <= nx < wrld.width() and 0 <= ny < wrld.height() and wrld.wall_at(nx, ny):
                    adj_walls.append((nx, ny))
        if not adj_walls:
            return False

        # 3. TRIGGER CONDITIONS (Optimistic):
        # A. Blocked: path to exit is blocked
        curr_dist = self.exit_dist_grid[me.y, me.x]
        if curr_dist >= 999:
            return True

        # B. Shortcut: blowing up a wall opens a path that cuts distance to exit
        my_euclid_exit = self._chebyshev(pos, exit_pos) if exit_pos else 0
        for wx, wy in adj_walls:
            if exit_pos and self._chebyshev((wx, wy), exit_pos) < my_euclid_exit:
                return True

        # C. Ambush Trap: Monster is pursuing within 2 to 4 steps
        if 2 <= min_m_dist <= 4:
            return True

        return False

    def _find_immediate_escape_step(self, wrld, bomb_pos):
        """Finds any immediately adjacent legal move that ducks out of the bomb's cardinal rays."""
        w, h = wrld.width(), wrld.height()
        bx, by = bomb_pos

        # Diagonal steps naturally dodge orthogonal blast crosses
        # Cardinal steps behind corners also work
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = bx + dx, by + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if not wrld.wall_at(nx, ny) and not wrld.bomb_at(nx, ny) and not wrld.explosion_at(nx, ny):
                        # Moving diagonally gets out of the row & column line-of-sight
                        if dx != 0 and dy != 0:
                            return (dx, dy)
                        # Moving cardinally is safe if turning behind an obstruction
                        return (dx, dy)
        return None

    # ---------------------------------------------------------------
    # Expectimax Nodes
    # ---------------------------------------------------------------
    def _chance(self, sim_wrld, depth, exit_pos):
        sim_me = sim_wrld.me(self)
        if sim_me is None:
            return self.DEATH_PENALTY

        char_pos = (sim_me.x, sim_me.y)
        monsters = []
        for mlist in sim_wrld.monsters.values():
            for m in mlist:
                if self._chebyshev(char_pos, (m.x, m.y)) <= self.ENGAGE_RADIUS:
                    monsters.append(m)

        if not monsters:
            step_world = SensedWorld.from_world(sim_wrld)
            next_world, events = step_world.next()
            node_value = self._evaluate_events(next_world, events)
            if node_value is not None:
                return node_value
            if next_world.me(self) is None:
                return self.EXIT_BONUS
            return self._heuristic(next_world, exit_pos) if depth <= 1 else self._max(next_world, depth - 1, exit_pos)

        # Behavioral monster move distribution (Hunting vs. Wandering)
        monster_dists = [
            self._get_monster_action_distribution(sim_wrld, (m.x, m.y), char_pos)
            for m in monsters
        ]

        joint_combinations = list(product(*monster_dists))
        total_val = 0.0

        for joint in joint_combinations:
            joint_prob = 1.0
            for _, prob in joint:
                joint_prob *= prob

            # Prune highly improbable branches to save simulation time
            if joint_prob < 0.03:
                continue

            step_world = SensedWorld.from_world(sim_wrld)
            for m_orig, (mv, _) in zip(monsters, joint):
                m_target = self._find_matching_monster(step_world, m_orig)
                if m_target:
                    m_target.move(*mv)

            next_world, events = step_world.next()

            node_value = self._evaluate_events(next_world, events)
            if node_value is not None:
                total_val += joint_prob * node_value
                continue

            if next_world.me(self) is None:
                total_val += joint_prob * self.EXIT_BONUS
                continue

            if depth <= 1:
                total_val += joint_prob * self._heuristic(next_world, exit_pos)
            else:
                total_val += joint_prob * self._max(next_world, depth - 1, exit_pos)

        return total_val

    def _max(self, sim_wrld, depth, exit_pos):
        sim_me = sim_wrld.me(self)
        if sim_me is None:
            return self.DEATH_PENALTY

        moves = self._legal_character_moves(sim_wrld, sim_me.x, sim_me.y)
        best = -math.inf

        for move in moves:
            branch = SensedWorld.from_world(sim_wrld)
            me_branch = branch.me(self)
            me_branch.move(*move)
            best = max(best, self._chance(branch, depth, exit_pos))

        return best

    def _chebyshev(self, a, b):
        """True step distance for 8-neighborhood grid motion (L_infinity norm)."""
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def _get_monster_action_distribution(self, wrld, monster_pos, char_pos):
        legal = self._legal_monster_moves(wrld, *monster_pos)
        if not legal:
            return [((0, 0), 1.0)]

        # 1-step hunting trigger uses Chebyshev distance
        dist = self._chebyshev(monster_pos, char_pos)
        if dist > self.HUNT_RADIUS:
            prob = 1.0 / len(legal)
            return [(mv, prob) for mv in legal]

        # Greedy pursuit in 8 directions
        move_dists = [
            (mv, self._chebyshev((monster_pos[0] + mv[0], monster_pos[1] + mv[1]), char_pos))
            for mv in legal
        ]
        min_d = min(d for _, d in move_dists)
        best_moves = [mv for mv, d in move_dists if d == min_d]

        prob = 1.0 / len(best_moves)
        return [(mv, prob) for mv in best_moves]

    def _heuristic(self, wrld, exit_pos):
        me = wrld.me(self)
        if me is None:
            return self.EXIT_BONUS

        # O(1) vectorized distance lookup
        exit_dist = self.exit_dist_grid[me.y, me.x]
        score = self.W_EXIT_DIST * exit_dist

        # Accurate 8-neighborhood threat penalties
        for mlist in wrld.monsters.values():
            for m in mlist:
                step_dist = self._chebyshev((me.x, me.y), (m.x, m.y))
                # True 1-step lethal threat (cardinal OR diagonal)
                if step_dist <= 1:
                    score -= 5000.0
                elif step_dist == 2:
                    score -= 400.0
                elif step_dist == 3:
                    score -= 60.0

        # Immediate penalty for stepping into an active blast
        if self.danger_mask[me.y, me.x]:
            score -= 8000.0

        return score

    def _evaluate_events(self, next_world, events):
        for e in events:
            if e.tpe == Event.CHARACTER_FOUND_EXIT:
                return self.EXIT_BONUS
            if e.tpe in (Event.CHARACTER_KILLED_BY_MONSTER, Event.BOMB_HIT_CHARACTER):
                return self.DEATH_PENALTY
            if e.tpe == Event.BOMB_HIT_MONSTER:
                return 5000.0
            if e.tpe == Event.BOMB_HIT_WALL:
                return 300.0
        return None


    # ---------------------------------------------------------------
    # Grid Utilities
    # ---------------------------------------------------------------
    def _find_exit(self, wrld):
        if hasattr(wrld, 'exitcell') and wrld.exitcell is not None:
            return wrld.exitcell
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.exit_at(x, y):
                    return (x, y)
        return None

    def _legal_character_moves(self, wrld, x, y):
        moves = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < wrld.width() and 0 <= ny < wrld.height():
                    if not wrld.wall_at(nx, ny) and not wrld.bomb_at(nx, ny) and not wrld.explosion_at(nx, ny):
                        moves.append((dx, dy))
        return moves if moves else [(0, 0)]

    def _legal_monster_moves(self, wrld, x, y):
        moves = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < wrld.width() and 0 <= ny < wrld.height():
                    if not wrld.wall_at(nx, ny):
                        moves.append((dx, dy))
        return moves if moves else [(0, 0)]

    def _find_matching_monster(self, target_world, source_monster):
        for mlist in target_world.monsters.values():
            for m in mlist:
                if m.x == source_monster.x and m.y == source_monster.y:
                    return m
        return None
