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
    SEARCH_DEPTH = 10
    FAR_SEARCH_DEPTH = 1
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

        # 1. VECTORIZED PRECOMPUTATION
        self.exit_dist_grid = self._compute_exit_grid(wrld, exit_pos, w, h)
        self.danger_mask = self._compute_danger_mask(wrld, w, h)

        all_monsters = [
            (m.x, m.y) for mlist in wrld.monsters.values() for m in mlist
        ]

        
        # 2. IMMEDIATE WIN CHECK (Guarded against monsters camping exit)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = me.x + dx, me.y + dy
                if exit_pos and (nx, ny) == exit_pos:
                    # update_character_move checks monster collisions BEFORE exit cells
                    if not any(nx == mx and ny == my for mx, my in all_monsters):
                        self.move(dx, dy)
                        return

        # 3. OPTIMISTIC BOMB PLACEMENT
        if not self._is_bomb_active(wrld):
            if self._should_place_bomb_optimistic(wrld, me, exit_pos, all_monsters):
                escape_step = self._find_immediate_escape_step(wrld, pos, all_monsters)
                if escape_step is not None:
                    self.place_bomb()
                    self.move(*escape_step)
                    nx = max(0, min(w - 1, me.x + escape_step[0]))
                    ny = max(0, min(h - 1, me.y + escape_step[1]))
                    self.set_cell_color(nx, ny, Fore.MAGENTA + Back.RED)
                    return

        # 4. THREAT ASSESSMENT & MOVE FILTERING
        legal_moves = self._legal_character_moves(wrld, me.x, me.y)
        curr_min_m_dist = min((self._chebyshev(pos, m) for m in all_monsters), default=999)

        scored_moves = []
        for mv in legal_moves:
            nx, ny = me.x + mv[0], me.y + mv[1]

            if self.danger_mask[ny, nx]:
                continue
            if any(nx == mx and ny == my for mx, my in all_monsters):
                continue

            new_m_dist = min((self._chebyshev((nx, ny), m) for m in all_monsters), default=999)
            scored_moves.append((mv, new_m_dist))

        if not scored_moves:
            self.move(0, 0)
            return

        # Pure evasion override if actively threatened
        if curr_min_m_dist <= 5:
            max_dist_available = max(d for _, d in scored_moves)
            candidate_moves = [mv for mv, d in scored_moves if d == max_dist_available]
        else:
            safe_candidates = [mv for mv, d in scored_moves if d >= 3]
            candidate_moves = safe_candidates if safe_candidates else [mv for mv, _ in scored_moves]

        # 5. EXPECTIMAX WITH ADAPTIVE DEPTH
        depth = self._effective_depth(me, all_monsters)

        best_value = -math.inf
        best_moves = []

        for move in candidate_moves:
            sim_branch = SensedWorld.from_world(wrld)
            sim_me = sim_branch.me(self)
            sim_me.move(*move)

            (next_world_copy, events) = sim_branch.next()
            value = next_world_copy.scores["me"] - sim_branch.scores["me"]
            next_pos = [self.x+move[0], self.y+move[1]]
            value -= self._chebyshev(next_pos,exit_pos)
            value -= 4/(curr_min_m_dist*curr_min_m_dist)
            if value > best_value + 1e-5:
                best_value = value
                best_moves = [move]
            elif abs(value - best_value) <= 1e-5:
                best_moves.append(move)

        best_move = random.choice(best_moves) if best_moves else candidate_moves[0]
        self.move(*best_move)

        nx = max(0, min(w - 1, me.x + best_move[0]))
        ny = max(0, min(h - 1, me.y + best_move[1]))
        self.set_cell_color(nx, ny, Fore.WHITE + Back.BLUE)

    # Adaptive Depth
    def _effective_depth(self, me, all_monsters):
        if not all_monsters:
            return self.FAR_SEARCH_DEPTH
        nearest = min(self._chebyshev((me.x, me.y), m) for m in all_monsters)
        return self.SEARCH_DEPTH if nearest <= self.ENGAGE_RADIUS else self.FAR_SEARCH_DEPTH


    # NumPy Distance & Hazard Precomputation
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

    def _should_place_bomb_optimistic(self, wrld, me, exit_pos, all_monsters):
        pos = (me.x, me.y)

        min_m_dist = min((self._chebyshev(pos, m) for m in all_monsters), default=999)
        if min_m_dist <= 6:
            return True

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

        curr_dist = self.exit_dist_grid[me.y, me.x]
        if curr_dist >= 999:
            return True

        my_euclid_exit = self._chebyshev(pos, exit_pos) if exit_pos else 0
        for wx, wy in adj_walls:
            if exit_pos and self._chebyshev((wx, wy), exit_pos) < my_euclid_exit:
                return True

        return False

    def _find_immediate_escape_step(self, wrld, bomb_pos, all_monsters):
        w, h = wrld.width(), wrld.height()
        bx, by = bomb_pos

        candidates = []
        # Diagonal-only offsets guarantee escape from orthogonal blast lines
        for dx in (-1, 1):
            for dy in (-1, 1):
                nx, ny = bx + dx, by + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if not wrld.wall_at(nx, ny) and not wrld.bomb_at(nx, ny) and not wrld.explosion_at(nx, ny):
                        m_dist = min((self._chebyshev((nx, ny), m) for m in all_monsters), default=999)
                        if m_dist > 2:
                            candidates.append(((dx, dy), m_dist))

        if candidates:
            candidates.sort(key=lambda item: item[1], reverse=True)
            return candidates[0][0]
        return None

    # Expectimax Nodes
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
            event_score, terminal = self._evaluate_events(next_world, events)
            if terminal:
                return event_score

            if next_world.me(self) is None:
                return self.DEATH_PENALTY

            heuristic_val = event_score + self._heuristic(next_world, exit_pos)
            return heuristic_val if depth <= 1 else self._max(next_world, depth - 1, exit_pos)

        monster_dists = [
            self._get_monster_action_distribution(sim_wrld, (m.x, m.y), char_pos)
            for m in monsters
        ]

        joint_combinations = list(product(*monster_dists))
        total_val = 0.0
        prob_sum = 0.0

        for joint in joint_combinations:
            joint_prob = 1.0
            for _, prob in joint:
                joint_prob *= prob

            if joint_prob < 0.03:
                continue

            step_world = SensedWorld.from_world(sim_wrld)
            # Correctly updates monster commands using name match across cell buckets
            for m_orig, (mv, _) in zip(monsters, joint):
                m_target = self._find_matching_monster(step_world, m_orig)
                if m_target:
                    m_target.move(*mv)

            next_world, events = step_world.next()
            event_score, terminal = self._evaluate_events(next_world, events)

            if terminal:
                total_val += joint_prob * event_score
                prob_sum += joint_prob
                continue

            # Hard backstop for deaths not caught by custom event flags
            if next_world.me(self) is None:
                total_val += joint_prob * self.DEATH_PENALTY
                prob_sum += joint_prob
                continue

            if depth <= 1:
                total_val += joint_prob * (event_score + self._heuristic(next_world, exit_pos))
            else:
                total_val += joint_prob * (event_score + self._max(next_world, depth - 1, exit_pos))
            
            prob_sum += joint_prob

        return total_val / max(prob_sum, 1e-6)

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
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def _get_monster_action_distribution(self, wrld, monster_pos, char_pos):
        legal = self._legal_monster_moves(wrld, *monster_pos)
        if not legal:
            return [((0, 0), 1.0)]

        dist = self._chebyshev(monster_pos, char_pos)
        if dist > self.HUNT_RADIUS:
            prob = 1.0 / len(legal)
            return [(mv, prob) for mv in legal]

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

    # Event Evaluation & Ownership Checks
    def _evaluate_events(self, next_world, events):
        """
        Scans all events in the tick with ownership verification.
        Returns: (event_score, is_terminal)
        """
        total = 0.0
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
                total += 10.0

        return total, False

    # Grid Utilities
    def _find_exit(self, wrld):
        return wrld.exitcell

    def _legal_character_moves(self, wrld, x, y):
        w, h = wrld.width(), wrld.height()
        bomb_coords = {(b.x, b.y) for b in wrld.bombs.values()}
        expl_coords = {(e.x, e.y) for e in wrld.explosions.values()}

        moves = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if (nx, ny) not in bomb_coords and (nx, ny) not in expl_coords:
                        if not wrld.wall_at(nx, ny):
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
        """Finds target monster across all cell index buckets by unique name."""
        for mlist in target_world.monsters.values():
            for m in mlist:
                if m.name == source_monster.name:
                    return m
        return None