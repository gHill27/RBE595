import sys
sys.path.insert(0, '../bomberman')

import math
import heapq
from itertools import product
from functools import lru_cache

from entity import CharacterEntity
from sensed_world import SensedWorld
from events import Event
from colorama import Fore, Back


class TestCharacter(CharacterEntity):
    SEARCH_DEPTH = 2
    ENGAGE_RADIUS = 4

    W_EXIT_DIST = -8.0
    DEATH_PENALTY = -100000.0
    EXIT_BONUS = 100000.0

    def do(self, wrld):
        me = wrld.me(self)
        if me is None:
            return

        self._astar_dist.cache_clear()
        exit_pos = self._find_exit(wrld)

        # 1. IMMEDIATE WIN CHECK: If exit is 1 step away, take it immediately
        for move in self._legal_character_moves(wrld, me.x, me.y):
            nx, ny = me.x + move[0], me.y + move[1]
            if exit_pos and (nx, ny) == exit_pos:
                self.move(*move)
                return

        # 2. AGGRESSIVE OPPORTUNISTIC BOMB PLACEMENT
        if not self._is_bomb_active(wrld):
            if self._should_place_bomb_aggressive(wrld, me, exit_pos):
                self.place_bomb()

        # 3. EXPECTIMAX
        best_value = -math.inf
        best_move = (0, 0)

        legal_moves = self._legal_character_moves(wrld, me.x, me.y)
        for move in legal_moves:
            sim_branch = SensedWorld.from_world(wrld)
            sim_me = sim_branch.me(self)
            sim_me.move(*move)

            value = self._chance(sim_branch, self.SEARCH_DEPTH, exit_pos)
            if value > best_value:
                best_value = value
                best_move = move

        self.move(*best_move)

        nx = max(0, min(wrld.width() - 1, me.x + best_move[0]))
        ny = max(0, min(wrld.height() - 1, me.y + best_move[1]))
        self.set_cell_color(nx, ny, Fore.WHITE + Back.BLUE)

    # ---------------------------------------------------------------
    # Aggressive Opportunistic Bombing Logic
    # ---------------------------------------------------------------
    def _is_bomb_active(self, wrld):
        """Scans grid for active ticking bombs."""
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.bomb_at(x, y) is not None:
                    return True
        return False

    def _get_adjacent_walls(self, wrld, x, y):
        """Returns adjacent walls in 8 directions."""
        walls = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < wrld.width() and 0 <= ny < wrld.height():
                    if wrld.wall_at(nx, ny):
                        walls.append((nx, ny))
        return walls

    def _should_place_bomb_aggressive(self, wrld, me, exit_pos):
        pos = (me.x, me.y)

        # Baseline survival requirement: at least one escape step must exist right now
        escape_moves = [
            (dx, dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if (dx != 0 or dy != 0)
            and 0 <= me.x + dx < wrld.width()
            and 0 <= me.y + dy < wrld.height()
            and not wrld.wall_at(me.x + dx, me.y + dy)
            and not wrld.bomb_at(me.x + dx, me.y + dy)
            and not wrld.explosion_at(me.x + dx, me.y + dy)
        ]
        if not escape_moves:
            return False

        all_monsters = [
            (m.x, m.y) for mlist in wrld.monsters.values() for m in mlist
        ]
        min_monster_dist = min((self._manhattan(pos, m) for m in all_monsters), default=999)

        # Suicide guard: do not drop a bomb if a monster is directly adjacent (dist == 1)
        if min_monster_dist <= 1:
            return False

        adj_walls = self._get_adjacent_walls(wrld, me.x, me.y)

        # TRIGGER 1: Monster Ambush Gamble
        # If a monster is within 2 to 4 steps, drop a bomb behind/around us to hope it walks into the blast
        if 2 <= min_monster_dist <= 4:
            return True

        # TRIGGER 2: Demolition & Clearing
        # If any soft wall is adjacent, blow it up (unblocks shortcuts, destroys barriers, gains score)
        if adj_walls:
            return True

        # TRIGGER 3: Path to exit is blocked
        if exit_pos is not None:
            current_dist = self._astar_dist(wrld, pos, exit_pos)
            if current_dist >= 999:
                return True

        return False

    # ---------------------------------------------------------------
    # Expectimax using SensedWorld.next() and Events
    # ---------------------------------------------------------------
    def _chance(self, sim_wrld, depth, exit_pos):
        sim_me = sim_wrld.me(self)
        if sim_me is None:
            return self.DEATH_PENALTY

        monsters = []
        for mlist in sim_wrld.monsters.values():
            for m in mlist:
                if self._manhattan((sim_me.x, sim_me.y), (m.x, m.y)) <= self.ENGAGE_RADIUS:
                    monsters.append(m)

        move_options = [self._legal_monster_moves(sim_wrld, m.x, m.y) for m in monsters]
        joint_moves = list(product(*move_options)) if move_options else [()]

        total_val = 0.0
        prob = 1.0 / len(joint_moves)

        for joint in joint_moves:
            step_world = SensedWorld.from_world(sim_wrld)

            for m_orig, mv in zip(monsters, joint):
                m_target = self._find_matching_monster(step_world, m_orig)
                if m_target:
                    m_target.move(*mv)

            next_world, events = step_world.next()

            node_value = self._evaluate_events(next_world, events)
            if node_value is not None:
                total_val += prob * node_value
                continue

            if next_world.me(self) is None:
                total_val += prob * self.EXIT_BONUS
                continue

            if depth <= 1:
                total_val += prob * self._heuristic(next_world, exit_pos)
            else:
                total_val += prob * self._max(next_world, depth - 1, exit_pos)

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

    def _evaluate_events(self, next_world, events):
        for e in events:
            if e.tpe == Event.CHARACTER_FOUND_EXIT:
                return self.EXIT_BONUS
            if e.tpe in (Event.CHARACTER_KILLED_BY_MONSTER, Event.BOMB_HIT_CHARACTER):
                return self.DEATH_PENALTY
            # Reward successful bomb hits on monsters or walls in lookahead
            if e.tpe == Event.BOMB_HIT_MONSTER:
                return 5000.0
            if e.tpe == Event.BOMB_HIT_WALL:
                return 300.0
        return None

    def _heuristic(self, wrld, exit_pos):
        me = wrld.me(self)
        if me is None:
            return self.EXIT_BONUS

        score = 0.0
        pos = (me.x, me.y)

        # 1. Distance to exit
        if exit_pos is not None:
            exit_dist = self._astar_dist(wrld, pos, exit_pos)
            score += self.W_EXIT_DIST * exit_dist

        # 2. Distance to nearest monster
        all_monsters = [
            (m.x, m.y) for mlist in wrld.monsters.values() for m in mlist
        ]
        nearest_monster_dist = 999
        if all_monsters:
            nearest_monster_dist = min(self._astar_dist(wrld, pos, m) for m in all_monsters)
            if nearest_monster_dist <= 1:
                score -= 2000.0
            elif nearest_monster_dist <= 2:
                score -= 300.0
            elif nearest_monster_dist <= 3:
                score -= 50.0

        # 3. Anti-Cornering / Mobility Check
        if nearest_monster_dist <= 5:
            open_exits = self._open_neighbor_count(wrld, pos[0], pos[1])
            if open_exits <= 2:
                score -= 150.0
            elif open_exits <= 3:
                score -= 40.0
            else:
                score += 10.0 * open_exits

            local_volume = self._reachable_space(wrld, pos, limit=7)
            if local_volume < 5:
                score -= 300.0 / max(1, local_volume)

        # 4. Flee active bombs and ticking blasts
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                bomb = wrld.bomb_at(x, y)
                if bomb is not None:
                    # In direct crosshairs
                    if (pos[0] == x or pos[1] == y) and self._manhattan(pos, (x, y)) <= 4:
                        timer = getattr(bomb, 'timer', 2)
                        score -= 3000.0 / max(1, timer)

        return score

    # ---------------------------------------------------------------
    # Grid Utilities & Spatial Analysis
    # ---------------------------------------------------------------
    def _open_neighbor_count(self, wrld, x, y):
        count = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < wrld.width() and 0 <= ny < wrld.height():
                    if not wrld.wall_at(nx, ny) and not wrld.bomb_at(nx, ny):
                        count += 1
        return count

    def _reachable_space(self, wrld, start, limit=8):
        queue = [start]
        visited = {start}
        while queue and len(visited) < limit:
            curr = queue.pop(0)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = curr[0] + dx, curr[1] + dy
                    nbr = (nx, ny)
                    if 0 <= nx < wrld.width() and 0 <= ny < wrld.height():
                        if not wrld.wall_at(nx, ny) and not wrld.bomb_at(nx, ny) and nbr not in visited:
                            visited.add(nbr)
                            queue.append(nbr)
        return len(visited)

    def _find_exit(self, wrld):
        if hasattr(wrld, 'exitcell') and wrld.exitcell is not None:
            return wrld.exitcell
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.exit_at(x, y):
                    return (x, y)
        return None

    @lru_cache(maxsize=1024)
    def _astar_dist(self, wrld, start, goal):
        if start == goal:
            return 0

        frontier = [(self._manhattan(start, goal), 0, start)]
        cost_so_far = {start: 0}

        while frontier:
            f, g, current = heapq.heappop(frontier)

            if current == goal:
                return g
            if g > cost_so_far[current]:
                continue

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = current[0] + dx, current[1] + dy
                    if 0 <= nx < wrld.width() and 0 <= ny < wrld.height() and not wrld.wall_at(nx, ny):
                        new_cost = g + 1
                        nbr = (nx, ny)
                        if nbr not in cost_so_far or new_cost < cost_so_far[nbr]:
                            cost_so_far[nbr] = new_cost
                            priority = new_cost + self._manhattan(nbr, goal)
                            heapq.heappush(frontier, (priority, new_cost, nbr))

        return 999

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

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])