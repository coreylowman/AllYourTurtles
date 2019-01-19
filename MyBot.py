#!/usr/bin/env python3

# Import the Halite SDK, which will let you interact with the game.
import hlt
from hlt import constants

from copy import deepcopy
from datetime import datetime
import logging
from collections import defaultdict
import math
from math import ceil, floor
from statistics import mean
from heapq import nlargest


def log(s):
    # logging.info('[{}] {}'.format(datetime.now(), s))
    pass


def normalize(p):
    x, y = p
    return x % constants.WIDTH, y % constants.HEIGHT


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def cardinal_neighbors(p):
    return [normalize(add(p, d)) for d in constants.CARDINAL_DIRECTIONS]


def all_neighbors(p):
    return set(normalize(add(p, d)) for d in constants.ALL_DIRECTIONS)


def direction_between(a, b):
    if normalize(a) == normalize(b):
        return 0, 0

    for d in constants.CARDINAL_DIRECTIONS:
        if normalize(add(a, d)) == normalize(b):
            return d


def pos_around(p, radius):
    positions = set()
    for y in range(radius + 1):
        for x in range(radius + 1 - y):
            positions.add(normalize((p[0] + x, p[1] + y)))
            positions.add(normalize((p[0] - x, p[1] + y)))
            positions.add(normalize((p[0] - x, p[1] - y)))
            positions.add(normalize((p[0] + x, p[1] - y)))
    return positions


def ships_around(p, owner, max_radius):
    ships, other_ships = 0, 0
    for p in pos_around(p, max_radius):
        if MAP[p].is_occupied:
            if MAP[p].ship.owner == owner:
                ships += 1
            else:
                other_ships += 1
    return ships, other_ships


def opponent_halite_next_to(p):
    halite = 0
    for p in pos_around(p, 1):
        if MAP[p].ship is not None and MAP[p].ship.owner != ME.id:
            halite += MAP[p].ship.halite_amount
    return halite


def centroid(positions):
    total = [0, 0]
    for p in positions:
        np = normalize(p)
        total[0] += np[0]
        total[1] += np[1]
    total[0] /= len(positions)
    total[1] /= len(positions)
    return round(total[0]), round(total[1])


def get_halite_by_position():
    return {p: MAP[p].halite_amount for p in MAP.positions}


class IncomeEstimation:
    @staticmethod
    def hpt_of(turns_remaining, turns_to_move, turns_to_dropoff, halite_on_board, space_left, halite_on_ground,
               inspiration_bonus):
        # TODO consider attacking opponent
        # TODO discount on number of enemy forces in area vs mine
        # TODO consider blocking opponent from dropoff
        if turns_to_dropoff > turns_remaining:
            return 0, 0, 1

        if turns_to_dropoff == 0:
            # TODO also add in value indicating hpt of creating a new ship
            # TODO discount if blocked?
            amount_gained = halite_on_board
            inspiration_gained = 0
        else:
            # TODO take into account movement cost?
            # TODO consider the HPT of attacking an enemy ship
            amount_gained = halite_on_ground
            if amount_gained > space_left:
                amount_gained = space_left
            space_left -= amount_gained
            inspiration_gained = inspiration_bonus
            if inspiration_gained > space_left:
                inspiration_gained = space_left

        collect_hpt = amount_gained / (turns_to_move + 1)
        inspiration_hpt = inspiration_gained / (turns_to_move + 1)
        # TODO dropoff bonus scale with amoutn gained
        dropoff_bonus = 1 / (turns_to_dropoff + 1)

        return collect_hpt + inspiration_hpt + dropoff_bonus, amount_gained + inspiration_gained, turns_to_move + 1

    @staticmethod
    def time_spent_mining(turns_to_dropoff, space_left, halite_on_ground, runner_up_assignment, extract_multiplier,
                          bonus_multiplier):
        t = 0
        while space_left > 0 and halite_on_ground > 0:
            halite = constants.MAX_HALITE - space_left
            hpt, _, _ = IncomeEstimation.hpt_of(TURNS_REMAINING - t, 0, turns_to_dropoff, halite,
                                                space_left, halite_on_ground, halite_on_ground * bonus_multiplier)
            if hpt < runner_up_assignment[0] or hpt < halite / (turns_to_dropoff + 1):
                return t, halite_on_ground

            extracted = min(ceil(halite_on_ground * extract_multiplier), space_left)
            halite_on_ground -= extracted

            extracted *= 1 + bonus_multiplier
            space_left = max(space_left - extracted, 0)
            t += 1

        return t, halite_on_ground

    @staticmethod
    def roi():
        # TODO take into account growth curve?
        # TODO take into account efficiency?
        # TODO take into account turns remaining?
        # TODO take into account number of other players? not working well in 4 player mode
        expected_halite = N * HALITE_REMAINING / TOTAL_N if TOTAL_N > 0 else 0
        expected_halite_1 = (N + 1) * HALITE_REMAINING / (TOTAL_N + 1)
        halite_gained = expected_halite_1 - expected_halite
        return halite_gained - constants.SHIP_COST


class ResourceAllocation:
    @staticmethod
    def goals_for_ships(opponent_next_positions):
        # TODO if we have way more ships than opponent ATTACK
        goals = [DROPOFF_BY_POS[SHIPS[i].pos] for i in range(N)]
        mining_times = [0 for i in range(N)]
        scheduled = [False] * N
        halite = ME.halite_amount

        if ENDGAME:
            return goals, mining_times, [], []

        unscheduled = list(range(N))

        log('building assignments')
        assignments = ResourceAllocation.assignments(unscheduled)
        max_non_dropoff_hpt_for_ship = [0] * N
        ships = halite // constants.SHIP_COST
        for j, (hpt, i, pos, gained, time) in enumerate(assignments):
            if pos in DROPOFFS:
                if ROI > 0 and ships < (halite + gained) // constants.SHIP_COST:
                    assignments[j] = (hpt + ROI / time, i, pos, gained, time)
            elif hpt > max_non_dropoff_hpt_for_ship[i]:
                max_non_dropoff_hpt_for_ship[i] = hpt

        log('sorting assignments')
        assignments.sort(
            key=lambda a: (a[0] - max_non_dropoff_hpt_for_ship[a[1]] if a[2] in DROPOFFS else a[0], a[1], a[2]),
            reverse=True)

        log('gathering assignments')
        reservations_by_pos = defaultdict(int)
        halite_by_pos = {}
        while len(assignments) > 0:
            hpt, i, pos, gained, time = assignments[0]
            goals[i] = pos
            scheduled[i] = True
            unscheduled.remove(i)
            i_assignments = [a for a in assignments if a[1] == i]
            assignments = [a for a in assignments if a[1] != i]
            mining_times[i], halite_on_ground = IncomeEstimation.time_spent_mining(
                DROPOFF_DIST_BY_POS[pos], SHIPS[i].space_left, halite_by_pos.get(pos, MAP[pos].halite_amount),
                i_assignments[1], EXTRACT_MULTIPLIER_BY_POS[pos], BONUS_MULTIPLIER_BY_POS[pos])
            if goals[i] not in DROPOFFS and pos in opponent_next_positions and MAP.dist(SHIPS[i].pos, pos) <= 1:
                reservations_by_pos[pos] += 0
                halite_by_pos[pos] = halite_by_pos.get(pos, MAP[pos].halite_amount)
                halite_by_pos[pos] += SHIPS[i].halite_amount
                halite_by_pos[pos] += opponent_halite_next_to(pos)
                # halite_on_ground = halite_by_pos[pos]
            else:
                reservations_by_pos[pos] += mining_times[i] + 1
                halite_by_pos[pos] = halite_on_ground

            if pos in DROPOFFS:
                halite += gained

            inspiration_bonus = halite_on_ground * BONUS_MULTIPLIER_BY_POS[pos]
            ships = halite // constants.SHIP_COST
            for j, (old_hpt, a_i, a_pos, a_gained, a_time) in enumerate(assignments):
                if a_pos == pos:
                    new_hpt, gained, time = IncomeEstimation.hpt_of(
                        TURNS_REMAINING - reservations_by_pos[pos],
                        MAP.dist(SHIPS[a_i].pos, pos) + reservations_by_pos[pos] + DIFFICULTY[pos],
                        DROPOFF_DIST_BY_POS[pos], SHIPS[a_i].halite_amount, SHIPS[a_i].space_left, halite_on_ground,
                        inspiration_bonus)
                    roi_bonus = 0
                    if pos in DROPOFFS:
                        if ROI > 0 and ships < (halite + gained) // constants.SHIP_COST:
                            roi_bonus = ROI / a_time
                    elif new_hpt > max_non_dropoff_hpt_for_ship[a_i]:
                        max_non_dropoff_hpt_for_ship[a_i] = new_hpt
                    assignments[j] = (new_hpt + roi_bonus, a_i, a_pos, a_gained, a_time)

            assignments.sort(
                key=lambda a: (a[0] - max_non_dropoff_hpt_for_ship[a[1]] if a[2] in DROPOFFS else a[0], a[1], a[2]),
                reverse=True)

        log('gathering potential dropoffs')
        score_by_dropoff, goals_by_dropoff = ResourceAllocation.get_potential_dropoffs(goals)
        log(score_by_dropoff)
        log(goals_by_dropoff)

        planned_dropoffs = []
        scheduled_dropoffs = []
        costs = []
        ships_for_dropoffs = set(range(N))
        if N > 10:
            planned_dropoffs = [drp for drp in goals_by_dropoff if goals_by_dropoff[drp] > 1]
            planned_dropoffs = sorted(planned_dropoffs, key=score_by_dropoff.get)
            for new_dropoff in planned_dropoffs:
                log('dropoff position: {}'.format(new_dropoff))

                i = min(ships_for_dropoffs, key=lambda i: MAP.dist(SHIPS[i].pos, new_dropoff))
                costs.append(constants.DROPOFF_COST - SHIPS[i].halite_amount - MAP[new_dropoff].halite_amount)

                if ME.halite_amount >= costs[-1]:
                    log('chosen ship: {}'.format(SHIPS[i]))
                    goals[i] = None if SHIPS[i].pos == new_dropoff else new_dropoff
                    ships_for_dropoffs.remove(i)
                    scheduled_dropoffs.append(new_dropoff)

        for drp in scheduled_dropoffs:
            for i in ships_for_dropoffs:
                if goals[i] in DROPOFFS and MAP.dist(drp, SHIPS[i].pos) < DROPOFF_DIST_BY_POS[SHIPS[i].pos]:
                    goals[i] = drp

        return goals, mining_times, planned_dropoffs, costs

    @staticmethod
    def assignments(unscheduled):
        # TODO don't assign to a position nearby with an enemy ship on it
        assignments_for_ship = [[] for i in unscheduled]
        sxs = [SHIPS[i].pos[0] for i in unscheduled]
        sys = [SHIPS[i].pos[1] for i in unscheduled]
        halites = [SHIPS[i].halite_amount for i in unscheduled]
        spaces = [SHIPS[i].space_left for i in unscheduled]
        positions = MAP.positions
        if constants.NUM_PLAYERS == 4:
            positions = positions - {ship.pos for ship in OTHER_SHIPS}
            positions.update(DROPOFFS)
        for p in positions:
            x, y = p
            halite_on_ground = MAP[p].halite_amount
            inspiration_bonus = halite_on_ground * BONUS_MULTIPLIER_BY_POS[p]
            dropoff_dist = DROPOFF_DIST_BY_POS[p]
            difficulty = DIFFICULTY[p]
            for i in unscheduled:
                d = MAP.distance_table[sxs[i] - x] + MAP.distance_table[sys[i] - y] + difficulty
                hpt, gained, time = IncomeEstimation.hpt_of(TURNS_REMAINING, d, dropoff_dist, halites[i], spaces[i],
                                                            halite_on_ground, inspiration_bonus)
                assignments_for_ship[i].append((hpt, i, p, gained, time))

        log('getting n largest assignments')
        assignments = []
        for i in unscheduled:
            assignments_for_ship[i] = nlargest(N + 1, assignments_for_ship[i])
            assignments.extend(assignments_for_ship[i])
        return assignments

    @staticmethod
    def get_potential_dropoffs(goals):
        positions = set(nlargest(constants.WIDTH, MAP.positions, key=MAP.halite_at))

        if constants.NUM_PLAYERS == 4:
            for i in range(N):
                positions.update(all_neighbors(SHIPS[i].pos))
                positions.update(all_neighbors(goals[i]))

        # get biggest halite positions as dropoffs
        score_by_dropoff = {}
        goals_by_dropoff = {}
        for pos in positions:
            can, score, num_goals = ResourceAllocation.can_convert_to_dropoff(pos, goals)
            if can:
                score_by_dropoff[pos] = score
                goals_by_dropoff[pos] = num_goals

        # only take the biggest dropoff when there are multiple nearby
        winners = set()
        for drp in score_by_dropoff:
            conflicting_winners = {w for w in winners if MAP.dist(w, drp) < 2 * DROPOFF_RADIUS}
            if len(conflicting_winners) == 0:
                winners.add(drp)
            elif all([score_by_dropoff[drp] > score_by_dropoff[w] for w in conflicting_winners]):
                winners -= conflicting_winners
                winners.add(drp)

        # select winners
        score_by_dropoff = {drp: score_by_dropoff[drp] for drp in winners}
        goals_by_dropoff = {drp: goals_by_dropoff[drp] for drp in winners}

        return score_by_dropoff, goals_by_dropoff

    @staticmethod
    def can_convert_to_dropoff(pos, goals):
        if MAP[pos].has_structure:
            return False, 0, 0

        for drp in DROPOFFS:
            if MAP.dist(pos, drp) <= 2 * DROPOFF_RADIUS:
                return False, 0, 0

        # give bonus for the halite on the dropoff
        halite_around = MAP[pos].halite_amount
        goals_around = 0
        for p in pos_around(pos, DROPOFF_RADIUS):
            halite_around += MAP[p].halite_amount
            if MAP[p].is_occupied and MAP[p].ship.owner == ME.id:
                halite_around += MAP[p].ship.halite_amount
            if p in goals:
                goals_around += 1

        ally_dist = sum(1 / (MAP.dist(s.pos, pos) + 1) for s in SHIPS)
        opponent_dists = []
        for owner in GAME.others:
            ships = GAME.players[owner].get_ships()
            opponent_dists.append(sum(1 / (MAP.dist(s.pos, pos) + 1) for s in ships))

        worthwhile = halite_around > DROPOFF_COST_MULT * constants.DROPOFF_COST
        allies_closer = all(ally_dist > opponent_dist for opponent_dist in opponent_dists)
        return worthwhile and allies_closer, halite_around, goals_around


class PathPlanning:
    @staticmethod
    def next_positions_for(opponent_model, goals, mining_times, spawning):
        current = [SHIPS[i].pos for i in range(N)]
        next_positions = [current[i] for i in range(N)]
        reservations_all = defaultdict(set)
        reservations_outnumbered = defaultdict(set)
        reservations_self = defaultdict(set)
        scheduled = [False] * N
        conflicts = [0] * N

        log('reserving other ship positions')

        def add_reservation(pos, time, is_own, outnumbered=True):
            # if not a dropoff, just add
            # if is a dropoff, add if enemy is reserving or if not endgame
            if pos in DROPOFFS:
                if not ENDGAME and is_own:
                    reservations_all[time].add(pos)
                    reservations_self[time].add(pos)
                    if outnumbered:
                        reservations_outnumbered[time].add(pos)
            else:
                reservations_all[time].add(pos)
                if outnumbered:
                    reservations_outnumbered[time].add(pos)
                if is_own:
                    reservations_self[time].add(pos)

        def schedule(i, pos):
            if i is not None:
                next_positions[i] = pos
                scheduled[i] = True
            for j in range(N):
                if pos in all_neighbors(current[j]):
                    conflicts[j] += 1

        def plan_path(i):
            path = PathPlanning.a_star(current[i], goals[i], SHIPS[i].halite_amount, reservations_outnumbered)
            planned = True
            if path is None:
                path = PathPlanning.a_star(current[i], goals[i], SHIPS[i].halite_amount, reservations_self)
                if path is None:
                    path = PathPlanning.a_star(current[i], goals[i], SHIPS[i].halite_amount, reservations_self,
                                               window=2)
                    if path is None:
                        path = [(current[i], 0), (current[i], 1)]
                        planned = False
            for raw_pos, t in path:
                add_reservation(raw_pos, t, is_own=True)
            if planned and goals[i] not in DROPOFFS:
                move_time = len(path)
                for t in range(move_time, move_time + mining_times[i]):
                    add_reservation(goals[i], t, is_own=True)
            schedule(i, path[1][0])

        if spawning:
            add_reservation(ME.shipyard.pos, 1, is_own=True)
            schedule(None, ME.shipyard.pos)

        for opponent_ship in OTHER_SHIPS:
            add_reservation(opponent_ship.pos, 0, is_own=False)
            # TODO roi of losing ship?
            for next_pos in opponent_model.get_next_positions_for(opponent_ship):
                for t in range(1, 9):
                    add_reservation(next_pos, t, is_own=False,
                                    outnumbered=ALLIES_AROUND[next_pos] <= OPPONENTS_AROUND[next_pos])

        for drp in OPPONENT_DROPOFFS:
            for t in range(0, 9):
                add_reservation(drp, t, is_own=False)

        log('converting dropoffs')
        for i in range(N):
            if goals[i] is None:
                scheduled[i] = True
                next_positions[i] = None
                # add_reservation(current[i], 1, is_own=True)

        unscheduled = [i for i in range(N) if not scheduled[i]]

        log('locking stills')
        for i in unscheduled:
            cost = floor(MAP[current[i]].halite_amount / constants.MOVE_COST_RATIO)
            if cost > SHIPS[i].halite_amount:
                add_reservation(current[i], 1, is_own=True)
                schedule(i, current[i])

        unscheduled = [i for i in range(N) if not scheduled[i]]

        log('planning stills')
        for i in unscheduled:
            if current[i] == goals[i]:
                plan_path(i)

        log('planning paths')
        unscheduled = set(i for i in range(N) if not scheduled[i])
        while len(unscheduled) > 0:
            i = min(unscheduled, key=lambda i: (
                -(conflicts[i] >= 4), -int(goals[i] in DROPOFFS), DROPOFF_DIST_BY_POS[current[i]],
                -SHIPS[i].halite_amount, SHIPS[i].id))
            plan_path(i)
            unscheduled.remove(i)
        log('paths planned')

        return next_positions

    @staticmethod
    def a_star(start, goal, starting_halite, reservation_table, window=8):
        """windowed hierarchical cooperative a*"""

        start = normalize(start)
        goal = normalize(goal)

        heuristic_weight = 1 if goal in DROPOFFS else 2
        still_multiplier = 0 if goal in DROPOFFS else 1
        avoidance_weight = 1 + constants.NUM_OPPONENTS * starting_halite / constants.MAX_HALITE
        if constants.NUM_PLAYERS == 2:
            avoidance_weight = 0

        def heuristic(p):
            # distance is time + cost, so heuristic is time + distance, but time is just 1 for every square, so
            # we can just double
            return heuristic_weight * MAP.dist(p, goal)

        # log('{} -> {}'.format(start, goal))

        if start == goal and goal not in reservation_table[1]:
            return [(start, 0), (goal, 1)]

        closed_set = set()
        open_set = set()
        g_score = defaultdict(lambda: math.inf)
        h_score = defaultdict(lambda: math.inf)
        f_score = defaultdict(lambda: math.inf)
        came_from = {}
        halite_at = {}
        extractions_at = defaultdict(list)

        open_set.add((start, 0))
        g_score[(start, 0)] = 0
        h_score[(start, 0)] = heuristic(start)
        f_score[(start, 0)] = g_score[(start, 0)] + h_score[(start, 0)]
        halite_at[(start, 0)] = starting_halite
        extractions_at[(start, 0)] = []

        while len(open_set) > 0:
            cpt = min(open_set, key=lambda pt: (f_score[pt], h_score[pt]))
            current, t = cpt

            halite_left = halite_at[cpt]

            halite_on_ground = MAP[current].halite_amount
            for pos, _, amt in extractions_at[cpt]:
                if pos == current:
                    halite_on_ground -= amt

            if current == goal and not (t < window and current in reservation_table[t]) and t > 0:
                return PathPlanning._reconstruct_path(came_from, cpt)

            # log('\t\tExpanding {}. f={} g={} h={} halite={} ground={}'.format(cpt, f_score[cpt], g_score[cpt],
            #                                                                   h_score[cpt], halite_left,
            #                                                                   halite_on_ground))

            open_set.remove(cpt)
            closed_set.add(cpt)

            raw_move_cost = floor(halite_on_ground / constants.MOVE_COST_RATIO)
            raw_extracted = ceil(halite_on_ground / constants.EXTRACT_RATIO)
            move_cost = raw_move_cost / constants.MAX_HALITE
            nt = t + 1

            neighbors = [current]
            if raw_move_cost <= halite_left:
                neighbors.extend(cardinal_neighbors(current))

            for neighbor in neighbors:
                npt = (neighbor, nt)

                if npt in closed_set or (nt < window and neighbor in reservation_table[nt]):
                    continue

                # TODO make dist actual dist, add new score for cost, and use cost to break ties
                dist = 1 - still_multiplier * move_cost if current == neighbor else 1 + move_cost
                g = g_score[cpt] + dist + avoidance_weight * PROB_OCCUPIED[neighbor]

                if npt not in open_set:
                    open_set.add(npt)
                elif g >= g_score[npt]:
                    continue

                came_from[npt] = cpt
                g_score[npt] = g
                h_score[npt] = heuristic(neighbor)
                f_score[npt] = g_score[npt] + h_score[npt]

                if current == neighbor:
                    extracted = raw_extracted
                    halite_at[npt] = halite_left + extracted
                    extractions_at[npt] = extractions_at[cpt] + [(neighbor, nt, extracted)]
                else:
                    halite_at[npt] = halite_left - raw_move_cost
                    extractions_at[npt] = deepcopy(extractions_at[cpt])
                # log('-- Adding {} at {}. h={} g={}'.format(neighbor, nt, h_score[npt], g_score[npt]))

    @staticmethod
    def _reconstruct_path(prev_by_node, current):
        total_path = [current]
        while current in prev_by_node:
            current = prev_by_node[current]
            total_path.append(current)
        return list(reversed(total_path))


class OpponentModel:
    def __init__(self, n=10):
        self._n = n
        self._pos_by_ship = {}
        self._moves_by_ship = {}
        self._predicted_by_ship = {}
        self._potentials_by_ship = {}

        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def get_next_positions_for(self, ship):
        return self._predicted_by_ship[ship]

    def get_next_positions(self):
        positions = set()
        for ship in self._predicted_by_ship:
            positions.update(self._predicted_by_ship[ship])
        return positions

    def prob_occupied(self):
        prob_by_pos = defaultdict(float)
        for ship, positions in self._predicted_by_ship.items():
            for pos in positions:
                prob_by_pos[pos] += 1 / len(positions)

        # TODO do something else for frozen?
        for pos in prob_by_pos:
            if prob_by_pos[pos] > 1:
                prob_by_pos[pos] = 1

        return prob_by_pos

    def update_all(self):
        predicted = self.get_next_positions()
        actual = set(s.pos for s in OTHER_SHIPS)
        potentials = set()
        for ship, potentials in self._potentials_by_ship.items():
            potentials.update(potentials)

        for pos in potentials:
            was_predicted = pos in predicted
            was_taken = pos in actual
            if was_predicted and was_taken:
                self.tp += 1
            elif was_predicted and not was_taken:
                self.fp += 1
            elif not was_predicted and was_taken:
                self.fn += 1
            else:
                self.tn += 1

        total = self.tp + self.tn + self.fp + self.fn
        if total > 0:
            mcc = self.tp * self.tn - self.fp * self.fn
            denom = (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)
            mcc /= math.sqrt(1 if denom == 0.0 else denom)
            log('Opponent Model: tp={:.2f} tn={:.2f} fp={:.2f} fn={:.2f}'.format(
                100 * self.tp / total, 100 * self.tn / total, 100 * self.fp / total, 100 * self.fn / total))
            log('Opponent Model: mcc={}'.format(mcc))

        for opponent_ship in OTHER_SHIPS:
            self.update(opponent_ship)

        removed_ships = [ship for ship in self._predicted_by_ship if ship not in OTHER_SHIPS]
        for ship in removed_ships:
            del self._pos_by_ship[ship]
            del self._moves_by_ship[ship]
            del self._predicted_by_ship[ship]
            del self._potentials_by_ship[ship]

    def update(self, ship):
        if ship not in self._pos_by_ship:
            moves = [(0, 0)]
        else:
            moves = self._moves_by_ship[ship]
            moves.append(direction_between(ship.pos, self._pos_by_ship[ship]))
            moves = moves[-self._n:]
        self._moves_by_ship[ship] = moves

        self._pos_by_ship[ship] = tuple(ship.pos)
        neighbors = cardinal_neighbors(ship.pos)

        if ship.halite_amount < floor(MAP[ship.pos].halite_amount / constants.MOVE_COST_RATIO):
            predicted_moves = {(0, 0)}
        elif len(set(moves)) == 1 and moves[0] == (0, 0):
            predicted_moves = {(0, 0)}
        else:
            predicted_moves = list(constants.CARDINAL_DIRECTIONS) + [(0, 0)]

        self._predicted_by_ship[ship] = set(normalize(add(ship.pos, move)) for move in predicted_moves)
        self._potentials_by_ship[ship] = set(neighbors + [ship.pos])


class Commander:
    def __init__(self):
        GAME.ready("AllYourTurtles")
        self.opponent_model = OpponentModel()

    def run_once(self):
        GAME.update_frame()
        self.update_globals()
        start_time = datetime.now()
        log('Starting turn {}'.format(GAME.turn_number))
        queue = self.produce_commands()
        GAME.end_turn(queue)
        log('Turn took {}'.format((datetime.now() - start_time).total_seconds()))

    def update_globals(self):
        global GAME, MAP, ME, OTHER_PLAYERS, TURNS_REMAINING, ENDGAME, SHIPS, N, OTHER_SHIPS, OPPONENT_NS, TOTAL_N
        global DROPOFFS, OPPONENT_DROPOFFS, DROPOFF_BY_POS, DROPOFF_DIST_BY_POS
        global OPPONENTS_AROUND, ALLIES_AROUND, INSPIRED_BY_POS, EXTRACT_MULTIPLIER_BY_POS, BONUS_MULTIPLIER_BY_POS
        global HALITE_REMAINING, PCT_REMAINING, PCT_COLLECTED, DIFFICULTY, REMAINING_WEIGHT, COLLECTED_WEIGHT
        global PROB_OCCUPIED, ROI

        log('Updating data...')

        TURNS_REMAINING = constants.MAX_TURNS - GAME.turn_number
        SHIPS = ME.get_ships()
        N = len(SHIPS)
        OTHER_SHIPS = []
        OPPONENT_NS = []
        for other in OTHER_PLAYERS:
            OTHER_SHIPS.extend(other.get_ships())
            OPPONENT_NS.append(len(other.get_ships()))
        TOTAL_N = N + len(OTHER_SHIPS)

        self.opponent_model.update_all()
        prob_by_pos = self.opponent_model.prob_occupied()

        OPPONENTS_AROUND = defaultdict(int)
        ALLIES_AROUND = defaultdict(int)
        for ship in SHIPS:
            for p in pos_around(ship.pos, constants.INSPIRATION_RADIUS):
                ALLIES_AROUND[p] += 1
        for ship in OTHER_SHIPS:
            for p in pos_around(ship.pos, constants.INSPIRATION_RADIUS):
                OPPONENTS_AROUND[p] += 1

        DROPOFFS = set([ME.shipyard.pos] + [drp.pos for drp in ME.get_dropoffs()])

        OPPONENT_DROPOFFS = []
        for player in OTHER_PLAYERS:
            OPPONENT_DROPOFFS.append(player.shipyard.pos)
            for drp in player.get_dropoffs():
                OPPONENT_DROPOFFS.append(drp.pos)

        halite = 0
        for pos in MAP.positions:
            drp = min(DROPOFFS, key=lambda drp: MAP.dist(drp, pos))
            drp_dist = MAP.dist(pos, drp)
            inspired = OPPONENTS_AROUND[pos] >= constants.INSPIRATION_SHIP_COUNT
            extract = constants.INSPIRED_EXTRACT_MULTIPLIER if inspired else constants.EXTRACT_MULTIPLIER
            bonus = constants.INSPIRED_BONUS_MULTIPLIER if inspired else 0
            DROPOFF_BY_POS[pos] = drp
            DROPOFF_DIST_BY_POS[pos] = drp_dist
            INSPIRED_BY_POS[pos] = inspired
            EXTRACT_MULTIPLIER_BY_POS[pos] = extract
            BONUS_MULTIPLIER_BY_POS[pos] = bonus
            DIFFICULTY[pos] = 0
            halite += MAP[pos].halite_amount
            PROB_OCCUPIED[pos] = prob_by_pos[pos]
        HALITE_REMAINING = halite
        PCT_REMAINING = halite / TOTAL_HALITE
        PCT_COLLECTED = 1 - PCT_REMAINING
        REMAINING_WEIGHT = constants.NUM_OPPONENTS + PCT_REMAINING
        COLLECTED_WEIGHT = constants.NUM_OPPONENTS + PCT_COLLECTED

        for drp in DROPOFFS:
            DIFFICULTY[drp] = OPPONENTS_AROUND[drp]

        SHIPS = sorted(SHIPS,
                       key=lambda ship: (DROPOFF_DIST_BY_POS[ship.pos], -ship.halite_amount, ship.id))

        if not ENDGAME:
            ENDGAME = any(DROPOFF_DIST_BY_POS[ship.pos] >= TURNS_REMAINING for ship in SHIPS) or PCT_REMAINING == 0

        ROI = IncomeEstimation.roi()

        log('Updated data')

    def should_make_ship(self, goals):
        if ENDGAME:
            return False

        for i in range(N):
            if MAP.dist(SHIPS[i].pos, ME.shipyard.pos) == 1 and goals[i] == ME.shipyard.pos:
                return False

        my_produced = len(ME.ships_produced)
        opponent_produced = ceil(mean([len(other.ships_produced) for other in OTHER_PLAYERS]))
        return my_produced < opponent_produced or ROI > 0

    def produce_commands(self):
        goals, mining_times, planned_dropoffs, costs = ResourceAllocation.goals_for_ships(
            self.opponent_model.get_next_positions())
        log('allocated goals: {}'.format(goals))

        halite_available = ME.halite_amount
        spawning = False
        if halite_available >= constants.SHIP_COST and halite_available - sum(
                costs) >= constants.SHIP_COST and self.should_make_ship(goals):
            halite_available -= constants.SHIP_COST
            spawning = True
            log('spawning')

        next_positions = PathPlanning.next_positions_for(self.opponent_model, goals, mining_times, spawning)
        log('planned paths: {}'.format(next_positions))

        commands = []
        if spawning:
            commands.append(ME.shipyard.spawn())
        for i in range(N):
            if next_positions[i] is not None:
                commands.append(SHIPS[i].move(direction_between(SHIPS[i].pos, next_positions[i])))
            else:
                cost = constants.DROPOFF_COST - SHIPS[i].halite_amount - MAP[SHIPS[i].pos].halite_amount
                if halite_available >= cost:
                    commands.append(SHIPS[i].make_dropoff())
                    halite_available -= cost
                    log('Making dropoff with {}'.format(SHIPS[i]))
                    planned_dropoffs.remove(SHIPS[i].pos)
                else:
                    commands.append(SHIPS[i].stay_still())

        return commands


def main():
    commander = Commander()
    while True:
        commander.run_once()


GAME = hlt.Game()
MAP = GAME.game_map
ME = GAME.me
OTHER_PLAYERS = [GAME.players[oid] for oid in GAME.others]

TURNS_REMAINING = 0
ENDGAME = False

SHIPS = []
N = 0
OTHER_SHIPS = []
OPPONENT_NS = []
TOTAL_N = 0

DROPOFFS = set()
DROPOFF_RADIUS = 8 if constants.NUM_PLAYERS == 2 else 4
DROPOFF_COST_MULT = 5 if constants.NUM_PLAYERS == 2 else 3
OPPONENT_DROPOFFS = []
DROPOFF_BY_POS = {}
DROPOFF_DIST_BY_POS = {}

OPPONENTS_AROUND = {}
ALLIES_AROUND = {}
INSPIRED_BY_POS = {}
EXTRACT_MULTIPLIER_BY_POS = {}
BONUS_MULTIPLIER_BY_POS = {}
DIFFICULTY = {}

SIZE = constants.WIDTH * constants.HEIGHT
TOTAL_HALITE = sum(MAP[p].halite_amount for p in MAP.positions)
HALITE_REMAINING = TOTAL_HALITE
PCT_REMAINING = HALITE_REMAINING / TOTAL_HALITE
PCT_COLLECTED = 1 - PCT_REMAINING
REMAINING_WEIGHT = constants.NUM_OPPONENTS + PCT_REMAINING
COLLECTED_WEIGHT = constants.NUM_OPPONENTS + PCT_COLLECTED

ROI = 0

PROB_OCCUPIED = {}

main()
