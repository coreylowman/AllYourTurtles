#!/usr/bin/env python3

# Import the Halite SDK, which will let you interact with the game.
import hlt
from hlt import constants

from copy import deepcopy
from datetime import datetime
import logging
from collections import defaultdict
import math

DROPOFF_COST_MULTIPLIER = 0
VISION_BY_POS = {}


def set_constants(game):
    global DROPOFF_COST_MULTIPLIER, VISION_BY_POS

    DROPOFF_COST_MULTIPLIER = 5 if len(game.players) == 2 else 5

    vision = min(constants.WIDTH, 48)
    vision //= 2
    for pos in game.game_map.positions:
        VISION_BY_POS[pos] = pos_around(pos, vision)


def normalize(p):
    x, y = p
    return x % constants.WIDTH, y % constants.HEIGHT


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def cardinal_neighbors(p):
    return [normalize(add(p, d)) for d in constants.CARDINAL_DIRECTIONS]


def direction_between(a, b):
    if normalize(a) == normalize(b):
        return 0, 0

    for d in constants.CARDINAL_DIRECTIONS:
        if normalize(add(a, d)) == normalize(b):
            return d


def pos_around(p, radius):
    positions = set()
    for y in range(radius):
        for x in range(radius - y):
            positions.add(normalize((p[0] + x, p[1] + y)))
            positions.add(normalize((p[0] - x, p[1] + y)))
            positions.add(normalize((p[0] - x, p[1] - y)))
            positions.add(normalize((p[0] + x, p[1] - y)))
    return positions


def ships_around(gmap, p, owner, max_radius):
    ships, other_ships = 0, 0
    for p in pos_around(p, max_radius):
        if gmap[p].is_occupied:
            if gmap[p].ship.owner == owner:
                ships += 1
            else:
                other_ships += 1
    return ships, other_ships


def centroid(positions):
    total = [0, 0]
    for p in positions:
        np = normalize(p)
        total[0] += np[0]
        total[1] += np[1]
    total[0] /= len(positions)
    total[1] /= len(positions)
    return round(total[0]), round(total[1])


def get_halite_by_position(gmap):
    return {p: gmap[p].halite_amount for p in gmap.positions}


def log(s):
    # logging.info('[{}] {}'.format(datetime.now(), s))
    pass


class IncomeEstimation:
    @staticmethod
    def hpt_of(me, gmap, turns_remaining, ship, destination, closest_dropoff_then, inspired):
        # TODO consider attacking opponent
        # TODO discount on number of enemy forces in area vs mine
        # TODO consider blocking opponent from dropoff
        turns_to_move = gmap.dist(ship.pos, destination) + 1
        turns_to_dropoff = gmap.dist(destination, closest_dropoff_then) + 1

        if turns_to_dropoff > turns_remaining:
            return 0

        if gmap[destination].has_structure and gmap[destination].structure.owner == me.id:
            # TODO also add in value indicating hpt of creating a new ship
            # TODO discount if blocked?
            amount_gained = ship.halite_amount
        else:
            # TODO take into account movement cost?
            # TODO consider the HPT of attacking an enemy ship
            amount_can_gain = constants.MAX_HALITE - ship.halite_amount
            amount_extracted = gmap[destination].halite_amount

            if inspired:
                amount_extracted *= (1 + constants.INSPIRED_BONUS_MULTIPLIER)

            amount_gained = min(amount_can_gain, amount_extracted)

        collect_hpt = amount_gained / turns_to_move
        # TODO dropoff bonus scale with amoutn gained
        dropoff_bonus = 1 / turns_to_dropoff

        return collect_hpt + dropoff_bonus

    @staticmethod
    def roi(game, me, gmap):
        # TODO take into account growth curve?
        # TODO take into account efficiency?
        # TODO take into account turns remaining?
        # TODO take into account number of other players? not working well in 4 player mode
        halite_remaining = sum(map(gmap.halite_at, gmap.positions))
        my_ships = len(me.get_ships())
        total_ships = sum([len(player.get_ships()) for player in game.players.values()])
        # other_ships = total_ships - my_ships
        expected_halite = my_ships * halite_remaining / total_ships if total_ships > 0 else 0
        expected_halite_1 = (my_ships + 1) * halite_remaining / (total_ships + 1)
        halite_gained = expected_halite_1 - expected_halite
        return halite_gained - constants.SHIP_COST


class ResourceAllocation:
    @staticmethod
    def goals_for_ships(me, gmap, ships, other_ships, dropoffs, dropoff_by_pos, turns_remaining, endgame,
                        dropoff_radius=8):
        # TODO if we have way more ships than opponent ATTACK
        scheduled_positions = set()
        n = len(ships)
        goals = [ships[i].pos for i in range(n)]
        scheduled = [False] * n

        if endgame:
            return [dropoff_by_pos[ships[i].pos] for i in range(n)], [], []

        unscheduled = list(range(n))

        opponents_around = defaultdict(int)
        allies_around = defaultdict(int)
        for ship in ships:
            for p in pos_around(ship.pos, constants.INSPIRATION_RADIUS):
                allies_around[p] += 1
        for ship in other_ships:
            for p in pos_around(ship.pos, constants.INSPIRATION_RADIUS):
                opponents_around[p] += 1

        log('building assignments')
        hpt_by_assignment = {}
        for i in unscheduled:
            # TODO don't assign to a position nearby with an enemy ship on it
            for p in VISION_BY_POS[ships[i].pos]:
                hpt = IncomeEstimation.hpt_of(me, gmap, turns_remaining, ships[i], p, dropoff_by_pos[p],
                                              opponents_around[p] >= constants.INSPIRATION_SHIP_COUNT)
                hpt_by_assignment[(p, i)] = hpt

        log('sorting assignments')

        assignments = sorted(hpt_by_assignment, key=hpt_by_assignment.get, reverse=True)

        log('gathering assignments')

        for pos, i in assignments:
            if scheduled[i] or pos in scheduled_positions:
                continue

            goals[i] = pos
            scheduled[i] = True
            unscheduled.remove(i)
            if goals[i] not in dropoffs:
                scheduled_positions.add(pos)

            if len(unscheduled) == 0:
                break

        log('gathering potential dropoffs')
        score_by_dropoff, goals_by_dropoff = ResourceAllocation.get_potential_dropoffs(me, gmap, dropoffs, goals,
                                                                                       dropoff_radius)
        log(score_by_dropoff)
        log(goals_by_dropoff)

        planned_dropoffs = []
        costs = []
        ships_for_dropoffs = set(range(n))
        if n > 10:
            planned_dropoffs = [drp for drp in goals_by_dropoff if goals_by_dropoff[drp] > 1]
            planned_dropoffs = sorted(planned_dropoffs, key=score_by_dropoff.get)
            for new_dropoff in planned_dropoffs:
                log('dropoff position: {}'.format(new_dropoff))

                i = min(ships_for_dropoffs, key=lambda i: gmap.dist(ships[i].pos, new_dropoff))
                ships_for_dropoffs.remove(i)
                costs.append(constants.DROPOFF_COST - ships[i].halite_amount - gmap[new_dropoff].halite_amount)

                log('chosen ship: {}'.format(ships[i]))
                goals[i] = None if ships[i].pos == new_dropoff else new_dropoff

        return goals, planned_dropoffs, costs

    @staticmethod
    def get_potential_dropoffs(me, gmap, dropoffs, goals, dropoff_radius):
        halite_by_pos = get_halite_by_position(gmap)

        # get biggest halite positions as dropoffs
        score_by_dropoff = {}
        goals_by_dropoff = {}
        for pos in sorted(halite_by_pos, key=halite_by_pos.get, reverse=True)[:constants.WIDTH]:
            can, score, num_goals = ResourceAllocation.can_convert_to_dropoff(me, gmap, pos, dropoffs, goals,
                                                                              dropoff_radius)
            if can:
                score_by_dropoff[pos] = score
                goals_by_dropoff[pos] = num_goals

        # only take the biggest dropoff when there are multiple nearby
        winners = set()
        for drp in score_by_dropoff:
            conflicting_winners = {w for w in winners if gmap.dist(w, drp) < 2 * dropoff_radius}
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
    def can_convert_to_dropoff(me, gmap, pos, dropoffs, goals, dropoff_radius):
        if gmap[pos].has_structure:
            return False, 0, 0

        for drp in dropoffs:
            if gmap.dist(pos, drp) <= 2 * dropoff_radius:
                return False, 0, 0

        # give bonus for the halite on the dropoff
        halite_around = gmap[pos].halite_amount
        goals_around = 0
        for p in pos_around(pos, dropoff_radius):
            halite_around += gmap[p].halite_amount
            if gmap[p].is_occupied and gmap[p].ship.owner == me.id:
                halite_around += gmap[p].ship.halite_amount
            if p in goals:
                goals_around += 1

        return halite_around > DROPOFF_COST_MULTIPLIER * constants.DROPOFF_COST, halite_around, goals_around


class PathPlanning:
    @staticmethod
    def next_positions_for(me, gmap, ships, opponent_ships, opponent_model, turns_remaining, goals, spawning):
        n = len(ships)
        current = [ships[i].pos for i in range(n)]
        next_positions = [current[i] for i in range(n)]
        reservations_all = defaultdict(set)
        reservations_self = defaultdict(set)
        scheduled = [False] * n
        dropoffs = {me.shipyard.pos}
        dropoffs.update({drp.pos for drp in me.get_dropoffs()})

        log('reserving other ship positions')

        def add_reservation(pos, time, is_own):
            # if not a dropoff, just add
            # if is a dropoff, add if enemy is reserving or if not endgame
            if pos not in dropoffs or not is_own or turns_remaining - time > constants.WIDTH:
                reservations_all[time].add(pos)
                if is_own:
                    reservations_self[time].add(pos)

        if spawning:
            add_reservation(me.shipyard.pos, 1, is_own=True)

        for opponent_ship in opponent_ships:
            add_reservation(opponent_ship.pos, 0, is_own=False)
            num_my_ships, num_opponent_ships = ships_around(gmap, opponent_ship.pos, me.id, max_radius=8)
            if num_my_ships <= num_opponent_ships:
                for next_pos in opponent_model.get_next_positions_for(opponent_ship):
                    add_reservation(next_pos, 1, is_own=False)

        log('converting dropoffs')
        for i in range(n):
            if goals[i] is None:
                scheduled[i] = True
                next_positions[i] = None
                add_reservation(current[i], 1, is_own=True)

        unscheduled = [i for i in range(n) if not scheduled[i]]

        log('locking stills')
        for i in unscheduled:
            cost = gmap[current[i]].halite_amount / constants.MOVE_COST_RATIO
            if cost > ships[i].halite_amount:
                add_reservation(current[i], 1, is_own=True)
                scheduled[i] = True

        unscheduled = [i for i in range(n) if not scheduled[i]]

        log('planning paths')

        for q, i in enumerate(unscheduled):
            path = PathPlanning.a_star(gmap, current[i], goals[i], ships[i].halite_amount, reservations_all)
            if path is None:
                path = PathPlanning.a_star(gmap, current[i], goals[i], ships[i].halite_amount, reservations_self)
                if path is None:
                    path = [(current[i], 0), (current[i], 1)]

            for raw_pos, t in path:
                if raw_pos not in dropoffs or turns_remaining - t > constants.HEIGHT:
                    add_reservation(raw_pos, t, is_own=True)
            next_positions[i] = path[1][0]
        log('paths planned')

        return next_positions

    @staticmethod
    def a_star(gmap, start, goal, starting_halite, reservation_table, window=8):
        """windowed hierarchical cooperative a*"""

        start = normalize(start)
        goal = normalize(goal)

        def heuristic(p):
            # distance is time + cost, so heuristic is time + distance, but time is just 1 for every square, so
            # we can just double
            return 2 * gmap.dist(p, goal)

        # log('{} -> {}'.format(start_raw, goal_raw))

        if start == goal and goal not in reservation_table[1]:
            return [(start, 0), (goal, 1)]

        halite_by_pos = get_halite_by_position(gmap)
        max_halite = max(halite_by_pos.values())

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

            halite_on_ground = gmap[current].halite_amount
            for pos, _, amt in extractions_at[cpt]:
                if pos == current:
                    halite_on_ground -= amt

            if current == goal and current not in reservation_table[t] and t > 0:
                return PathPlanning._reconstruct_path(came_from, cpt)

            # log('- Expanding {} at {}. f={}'.format(current, t, f_score[current]))

            open_set.remove(cpt)
            closed_set.add(cpt)

            raw_move_cost = halite_on_ground / constants.MOVE_COST_RATIO
            raw_extracted = halite_on_ground / constants.EXTRACT_RATIO
            move_cost = halite_on_ground / max_halite
            nt = t + 1

            neighbors = [current]
            if halite_on_ground / constants.MOVE_COST_RATIO <= halite_left:
                neighbors.extend(cardinal_neighbors(current))

            for neighbor in neighbors:
                npt = (neighbor, nt)

                if npt in closed_set or (nt < window and neighbor in reservation_table[nt]):
                    continue

                cost = 0 if current == neighbor else move_cost
                dist = cost + 1
                g = g_score[cpt] + dist

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
                # log('-- Adding {} at {}. h={} g={}'.format(neighbor, nt, h_score[neighbor], g_score[neighbor]))

    @staticmethod
    def _reconstruct_path(prev_by_node, current):
        total_path = [current]
        while current in prev_by_node:
            current = prev_by_node[current]
            total_path.append(current)
        return list(reversed(total_path))


class OpponentModel:
    def __init__(self, n=5):
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

    def update_all(self, gmap, opponent_ships):
        predicted = self.get_next_positions()
        actual = set(s.pos for s in opponent_ships)
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

        for opponent_ship in opponent_ships:
            self.update(gmap, opponent_ship)

        removed_ships = [ship for ship in self._predicted_by_ship if ship not in opponent_ships]
        for ship in removed_ships:
            del self._pos_by_ship[ship]
            del self._moves_by_ship[ship]
            del self._predicted_by_ship[ship]
            del self._potentials_by_ship[ship]

    def update(self, gmap, ship):
        if ship not in self._pos_by_ship:
            moves = [(0, 0)]
        else:
            moves = self._moves_by_ship[ship]
            moves.append(direction_between(ship.pos, self._pos_by_ship[ship]))
            moves = moves[-self._n:]
        self._moves_by_ship[ship] = moves

        self._pos_by_ship[ship] = tuple(ship.pos)
        neighbors = cardinal_neighbors(ship.pos)

        if ship.halite_amount < gmap[ship.pos].halite_amount / constants.MOVE_COST_RATIO:
            predicted_moves = {(0, 0)}
        elif len(set(moves)) == 1 and moves[0] == (0, 0):
            predicted_moves = {(0, 0)}
        elif moves[-1] == (0, 0) and all(
                gmap[ship.pos].halite_amount > gmap[p].halite_amount for p in neighbors):
            predicted_moves = {(0, 0)}
        else:
            predicted_moves = set(moves)

        self._predicted_by_ship[ship] = set(normalize(add(ship.pos, move)) for move in predicted_moves)
        self._potentials_by_ship[ship] = set(neighbors + [ship.pos])


class Commander:
    def __init__(self):
        self.game = hlt.Game()

        set_constants(self.game)

        self.game.ready("AllYourTurtles")
        self.plan_by_ship = {}
        self.endgame = False

        self.opponent_model = OpponentModel()

    @property
    def turns_remaining(self):
        return constants.MAX_TURNS - self.game.turn_number

    def run_once(self):
        self.game.update_frame()
        start_time = datetime.now()
        log('Starting turn {}'.format(self.game.turn_number))
        queue = self.produce_commands(self.game.me, self.game.game_map)
        self.game.end_turn(queue)
        log('Turn took {}'.format(datetime.now() - start_time))

    def should_make_ship(self, me):
        my_ships = len(me.ships_produced)
        other_ships = [len(self.game.players[other].ships_produced) for other in self.game.others]
        other_avg = math.ceil(sum(other_ships) / len(other_ships))
        roi = IncomeEstimation.roi(self.game, me, self.game.game_map)
        return not self.endgame and (my_ships <= other_avg or roi > 0)

    def produce_commands(self, me, gmap):
        dropoffs = [me.shipyard.pos] + [drp.pos for drp in me.get_dropoffs()]
        dropoff_by_pos = {pos: min(dropoffs, key=lambda drp: gmap.dist(drp, pos)) for pos in gmap.positions}
        dropoff_by_ship = {ship: dropoff_by_pos[ship.pos] for ship in me.get_ships()}
        dropoff_dist_by_ship = {ship: gmap.dist(dropoff_by_ship[ship], ship.pos) for ship in me.get_ships()}
        ships = sorted(dropoff_dist_by_ship,
                       key=lambda ship: (dropoff_dist_by_ship[ship], -ship.halite_amount, ship.id))

        log('sorted ships: {}'.format(ships))

        if not self.endgame:
            turns_remaining = self.turns_remaining
            self.endgame = any(dropoff_dist_by_ship[ship] >= turns_remaining for ship in ships)

        other_ships = []
        for oid in self.game.others:
            other_ships.extend(self.game.players[oid].get_ships())

        self.opponent_model.update_all(gmap, other_ships)

        log('Updated opponent model')

        goals, planned_dropoffs, costs = ResourceAllocation.goals_for_ships(me, gmap, ships, other_ships, dropoffs,
                                                                            dropoff_by_pos, self.turns_remaining,
                                                                            self.endgame)
        log('allocated goals: {}'.format(goals))

        halite_available = me.halite_amount
        spawning = False
        if halite_available >= constants.SHIP_COST and halite_available - sum(
                costs) >= constants.SHIP_COST and self.should_make_ship(me):
            halite_available -= constants.SHIP_COST
            spawning = True
            log('spawning')

        next_positions = PathPlanning.next_positions_for(me, gmap, ships, other_ships, self.opponent_model,
                                                         self.turns_remaining, goals, spawning)
        log('planned paths: {}'.format(next_positions))

        commands = []
        if spawning:
            commands.append(me.shipyard.spawn())
        for i in range(len(ships)):
            if next_positions[i] is not None:
                commands.append(ships[i].move(direction_between(ships[i].pos, next_positions[i])))
            else:
                cost = constants.DROPOFF_COST - ships[i].halite_amount - gmap[ships[i].position].halite_amount
                if halite_available >= cost:
                    commands.append(ships[i].make_dropoff())
                    halite_available -= cost
                    log('Making dropoff with {}'.format(ships[i]))
                    planned_dropoffs.remove(ships[i].pos)
                else:
                    commands.append(ships[i].stay_still())

        return commands


def main():
    commander = Commander()
    while True:
        commander.run_once()


main()
