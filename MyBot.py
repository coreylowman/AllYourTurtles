#!/usr/bin/env python3

# Import the Halite SDK, which will let you interact with the game.
import hlt
from hlt import constants

from copy import deepcopy
from datetime import datetime
import logging
from collections import defaultdict
import math


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


def iterate_by_radius(p, max_radius=math.inf):
    p = normalize(p)
    explored = set()
    current = {p}
    r = 0
    while len(current) > 0:
        if r > max_radius:
            return

        yield current

        nexts = set()
        for p in current:
            for n in cardinal_neighbors(p):
                if n not in explored:
                    nexts.add(n)
                    explored.add(n)
        current = nexts
        r += 1


def get_halite_by_position(gmap):
    return {p: gmap[p].halite_amount for p in gmap.positions}


def log(s):
    # logging.info('[{}] {}'.format(datetime.now(), s))
    pass


class IncomeEstimation:
    @staticmethod
    def hpt_of(me, gmap, turns_remaining, ship, closest_dropoff, destination, halite_weight=4, time_weight=1):
        turns_to_move = gmap.dist(ship.pos, destination)
        closest_dropoff_distance = gmap.dist(ship.pos, closest_dropoff)
        if gmap[destination].has_structure and gmap[destination].structure.owner == me.id:
            # TODO also add in value indicating hpt of creating a new ship
            amount_gained = ship.halite_amount
            turns_to_collect = 0
        elif turns_remaining < closest_dropoff_distance * 2:
            amount_gained = 0
            turns_to_collect = 1
        else:
            # TODO what about a large amount of halite?
            # TODO take into account movement cost?
            # TODO consider the HPT of attacking an enemy ship
            amount_can_gain = constants.MAX_HALITE - ship.halite_amount
            raw_amount_extracted = gmap[destination].halite_amount
            amount_gained = min(amount_can_gain, raw_amount_extracted)
            turns_to_collect = 1

        total_turns = turns_to_move + turns_to_collect
        if total_turns == 0:
            total_turns = 1

        # TODO this multiplier makes halite have greater weight than time, maybe experiment with different kinds?
        return (halite_weight * amount_gained) / (total_turns * time_weight)

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
    def goals_for_ships(me, gmap, ships, turns_remaining):
        scheduled_positions = set()
        n = len(ships)
        goals = [ships[i].pos for i in range(n)]
        scheduled = [False for i in range(n)]
        dropoffs = [me.shipyard.pos]
        dropoffs.extend([drp.pos for drp in me.get_dropoffs()])

        log('allocating dropoffs')
        halite = me.halite_amount
        while halite > constants.DROPOFF_COST:
            # TODO may have less than 4k halite, but still be able to create dropoff due to bonus
            log('gathering potential dropoffs')
            score_by_dropoff = {}
            for i in range(n):
                can, score = ResourceAllocation.can_convert_to_dropoff(me, gmap, ships[i], dropoffs)
                if can:
                    score_by_dropoff[i] = score

            if len(score_by_dropoff) == 0:
                break

            log('choosing best dropoff')
            i = max(score_by_dropoff, key=score_by_dropoff.get)
            scheduled[i] = True
            goals[i] = None
            dropoffs.append(ships[i].pos)
            halite -= constants.DROPOFF_COST - gmap[ships[i].pos].halite_amount - ships[i].halite_amount

        unscheduled = [i for i in range(n) if not scheduled[i]]

        log('building assignments')

        hpt_by_assignment = {}
        for i in unscheduled:
            closest_dropoff = min(dropoffs, key=lambda drp: gmap.dist(ships[i].pos, drp))

            for p in dropoffs:
                hpt = IncomeEstimation.hpt_of(me, gmap, turns_remaining, ships[i], closest_dropoff, p)
                hpt_by_assignment[(p, i)] = hpt

            # TODO don't assign to a position nearby with an enemy ship on it
            for ps in iterate_by_radius(ships[i].pos):
                ps -= scheduled_positions
                if len(ps) == 0:
                    continue

                best = max(ps, key=gmap.halite_at)
                hpt = IncomeEstimation.hpt_of(me, gmap, turns_remaining, ships[i], closest_dropoff, best)
                hpt_by_assignment[(best, i)] = hpt

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

        return goals

    @staticmethod
    def can_convert_to_dropoff(me, gmap, ship, dropoffs, dropoff_radius=16):
        if gmap[ship.pos].has_structure:
            return False, 0

        for drp in dropoffs:
            if gmap.dist(ship.pos, drp) <= 2 * dropoff_radius:
                return False, 0

        # TODO check goals of ships, not current positions
        halite_around = gmap[ship.pos].halite_amount
        turtles_around = 0
        for ps in iterate_by_radius(ship.pos, max_radius=dropoff_radius):
            for p in ps:
                halite_around += gmap[p].halite_amount
                if gmap[p].is_occupied and gmap[p].ship.owner == me.id:
                    turtles_around += 1

        return halite_around > 1.5 * constants.DROPOFF_COST and turtles_around > 2, halite_around


class PathPlanning:
    @staticmethod
    def next_positions_for(me, gmap, ships, other_ships, turns_remaining, goals):
        n = len(ships)
        current = [ships[i].pos for i in range(n)]
        next_positions = [current[i] for i in range(n)]
        reservation_table = defaultdict(set)
        scheduled = [False] * n
        dropoffs = {me.shipyard.pos}
        dropoffs.update({drp.pos for drp in me.get_dropoffs()})

        log('reserving other ship positions')

        # TODO if we outnumber the other ship, dont reserve its location
        for ship in other_ships:
            reservation_table[0].add(ship.pos)
            reservation_table[1].add(ship.pos)
            for neighbor in cardinal_neighbors(ship.pos):
                reservation_table[1].add(neighbor)

        log('converting dropoffs')
        for i in range(n):
            if goals[i] is None:
                scheduled[i] = True
                next_positions[i] = None

        unscheduled = [i for i in range(n) if not scheduled[i]]

        log('locking stills')
        for i in unscheduled:
            if current[i] == goals[i] or gmap[current[i]].halite_amount / constants.MOVE_COST_RATIO > ships[
                i].halite_amount:
                log(ships[i])
                if current[i] not in dropoffs or turns_remaining - 1 > constants.WIDTH:
                    reservation_table[1].add(current[i])
                scheduled[i] = True

        unscheduled = [i for i in range(n) if not scheduled[i]]
        total = len(unscheduled)

        log('planning paths')

        for q, i in enumerate(unscheduled):
            log('ship {} ({}/{})...'.format(ships[i].id, q, total))
            path = PathPlanning.a_star(gmap, current[i], goals[i], ships[i].halite_amount, reservation_table)
            log(path)
            for raw_pos, t in path:
                if raw_pos not in dropoffs or turns_remaining - t > constants.HEIGHT:
                    reservation_table[t].add(raw_pos)
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
        g_score[start] = 0
        h_score[start] = heuristic(start)
        f_score[start] = g_score[start] + h_score[start]
        halite_at[(start, 0)] = starting_halite
        extractions_at[(start, 0)] = []

        while len(open_set) > 0:
            cpt = min(open_set, key=lambda pt: f_score[pt[0]])
            current, t = cpt

            halite_left = halite_at[cpt]

            halite_on_ground = gmap[current].halite_amount
            for pos, _, amt in extractions_at[cpt]:
                if pos == current:
                    halite_on_ground -= amt

            if current == goal:
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
                g = g_score[current] + dist

                if npt not in open_set:
                    open_set.add(npt)
                elif g >= g_score[neighbor]:
                    continue

                came_from[npt] = cpt
                g_score[neighbor] = g
                h_score[neighbor] = heuristic(neighbor)
                f_score[neighbor] = g_score[neighbor] + h_score[neighbor]

                if current == neighbor:
                    extracted = raw_extracted
                    halite_at[npt] = halite_left + extracted
                    extractions_at[npt] = extractions_at[cpt] + [(neighbor, nt, extracted)]
                else:
                    halite_at[npt] = halite_left - raw_move_cost
                    extractions_at[npt] = deepcopy(extractions_at[cpt])
                # log('-- Adding {} at {}. h={} g={}'.format(neighbor, nt, h_score[neighbor], g_score[neighbor]))

        if start in reservation_table[1]:
            for neighbor in cardinal_neighbors(start):
                if neighbor not in reservation_table[1]:
                    return [(start, 0), (neighbor, 1)]

        return [(start, 0), (start, 1)]

    @staticmethod
    def _reconstruct_path(prev_by_node, current):
        total_path = [current]
        while current in prev_by_node:
            current = prev_by_node[current]
            total_path.append(current)
        return list(reversed(total_path))


class Commander:
    def __init__(self):
        self.game = hlt.Game()
        self.game.ready("AllYourTurtles")
        self.plan_by_ship = {}

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

    def can_make_ship(self, me, gmap, next_positions, halite_left):
        have_enough_halite = halite_left >= constants.SHIP_COST
        not_occupied = not gmap[me.shipyard].is_occupied
        not_occupied_next_turn = me.shipyard.pos not in filter(None, next_positions)
        return have_enough_halite and not_occupied and not_occupied_next_turn

    def should_make_ship(self, me, gmap):
        roi = IncomeEstimation.roi(self.game, me, gmap)
        return roi > 0 and self.turns_remaining > 50

    def produce_commands(self, me, gmap):
        ships = list(me.get_ships())
        ships = sorted(ships, key=lambda s: gmap.dist(s.pos, me.shipyard.pos), reverse=True)
        ships = sorted(ships, key=lambda s: s.halite_amount, reverse=True)

        other_ships = []
        for oid in self.game.others:
            other_ships.extend(self.game.players[oid].get_ships())

        log('sorted ships: {}'.format(ships))

        goals = ResourceAllocation.goals_for_ships(me, gmap, ships, self.turns_remaining)
        log('allocated goals: {}'.format(goals))

        next_positions = PathPlanning.next_positions_for(me, gmap, ships, other_ships, self.turns_remaining, goals)
        log('planned paths: {}'.format(next_positions))

        halite_available = me.halite_amount
        commands = []
        for i in range(len(ships)):
            if next_positions[i] is not None:
                commands.append(ships[i].move(direction_between(ships[i].pos, next_positions[i])))
            else:
                commands.append(ships[i].make_dropoff())
                halite_available -= constants.DROPOFF_COST - ships[i].halite_amount - gmap[
                    ships[i].position].halite_amount
                log('Making dropoff with {}'.format(ships[i]))

        if self.can_make_ship(me, gmap, next_positions, halite_available) and self.should_make_ship(me, gmap):
            commands.append(me.shipyard.spawn())
            log('spawning')

        return commands


def main():
    commander = Commander()
    while True:
        commander.run_once()


main()
