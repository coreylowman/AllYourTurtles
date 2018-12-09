#!/usr/bin/env python3

# Import the Halite SDK, which will let you interact with the game.
import hlt
from hlt import constants, Position, Direction, entity

from datetime import datetime
import random
import logging
from collections import defaultdict
import math

width = 0
height = 0
cardinal_directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def get_positions(gmap):
    positions = []
    for y in range(gmap.height):
        for x in range(gmap.width):
            positions.append(Position(x, y))
    return positions


def normalize(x, y):
    return x % width, y % height


def cardinal_neighbors(p):
    return [normalize(p[0] + d[0], p[1] + d[1]) for d in cardinal_directions]


def direction_between(gmap, a, b):
    for dir in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]:
        if gmap.normalize(a.directional_offset(dir)) == gmap.normalize(b):
            return dir


def iterate_by_radius(x, y, max_radius=math.inf):
    explored = set()
    open = {normalize(x, y)}
    r = 0
    while len(open) > 0:
        if r > max_radius:
            return

        yield open

        next = set()
        for p in open:
            for n in cardinal_neighbors(p):
                if n not in explored:
                    next.add(n)
                    explored.add(n)
        open = next
        r += 1


def get_halite_by_position(gmap):
    halite_by_position = {}
    for position in get_positions(gmap):
        halite_by_position[(position.x, position.y)] = gmap[position].halite_amount
    return halite_by_position


def log(s):
    # logging.info('[{}] {}'.format(datetime.now(), s))
    pass


class IncomeEstimation:
    @staticmethod
    def hpt_of(me, gmap, turns_remaining, ship, closest_dropoff, destination, halite_weight=4, time_weight=1):
        turns_to_move = gmap.calculate_distance(ship.position, destination)
        closest_dropoff_distance = gmap.calculate_distance(ship.position, closest_dropoff)
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
        halite_remaining = sum(map(lambda p: gmap[p].halite_amount, get_positions(gmap)))
        my_ships = len(me.get_ships())
        total_ships = sum([len(player.get_ships()) for player in game.players.values()])
        other_ships = total_ships - my_ships
        expected_halite = my_ships * halite_remaining / total_ships if total_ships > 0 else 0
        expected_halite_1 = (my_ships + 1) * halite_remaining / (total_ships + 1)
        halite_gained = expected_halite_1 - expected_halite
        return halite_gained - constants.SHIP_COST


class ResourceAllocation:
    @staticmethod
    def goals_for_ships(me, gmap, ships, turns_remaining):
        available_positions = get_positions(gmap)
        scheduled_positions = set()
        n = len(ships)
        goals = [ships[i].position for i in range(n)]
        scheduled = [False for i in range(n)]
        dropoffs = [me.shipyard.position]
        dropoffs.extend([drp.position for drp in me.get_dropoffs()])

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
            dropoffs.append(ships[i].position)
            halite -= constants.DROPOFF_COST - gmap[ships[i].position].halite_amount - ships[i].halite_amount

        unscheduled = [i for i in range(n) if not scheduled[i]]

        log('allocating stills')
        for i in unscheduled:
            if gmap[ships[i].position].halite_amount / constants.MOVE_COST_RATIO > ships[i].halite_amount:
                scheduled[i] = True
                p = gmap.normalize(ships[i].position)
                available_positions.remove(p)
                scheduled_positions.add((p.x, p.y))

        unscheduled = [i for i in range(n) if not scheduled[i]]

        log('building assignments')

        halite_by_pos = get_halite_by_position(gmap)

        hpt_by_assignment = {}
        for i in unscheduled:
            closest_dropoff = min(dropoffs, key=lambda drp: gmap.calculate_distance(ships[i].position, drp))

            for p in dropoffs:
                hpt = IncomeEstimation.hpt_of(me, gmap, turns_remaining, ships[i], closest_dropoff, p)
                hpt_by_assignment[(p.x, p.y, i)] = hpt

            # TODO don't assign to a position nearby with an enemy ship on it
            for ps in iterate_by_radius(ships[i].position.x, ships[i].position.y):
                ps -= scheduled_positions
                if len(ps) == 0:
                    continue

                best_raw = max(ps, key=halite_by_pos.get)
                best = Position(*best_raw)
                hpt = IncomeEstimation.hpt_of(me, gmap, turns_remaining, ships[i], closest_dropoff, best)
                hpt_by_assignment[(best_raw[0], best_raw[1], i)] = hpt

        log('sorting assignments')

        assignments = sorted(hpt_by_assignment, key=hpt_by_assignment.get, reverse=True)

        log('gathering assignments')

        for x, y, i in assignments:
            if scheduled[i] or (x, y) in scheduled_positions:
                continue

            goals[i] = Position(x, y)
            scheduled[i] = True
            unscheduled.remove(i)
            if goals[i] not in dropoffs:
                scheduled_positions.add((x, y))

            if len(unscheduled) == 0:
                break

        return goals

    @staticmethod
    def can_convert_to_dropoff(me, gmap, ship, dropoffs, dropoff_radius=8):
        position = ship.position
        for drp in dropoffs:
            if gmap.calculate_distance(position, drp) <= 2 * dropoff_radius:
                return False, 0

        position = gmap.normalize(position)

        # TODO check goals of ships, not current positions
        halite_around = ship.halite_amount + gmap[position].halite_amount
        turtles_around = 0
        for ps in iterate_by_radius(position.x, position.y, dropoff_radius):
            for p in ps:
                p = Position(*p)
                halite_around += gmap[p].halite_amount
                if gmap[p].is_occupied and gmap[p].ship.owner == me.id:
                    turtles_around += 1
                    halite_around += gmap[p].ship.halite_amount

        return halite_around > constants.DROPOFF_COST and turtles_around > 2, halite_around


class PathPlanning:
    @staticmethod
    def next_positions_for(me, gmap, ships, other_ships, turns_remaining, goals):
        n = len(ships)
        current = [ships[i].position for i in range(n)]
        next_positions = [current[i] for i in range(n)]
        reservation_table = defaultdict(set)
        scheduled = [False for i in range(n)]
        dropoffs = {(me.shipyard.position.x, me.shipyard.position.y)}
        dropoffs.update({(drp.position.x, drp.position.y) for drp in me.get_dropoffs()})

        log('reserving other ship positions')

        # TODO if we outnumber the other ship, dont reserve its location
        for ship in other_ships:
            curr = normalize(ship.position.x, ship.position.y)
            reservation_table[0].add(curr)
            reservation_table[1].add(curr)
            for next in cardinal_neighbors(curr):
                reservation_table[1].add(next)

        log('converting dropoffs')
        for i in range(n):
            if goals[i] is None:
                scheduled[i] = True
                next_positions[i] = None

        unscheduled = [i for i in range(n) if not scheduled[i]]

        log('locking stills')
        for i in unscheduled:
            if current[i] == goals[i]:
                log(ships[i])
                raw_pos, t = (current[i].x, current[i].y), 1
                if raw_pos not in dropoffs or turns_remaining - t > width:
                    reservation_table[t].add(raw_pos)
                scheduled[i] = True

        unscheduled = [i for i in range(n) if not scheduled[i]]
        total = len(unscheduled)

        log('planning paths')

        for q, i in enumerate(unscheduled):
            log('ship {} ({}/{})...'.format(ships[i].id, q, total))
            path = PathPlanning.a_star(gmap, current[i], goals[i], reservation_table)
            log(path)
            for raw_pos, t in path:
                if raw_pos not in dropoffs or turns_remaining - t > width:
                    reservation_table[t].add(raw_pos)
            next_positions[i] = Position(*path[1][0])

        log('paths planned')

        return next_positions

    @staticmethod
    def path_stats(gmap, start, goal):
        delta = 0
        path = PathPlanning.a_star(gmap, start, goal, defaultdict(set))
        for i in range(1, len(path)):
            prev, cur = path[i - 1], path[i]
            if prev == cur:
                delta += gmap[Position(*prev)].halite_amount / constants.EXTRACT_RATIO
            else:
                delta -= gmap[Position(*prev)].halite_amount / constants.MOVE_COST_RATIO
        return path, delta

    @staticmethod
    def a_star(gmap, start, goal, reservation_table, WINDOW=8):
        """windowed hierarchical cooperative a*"""

        def heuristic(p):
            # distance is time + cost, so heuristic is time + distance, but time is just 1 for every square, so
            # we can just double
            return 2 * gmap.calculate_distance(p, goal)

        start_raw = normalize(start.x, start.y)
        goal_raw = normalize(goal.x, goal.y)

        # log('{} -> {}'.format(start_raw, goal_raw))

        if start_raw == goal_raw:
            return [(start_raw, 0), (goal_raw, 1)]

        max_halite = max(map(lambda p: gmap[p].halite_amount, get_positions(gmap)))

        closed_set = set()
        open_set = set()
        g_score = defaultdict(lambda: math.inf)
        h_score = defaultdict(lambda: math.inf)
        f_score = defaultdict(lambda: math.inf)
        came_from = {}

        open_set.add((start_raw, 0))
        g_score[start_raw] = 0
        h_score[start_raw] = heuristic(start)
        f_score[start_raw] = g_score[start_raw] + h_score[start_raw]

        while len(open_set) > 0:
            cpt = min(open_set, key=lambda pt: f_score[pt[0]])

            current_raw, t = normalize(*cpt[0]), cpt[1]
            current = Position(*current_raw)
            if current == goal:
                return PathPlanning._reconstruct_path(came_from, (current_raw, t))

            # log('- Expanding {} at {}. f={}'.format(current_raw, t, f_score[current_raw]))

            open_set.remove(cpt)
            closed_set.add(cpt)

            move_cost = gmap[current].halite_amount / max_halite
            nt = t + 1

            for neighbor in current.get_surrounding_cardinals() + [current]:
                neighbor_raw = normalize(neighbor.x, neighbor.y)
                npt = (neighbor_raw, nt)

                if npt in closed_set or (nt < WINDOW and neighbor_raw in reservation_table[nt]):
                    continue

                cost = 0 if current == neighbor else move_cost
                dist = cost + 1
                g = g_score[current_raw] + dist

                if npt not in open_set:
                    open_set.add(npt)
                elif g >= g_score[neighbor_raw]:
                    continue

                came_from[npt] = (current_raw, t)
                g_score[neighbor_raw] = g
                h_score[neighbor_raw] = heuristic(neighbor)
                f_score[neighbor_raw] = g_score[neighbor_raw] + h_score[neighbor_raw]

                # log('-- Adding {} at {}. h={} g={}'.format(neighbor_raw, nt, h_score[neighbor_raw], g_score[neighbor_raw]))

        if start_raw in reservation_table[1]:
            for neighbor_raw in cardinal_neighbors(start_raw):
                if neighbor_raw not in reservation_table[1]:
                    return [(start_raw, 0), (neighbor_raw, 1)]

        return [(start_raw, 0), (start_raw, 1)]

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
        not_occupied_next_turn = me.shipyard.position not in filter(None, next_positions)
        return have_enough_halite and not_occupied and not_occupied_next_turn

    def should_make_ship(self, me, gmap):
        roi = IncomeEstimation.roi(self.game, me, gmap)
        return roi > 0 and len(me.get_ships()) < 50 and self.turns_remaining > 50

    def produce_commands(self, me, gmap):
        ships = list(me.get_ships())
        ships = sorted(ships, key=lambda s: gmap.calculate_distance(s.position, me.shipyard.position), reverse=True)
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
                commands.append(ships[i].move(direction_between(gmap, ships[i].position, next_positions[i])))
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
    global width, height
    commander = Commander()
    width = commander.game.game_map.width
    height = commander.game.game_map.height
    while True:
        commander.run_once()


main()
