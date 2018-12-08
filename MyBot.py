#!/usr/bin/env python3

# Import the Halite SDK, which will let you interact with the game.
import hlt
from hlt import constants, Position, Direction, entity

from datetime import datetime
import random
import logging
from collections import defaultdict
from math import inf


"""
1. When to return? - when its more valuable to turn in halite than to keep collecting (take into account halite from new turtles created by turn in)
2. What path to take?
3. how to avoid self
4. how to avoid enemies
5. When to create ships? - when return of ships is higher than their cost
6. when to attack enemies?


# excavators
#

Path planning:
    - http://www.aaai.org/Papers/AIIDE/2005/AIIDE05-020.pdf
    - http://www.aaai.org/ocs/index.php/AAAI/AAAI10/paper/viewFile/1926/1950
*   - https://www.aaai.org/ocs/index.php/IJCAI/IJCAI11/paper/viewFile/3135/3585
    - https://arxiv.org/ftp/arxiv/papers/1401/1401.3905.pdf

    - https://www.aaai.org/ocs/index.php/IJCAI/IJCAI11/paper/viewFile/3132/3376

    - http://users.isr.ist.utl.pt/~ahaeusler/material/papers/Cooperative_Path_Planning_for_Multiple_UAVs_in_Dynamic_and_Uncertain_Environments.pdf
    - http://airspot.ru/book/file/1151/Brian_White_-_Cooperative_Path_Planning_of_Unmanned_Aerial_Vehicles_-_2011.pdf
    - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.4.8605&rep=rep1&type=pdf
"""

commander = None
width = None
height = None
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


def iterate_by_radius(gmap, x, y):
    explored = set()
    open = {normalize(x, y)}
    while len(open) > 0:
        yield open

        next = set()
        for p in open:
            for n in cardinal_neighbors(p):
                if n not in explored:
                    next.add(n)
                    explored.add(n)
        open = next


def get_halite_by_position(gmap):
    halite_by_position = {}
    for position in get_positions(gmap):
        halite_by_position[(position.x, position.y)] = gmap[position].halite_amount
    return halite_by_position


def clear_log():
    open('bot-{}.log'.format(commander.game.my_id), 'w').close()


def log(s):
    with open('bot-{}.log'.format(commander.game.my_id), 'a') as fp:
        fp.write('[{}] {}\n'.format(datetime.now(), s))


class IncomeEstimation:
    @staticmethod
    def hpt_of(me, gmap, turns_remaining, ship, closest_dropoff, destination):
        turns_to_move = gmap.calculate_distance(ship.position, destination)
        closest_dropoff_distance = gmap.calculate_distance(ship.position, closest_dropoff)
        if gmap[destination].has_structure and gmap[destination].structure.owner == me.id:
            # TODO also add in value indicating hpt of creating a new ship
            amount_gained = ship.halite_amount
            turns_to_collect = 0
        elif turns_remaining < closest_dropoff_distance * 1.5:
            amount_gained = 0
            turns_to_collect = 1
        else:
            # TODO what about a large amount of halite?
            # TODO take into account movement cost?
            # TODO consider the HPT of attacking an enemy ship
            amount_can_gain = constants.MAX_HALITE - ship.halite_amount
            raw_amount_extracted = constants.EXTRACT_RATIO * gmap[destination].halite_amount
            amount_gained = min(amount_can_gain, raw_amount_extracted)
            turns_to_collect = 1

        total_turns = turns_to_move + turns_to_collect
        if total_turns == 0:
            total_turns = 1

        return amount_gained / total_turns

    @staticmethod
    def roi(game, me, gmap, ships):
        # TODO take into account growth curve?
        # TODO take into account efficiency?
        # TODO take into account turns remaining?
        # TODO take into account number of other players? not working well in 4 player mode
        halite_remaining = sum(map(lambda p: gmap[p].halite_amount, get_positions(gmap)))
        my_ships = len(ships)
        total_ships = sum([len(player.get_ships()) for player in game.players.values()])
        other_ships = total_ships - my_ships
        expected_halite = my_ships * halite_remaining / total_ships if total_ships > 0 else 0
        expected_halite_1 = (my_ships + 1) * halite_remaining / (total_ships + 1)
        halite_gained = expected_halite_1 - expected_halite
        return halite_gained - constants.SHIP_COST

    @staticmethod
    def max_hpt(me, gmap, ship, available_positions):
        return max(map(lambda p: IncomeEstimation.hpt_of(me, gmap, ship, p), available_positions))


class ResourceAllocation:
    @staticmethod
    def goals_for_ships(me, gmap, ships, turns_remaining):
        available_positions = get_positions(gmap)
        scheduled_positions = set()
        n = len(ships)
        goals = [ships[i].position for i in range(n)]
        scheduled = [False for i in range(n)]
        dropoffs = {(me.shipyard.position.x, me.shipyard.position.y)}

        log('allocating stills')

        # schedule stills
        for i in range(n):
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
            closest_dropoff = me.shipyard.position
            for p in dropoffs:
                hpt_by_assignment[(p[0], p[1], i)] = IncomeEstimation.hpt_of(me, gmap, turns_remaining, ships[i], closest_dropoff, Position(*p))

            for ps in iterate_by_radius(gmap, ships[i].position.x, ships[i].position.y):
                ps -= scheduled_positions
                if len(ps) > 0:
                    best = max(ps - scheduled_positions, key=halite_by_pos.get)
                    hpt_by_assignment[(best[0], best[1], i)] = IncomeEstimation.hpt_of(me, gmap, turns_remaining, ships[i], closest_dropoff, Position(*best))

        log('sorting assignments')

        assignments = sorted(hpt_by_assignment, key=hpt_by_assignment.get, reverse=True)

        log('gathering assignments')

        for x, y, i in assignments:
            if scheduled[i] or (x, y) in scheduled_positions:
                continue
            
            goals[i] = Position(x, y)
            scheduled[i] = True
            unscheduled.remove(i)
            if (x, y) not in dropoffs:
                scheduled_positions.add((x, y))

            if len(unscheduled) == 0:
                break

        return goals

    @staticmethod
    def goal_for(me, gmap, ship, available_positions):
        return max(available_positions, key=lambda p: IncomeEstimation.hpt_of(me, gmap, ship, p))


class PathPlanning:
    @staticmethod
    def commands_for(me, gmap, ships, turns_remaining, goals):
        n = len(ships)
        current = [ships[i].position for i in range(n)]
        next_positions = [current[i] for i in range(n)]
        reservation_table = defaultdict(set)
        scheduled = [False for i in range(n)]
        dropoffs = {(me.shipyard.position.x, me.shipyard.position.y)}

        log('locking stills')

        # lock in anyone that wants to stay still
        for i in range(n):
            if current[i] == goals[i]:
                raw_pos, t = (current[i].x, current[i].y), 1
                if raw_pos in dropoffs and turns_remaining - t < width:
                    continue
                else:
                    reservation_table[t].add(raw_pos)
                scheduled[i] = True
        
        unscheduled = [i for i in range(n) if not scheduled[i]]
        total = len(unscheduled)

        log('planning paths')

        for q, i in enumerate(unscheduled):
            log('ship {} ({}/{})...'.format(i, q, total))
            path = PathPlanning.a_star(gmap, current[i], goals[i], ships[i].halite_amount, reservation_table)
            for raw_pos, t in path:
                if raw_pos in dropoffs and turns_remaining - t < width:
                    continue
                else:
                    reservation_table[t].add(raw_pos)
            next_positions[i] = Position(*path[1][0])

        log('paths planned')

        directions = [PathPlanning.direction_between(gmap, current[i], next_positions[i]) for i in range(n)]
        commands = [ships[i].move(directions[i]) for i in range(n)]
        return commands, next_positions

    @staticmethod
    def direction_between(gmap, a, b):
        for dir in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]:
            if gmap.normalize(a.directional_offset(dir)) == gmap.normalize(b):
                return dir

    @staticmethod
    def a_star(gmap, start, goal, halite, reservation_table, WINDOW=8):
        """windowed hierarchical cooperative a*"""
        def heuristic(p):
            return gmap.calculate_distance(p, goal)

        start_raw = normalize(start.x, start.y)
        goal_raw = normalize(goal.x, goal.y)

        log('{} -> {}'.format(start_raw, goal_raw))

        if start_raw == goal_raw:
            return [(start_raw, 0), (start_raw, 1)]

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

#            log('- Expanding {} at {}'.format(current_raw, t))

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

#                log('-- Adding {} at {}'.format(neighbor_raw, nt))

                came_from[npt] = (current_raw, t)
                g_score[neighbor_raw] = g
                h_score[neighbor_raw] = heuristic(neighbor)
                f_score[neighbor_raw] = g_score[neighbor_raw] + h_score[neighbor_raw]

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
        clear_log()
        log('Starting turn')
        queue = self.produce_commands(self.game.me, self.game.game_map)
        self.game.end_turn(queue)
        log('Turn took {}'.format(datetime.now() - start_time))

    def produce_commands(self, me, gmap):
        ships = list(me.get_ships())
        ships = sorted(ships, key=lambda s: gmap.calculate_distance(s.position, me.shipyard.position), reverse=True)
        ships = sorted(ships, key=lambda s: s.halite_amount, reverse=True)

        log('sorted ships')

        goals = ResourceAllocation.goals_for_ships(me, gmap, ships, self.turns_remaining)

        log('allocated goals')

        queue, next_positions = PathPlanning.commands_for(me, gmap, ships, self.turns_remaining, goals)

        log('planned paths')

        if me.halite_amount >= constants.SHIP_COST and not gmap[me.shipyard].is_occupied and me.shipyard.position not in next_positions:
            roi = IncomeEstimation.roi(self.game, me, gmap, ships)
            if roi > 0 and len(ships) < 50 and self.turns_remaining > 50:
                queue.append(me.shipyard.spawn())

        log('tried spawning')

        return queue


def main():
    global commander, width, height
    commander = Commander()
    width = commander.game.game_map.width
    height = commander.game.game_map.height
    while True:
        commander.run_once()

main()