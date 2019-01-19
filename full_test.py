import subprocess
import datetime
from statistics import mean
import random

BOT = '"python MyBot.py"'
BOT_LAST = '"python MyBot_last.py"'


def extract_halite(line):
    return int(line.split(' ')[-2])


def run_game(size, opponents, players, seed):
    output = subprocess.check_output(
        'halite.exe -s {} -vvv --no-logs --width {} --height {} {}'.format(
            seed, size, size, players),
        stderr=subprocess.STDOUT,
        shell=True,
    )

    output = output.decode('utf-8')
    if '[error]' in output:
        raise ValueError(output)

    # print(output)
    lines = output.splitlines()

    i = 0
    while True:
        # if lines[i].startswith('[warn]') and 'P0' in lines[i]:
        #     print(lines[i])
        if lines[i].startswith('[info] Opening a file at'):
            break
        i += 1

    players = opponents + 1
    results = lines[i + 1:i + 1 + players]
    return results


delta_by_config = {}

old_2p_ranks = []
old_4p_ranks = []
new_2p_ranks = []
new_4p_ranks = []
for size in [
    32,
    40,
    48,
    56,
    64
]:
    for opponents in [
        1,
        3
    ]:
        for i in range(5):
            seed = random.getrandbits(32)
            print(datetime.datetime.now(), 1 + opponents, size, seed)

            for i in range(opponents + 1):
                players = [BOT_LAST] * opponents
                players.insert(i, BOT)

                results_new = run_game(size, opponents, ' '.join(players), seed)

                halite = extract_halite(results_new[i])
                opponents_halite = list(map(extract_halite, results_new[:i] + results_new[i + 1:]))
                ds_new = [halite - oh for oh in opponents_halite]
                rank_new = sum(int(d > 0) for d in ds_new)
                print('\t', players, 'new:', ds_new, rank_new)

                if opponents == 1:
                    new_2p_ranks.append(rank_new)
                else:
                    new_4p_ranks.append(rank_new)
            if opponents == 1:
                old_2p_ranks.extend(list(range(opponents + 1)))
                print('\t\t2p:', mean(new_2p_ranks), mean(old_2p_ranks))
            else:
                old_4p_ranks.extend(list(range(opponents + 1)))
                print('\t\t4p:', mean(new_4p_ranks), mean(old_4p_ranks))
