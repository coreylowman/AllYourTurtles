import subprocess
import datetime
from statistics import mean
import random


def extract_halite(line):
    return int(line.split(' ')[-2])


def run_game(size, opponents, bot, seed):
    output = subprocess.check_output(
        'halite.exe -s {} -vvv --no-logs --width {} --height {} {} {}'.format(
            seed, size, size, bot, '"python MyBot_last.py" ' * opponents),
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

            results_new = run_game(size, opponents, '"python MyBot.py"', seed)

            halite = extract_halite(results_new[0])
            opponents_halite = list(map(extract_halite, results_new[1:]))
            ds_new = [halite - oh for oh in opponents_halite]
            rank_new = sum(int(d > 0) for d in ds_new)
            print('\tnew:', ds_new, rank_new)

            results_old = run_game(size, opponents, '"python MyBot_last.py"', seed)
            halite = extract_halite(results_old[0])
            opponents_halite = list(map(extract_halite, results_old[1:]))
            ds_old = [halite - oh for oh in opponents_halite]
            rank_old = sum(int(d > 0) for d in ds_old)
            print('\told:', ds_old, rank_old)

            if opponents == 1:
                new_2p_ranks.append(rank_new)
                old_2p_ranks.append(rank_old)
                print('\t\t2p:', mean(new_2p_ranks), mean(old_2p_ranks))
            else:
                new_4p_ranks.append(rank_new)
                old_4p_ranks.append(rank_old)
                print('\t\t4p:', mean(new_4p_ranks), mean(old_4p_ranks))
