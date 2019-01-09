import subprocess
import datetime
from statistics import mean


def extract_halite(line):
    return int(line.split(' ')[-2])


def run_game(size, opponents, bot, seed=None):
    if seed is None:
        seed = ''
    else:
        seed = '-s ' + seed
    output = subprocess.check_output(
        'halite.exe {} -vvv --no-logs --width {} --height {} {} {}'.format(
            seed, size, size, bot, '"python MyBot_last.py" ' * opponents),
        stderr=subprocess.STDOUT,
        shell=True,
    )

    output = output.decode('utf-8')
    if '[error]' in output:
        raise ValueError(output)

    # print(output)
    lines = output.splitlines()
    seed = lines[0].split(' ')[-1]

    i = 0
    while True:
        if lines[i].startswith('[info] Opening a file at'):
            break
        i += 1

    players = opponents + 1
    results = lines[i + 1:i + 1 + players]
    return seed, results


delta_by_config = {}

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
        for _ in range(5):
            print(datetime.datetime.now(), 1 + opponents, size)

            seed_new, results_new = run_game(size, opponents, '"python MyBot.py"', seed=None)
            halite = extract_halite(results_new[0])
            opponents_halite = list(map(extract_halite, results_new[1:]))
            ds_new = [halite - oh for oh in opponents_halite]
            print('\tnew:', ds_new)

            seed_old, results_old = run_game(size, opponents, '"python MyBot_last.py"', seed=seed_new)
            halite = extract_halite(results_old[0])
            opponents_halite = list(map(extract_halite, results_old[1:]))
            ds_old = [halite - oh for oh in opponents_halite]
            print('\told:', ds_old)
