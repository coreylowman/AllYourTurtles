import subprocess
import datetime
from statistics import mean


def extract_halite(line):
    return int(line.split(' ')[-2])


delta_by_config = {}

for opponents in [
    # 1,
    3
]:
    for size in [
        32,
        40,
        48,
        56,
        64
    ]:
        deltas = []
        for i in range(5):
            output = subprocess.check_output(
                'halite.exe -vvv --no-logs --width {} --height {} "python MyBot.py" {}'.format(
                    size, size, '"python MyBot_last.py" ' * opponents),
                stderr=subprocess.STDOUT,
                shell=True,
            )

            output = output.decode('utf-8')
            if '[error]' in output:
                raise ValueError(output)

            # print(output)
            lines = output.splitlines()
            for i, line in enumerate(lines):
                if line.startswith('[info] Opening a file at'):
                    break

            players = opponents + 1
            results = lines[i + 1:i + 1 + players]
            halite = extract_halite(results[0])
            opponents_halite = mean(list(map(extract_halite, results[1:])))
            deltas.append(halite - opponents_halite)
            print(datetime.datetime.now(), players, size, deltas[-1])
        delta_by_config[(opponents, size)] = mean(deltas)

for config, delta in delta_by_config.items():
    print(config, '\t', delta)
