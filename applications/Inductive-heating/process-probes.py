#!/usr/bin/env python3

import os.path

dim = 2
file_in = f"probes-temperature-{dim}d.txt"
file_out = f"results-temperature-{dim}d.dat"


with open(file_in) as f:
    data = f.readlines()


write_header = not os.path.exists(file_out)

with open(file_out, "a") as f:
    if write_header:
        print("Writing header")
        for id, line in enumerate(data):
            if line.startswith("#"):
                f.write(line)
            if not line.startswith("#"):
                f.write(line)
                break

    print("Writing last result")
    f.write(data[-1])

print("Done")
