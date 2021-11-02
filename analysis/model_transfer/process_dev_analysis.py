"""
Generates space-separated file from `dev_analysis.py` output
that can then be used to generate Figure 3.
"""

import sys

f = open(sys.argv[1], 'r').readlines()

wait = ""
for line in f:
    if "RESULT" in line:
        values = line.split(":")
        size = int(values[-2])
        data = values[-1].split()
        names = values[-4].split("_")
        dataset = names[-4]
        num_train = int(names[-2])
        data_string = f"{dataset} {num_train} {size} {100 * float(data[-3]):.5f} {100 * float(data[-2]):.5f} {data[-1] if size != 500 else 20}"
        if size == 500:
            print (wait)
            wait = data_string
        else:
            print (data_string)
print (wait)
