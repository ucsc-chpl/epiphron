import re, os, math
import scienceplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

devices = ['AMD', 'NVIDIA', 'Intel', 'Apple']
x = np.arange(len(devices))
files = ['amd-contiguous-fetchadd.txt', 'nvidia-contiguous-fetchadd.txt', 'intel-contiguous-fetchadd.txt', 'apple-contiguous-fetchadd.txt']
mins = []
maxs = []
for f in files:
    lines = []
    with open(os.path.join('results', f)) as fp:
        lines = fp.readlines()
    lines = lines[1:] # skip first info line
    lines = [re.search(r'\(.*, .*, (.*)\)\n?', line).group(1) for line in lines] # extract rates from lines
    rates = [float(l) for l in lines]
    mins.append(min(rates))
    maxs.append(max(rates))

diffs = [maxs[i] / mins[i] for i in range(len(devices))]

plt.style.use('science')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Linux Libertine'
plt.rcParams['text.usetex'] = False

colors = ['#FF2C00', '#00B945', '#0C5DA5', '#FF9500']
dark_colors = ['#CC2200', '#008732', '#084173', '#CC7700']

fig, ax = plt.subplots(figsize=(4, 3))

plt.bar(x-0.15, mins, 0.3, color=dark_colors, edgecolor='black')
plt.bar(x+0.15, maxs, 0.3, color=colors, edgecolor='black')
plt.xticks(x, devices)
plt.xlabel('Devices')
plt.ylabel('Atomic Operations per Microsecond')
plt.yscale('log')
plt.ylim(1, (math.e ** 1.4) * max(maxs))

rects = ax.patches
labels = (['Min'] * len(devices)) + (['Max'] * len(devices))
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, label, ha='center', va='bottom')

plt.savefig('graphs/throughput-diffs.png', bbox_inches='tight')
plt.savefig('graphs/throughput-diffs.svg', format='svg', bbox_inches='tight')

fig, ax = plt.subplots(figsize=(4, 3))
plt.xticks(x, devices)
plt.xlabel('Devices')
plt.ylabel('Maximum throughput falloff')
plt.bar(x, diffs, 0.7, color=colors, edgecolor='black')
plt.yscale('log')
plt.ylim(1, (math.e ** 1.00) * max(diffs))

rects = ax.patches
labels = [f'{int(diffs[i])}x' for i in range(len(rects))]
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height, label, ha='center', va='bottom')

plt.savefig('graphs/throughput-multipliers.png', bbox_inches='tight')
plt.savefig('graphs/throughput-multipliers.svg', format='svg', bbox_inches='tight')

plt.show()

