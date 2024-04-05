# HEATMAP CODE
from ast import parse
from cmath import nan
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import matplotlib.ticker as ticker
import re
import os
import math

def generate_graph(coordinates, title, color):
    contention_values = [coord[0] for coord in coordinates]
    throughput_values = [coord[1] for coord in coordinates]
    plt.plot(contention_values, throughput_values, marker='o', linestyle='-', label=title, color=color, markersize=6)

def extract_coordinates_from_file(filename):
    coordinates = []
    current_title = ""

    with open(filename, 'r') as file:
        for line in file:
            if re.match(r"\(\d+, \d+, \d+.\d+\)", line) or "inf" in line:
                parts = line.strip("()\n").split(", ")
                x = int(parts[0])
                value = float(parts[2])
                coordinates.append((x, value, current_title))
            else:
                current_title = line.strip()

    return coordinates

# File name
filename = "result.txt"
# Extract coordinates from the file
coordinates = extract_coordinates_from_file(filename)
titles = set(coord[2] for coord in coordinates)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
lines_data = {}

for title in sorted(titles):
    if "random_access:" in title:
        graph_coordinates = [c for c in coordinates if c[2] == title]
        lines_data[title] = graph_coordinates

plt.figure(figsize=(10, 6))  # Adjust figure size if needed

for i, (title, graph_coordinates) in enumerate(lines_data.items()):
    generate_graph(graph_coordinates, title, colors[i % len(colors)])

x_ticks = [2 ** i for i in range(0, 11)]  # Powers of 2 from 1 to 1024
x_labels = [str(2 ** i) for i in range(0, 11)]
plt.xscale('log')
plt.xticks(x_ticks, x_labels)

plt.legend(title='Legend', loc='upper right')
plt.xlabel('# of Atomics')
plt.ylabel('Atomic operations per microsecond')
plt.grid(True)
plt.legend()

save_folder = "heatmaps"
os.makedirs(save_folder, exist_ok=True)

filename = os.path.join(save_folder, "combined_plot.png")
plt.savefig(filename, format='png', bbox_inches='tight')
plt.show()