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
    title_information = title.split(":", 1)
    workgroup_information = title_information[0].split(",")
    description = title_information[1].split(", ")
    #print(description)
    plt.plot(contention_values, throughput_values, marker='o', linestyle='-', label=description[0], color=color, markersize=6)
    #plt.text(-0.15, 1.12, description[0], transform=plt.gca().transAxes, fontsize=12, va='center')
    #plt.text(0.9, 1.12, "workgroup_size: "+ workgroup_information[0], transform=plt.gca().transAxes, fontsize=7)
    #plt.text(0.9, 1.07, "workgroups: "+ workgroup_information[1], transform=plt.gca().transAxes, fontsize=7)
    plt.title(description[0] + "\nRandom Access: atomic_fetch_add\nWorkgroups: (" + workgroup_information[0] + ", 1) × " + workgroup_information[1])


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
GPU_info = ""

for i, title in enumerate(sorted(titles)):
    if "random_access:" in title:
        GPU_info = title.split(":", 1)[1]
        graph_coordinates = [c for c in coordinates if c[2] == title]
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

        filename = os.path.join(save_folder, f"{GPU_info}.png".replace(":", "-").replace(" ", "_"))
        plt.savefig(filename, format='png', bbox_inches='tight')
        plt.close()