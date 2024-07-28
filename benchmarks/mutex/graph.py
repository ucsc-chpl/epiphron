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
    plt.plot(contention_values, throughput_values, marker='o', linestyle='-', label=description[1], color=color, markersize=6)
    #plt.text(-0.15, 1.12, description[0], transform=plt.gca().transAxes, fontsize=12, va='center')
    plt.text(0.9, 1.12, "workgroup_size: "+ workgroup_information[0], transform=plt.gca().transAxes, fontsize=7)
    plt.text(0.9, 1.07, "workgroups: "+ workgroup_information[1], transform=plt.gca().transAxes, fontsize=7)


def extract_coordinates_from_file(filename):
    coordinates = []
    current_title = ""

    with open("results/" + filename, 'r') as file:
        for line in file:
            if re.match(r"\(\d+, \d+.\d+\)", line):
                parts = line.strip("()\n").split(", ")
                x = int(parts[0])
                value = float(parts[1])
                coordinates.append((x, value, current_title))
            else:
                current_title = line.strip()

    return coordinates

# File name
count = 0
for filename in filter(lambda r: r.endswith(".txt"), os.listdir("results/")):
    # Extract coordinates from the file
    coordinates = extract_coordinates_from_file(filename)
    titles = set(coord[2] for coord in coordinates)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i, title in enumerate(sorted(titles)):
        graph_coordinates = [c for c in coordinates if c[2] == title]
        generate_graph(graph_coordinates, title, colors[count % len(colors)])
        count += 1

x_ticks = [2 ** i for i in range(0, 8)]  # Powers of 2 from 1 to 1024
x_labels = [str(2 ** i) for i in range(0, 8)]
plt.xscale('log')
plt.xticks(x_ticks, x_labels)

plt.legend(title='Legend', loc='upper right')
plt.title('Lock implementations')
plt.xlabel('# of Locks')
plt.ylabel('Throughput')
plt.grid(True)
plt.legend()

save_folder = "graphs"
os.makedirs(save_folder, exist_ok=True)

# Save the plot in the specified folder
savedfilename = os.path.join(save_folder, "combined_plot.svg")
print(f"Saving '{savedfilename}'...")
plt.savefig(savedfilename, format='svg', bbox_inches='tight')

savedfilename_png = os.path.join(save_folder, "combined_plot.png")
print(f"Saving '{savedfilename_png}'...")
plt.savefig(savedfilename_png, format='png', bbox_inches='tight')
plt.close()