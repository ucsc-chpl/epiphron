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

def generate_heatmap(coordinates, title):

 # workgroup and title extraction
 title_information = title.split(":", 1)
 workgroup_information = title_information[0].split(",")
 workgroup_size = int(workgroup_information[0])
 workgroups = int(workgroup_information[1])
 final_size = 0
 if 'local' in title_information[1]:
    final_size = 256
 elif workgroup_size * workgroups > 1024:
    final_size = 1024
 else:
    final_size = workgroup_size * workgroups
 grid_scale = int(math.log2(final_size)) + 1
 
 contention = grid_scale
 padding = grid_scale
 if 'local' in title_information[1]:
    padding = 4 # scaling to 8
 data_array = np.zeros((padding, contention))

 # Assign values to the data array based on coordinates
 for x, y, value, _ in coordinates:
    x_index = int(np.log2(x))
    y_index = int(np.log2(y))
    data_array[y_index, x_index] = value

 # Get the minimum and maximum values from the data_array
 data_min = math.floor(np.min(data_array))
 data_max = math.floor(np.max(data_array))


 # Set up the figure and axes
 fig_x = 10
 fig_y = 8
 if 'local' in title_information[1]:
    fig_x = 8
    fig_y = 6
 fig, ax = plt.subplots(figsize=(fig_x, fig_y), dpi=300)

 # Create the heatmap
 heatmap = ax.imshow(data_array, cmap='viridis', vmin=data_min, vmax=data_max)

 # Set appropriate axis labels
 plt.xlabel("Contention")
 plt.ylabel("Padding")

#  # Set the tick locations and labels for the x-axis
 x_ticks = [2 ** i for i in range(contention)]
 ax.set_xticks(np.arange(len(x_ticks)))
 ax.set_xticklabels(x_ticks, fontsize=7)

#  # Set the tick locations and labels for the y-axis
 y_ticks = [2 ** i for i in range(padding)]
 ax.set_yticks(np.arange(len(y_ticks)))
 ax.set_yticklabels(y_ticks, fontsize=7)

 # Add text annotations for data points
 # flag check here 
 #baseline = data_array[0][0]
 for i in range(data_array.shape[0]):
    for j in range(data_array.shape[1]):
        text = ax.text(j, i, int(data_array[i][j]),
                    ha="center", va="center", color="w", fontsize=6, path_effects=[pe.withStroke(linewidth=1, foreground="black")], weight='bold')

 # Customize the color bar range
 # flag check here 
 if 'local' in title_information[1]:
    cbar = plt.colorbar(heatmap, fraction=0.026, pad=0.04, ticks=[])
    cbar.set_label('Atomic Operations per Microsecond', rotation=270, labelpad=15)

    ax.invert_yaxis()  # Invert the y-axis
    
    description = title_information[1].split(", ")
    plt.text(-0.12, 1.2, description[0], transform=plt.gca().transAxes, fontsize=12, va='center')
    plt.text(-0.12, 1.13, "workgroup_size: "+ workgroup_information[0], transform=plt.gca().transAxes, fontsize=7)
    plt.text(-0.12, 1.08, "workgroups: "+ workgroup_information[1], transform=plt.gca().transAxes, fontsize=7)
    if 'cross_warp' in description[1]:
        plt.title("striding_access: local_atomic_fa_relaxed")
    plt.title(description[1])
 else:
    cbar = plt.colorbar(heatmap, fraction=0.046, pad=0.04, ticks=[])
    cbar.set_label('Atomic Operations per Microsecond', rotation=270, labelpad=15)

    ax.invert_yaxis()  # Invert the y-axis
    
    description = title_information[1].split(", ")
    plt.text(-0.32, 1.1, description[0], transform=plt.gca().transAxes, fontsize=12, va='center')
    plt.text(-0.32, 1.03, "workgroup_size: "+ workgroup_information[0], transform=plt.gca().transAxes, fontsize=7)
    plt.text(-0.32, 0.98, "workgroups: "+ workgroup_information[1], transform=plt.gca().transAxes, fontsize=7)
    plt.title(description[1])

 save_folder = "heatmaps"
 os.makedirs(save_folder, exist_ok=True)

 # Save the plot in the specified folder
 filename = os.path.join(save_folder, f"{title_information[1]},{title_information[0]}.png")

 plt.savefig(filename)

 plt.close()

def extract_coordinates_from_file(filename):
    coordinates = []
    current_title = ""

    with open(filename, 'r') as file:
        for line in file:
            if re.match(r"\(\d+, \d+, \d+.\d+\)", line) or "inf" in line:
                parts = line.strip("()\n").split(", ")
                x = int(parts[0])
                y = int(parts[1])
                value = float(parts[2])
                coordinates.append((x, y, value, current_title))
            else:
                current_title = line.strip()

    return coordinates

# File name
filename = "result.txt"
# Extract coordinates from the file
coordinates = extract_coordinates_from_file(filename)
titles = set(coord[3] for coord in coordinates)
for title in sorted(titles):
    if "random_access" not in title:
        graph_coordinates = [c for c in coordinates if c[3] == title]
        generate_heatmap(graph_coordinates, title)