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
    plt.xlabel("Contention", fontsize=16, labelpad=16)
    plt.ylabel("Padding", fontsize=16, labelpad=0)

    #  # Set the tick locations and labels for the x-axis
    x_ticks = [2 ** i for i in range(contention)]
    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_xticklabels(x_ticks, fontsize=13)

    #  # Set the tick locations and labels for the y-axis
    y_ticks = [2 ** i for i in range(padding)]
    ax.set_yticks(np.arange(len(y_ticks)))
    ax.set_yticklabels(y_ticks, fontsize=13)

    # Add text annotations for data points
    # flag check here 
    #baseline = data_array[0][0]
    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            pass
            # temporarily disabling heatmap labels
            #text = ax.text(j, i, int(data_array[i][j]),
            #    ha="center", va="center", color="w", fontsize=9, path_effects=[pe.withStroke(linewidth=1, foreground="black")], weight='bold')

    # Customize the color bar range
    # local mem color bar adjustments
    fraction = 0.046
    if 'local' in title_information[1]:
       fraction = 0.026
    cbar = plt.colorbar(heatmap, fraction=fraction, pad=0.03, ticks=[(data_max/8)*1.5,(data_max/8)*2.5,(data_max/8)*3.5,(data_max/8)*4.5,(data_max/8)*5.5,(data_max/8)*6.5,(data_max/8)*7.5])
    cbar.set_label('Atomic Operations per Microsecond', rotation=270, labelpad=24, fontsize=16)
    cbar.ax.tick_params(labelsize=13)

    ax.invert_yaxis()  # Invert the y-axis

    description = title_information[1].split(", ")
   
    # Thread access pattern
    tmp = ""
    if 'cross_warp' in description[1]:
       tmp = "Cross Warp"
    elif "contiguous_access" in description[1]:
       tmp = "Contiguous Access"
    elif "branched" in description[1]:
       tmp = "Branched"

    if 'local' in title_information[1]:
       tmp += " (Local Memory)"
   
    # Operation type
    if 'atomic_fa' in description[1]:
       tmp += ": atomic_fetch_add"
    elif 'cas' in description[1]:
       tmp += ": atomic_compare_exchange"
    else:
        tmp += description[1][description[1].find(':'):]

    plt.title(description[0] + "\n" + tmp + "\nWorkgroups: (" + workgroup_information[0] + ", 1) × " + workgroup_information[1], fontsize=20)


    save_folder = "heatmaps"
    os.makedirs(save_folder, exist_ok=True)

    # Save the plot in the specified folder
    filename = os.path.join(save_folder, f"{title_information[1]},{title_information[0]}.svg".replace(":", "-").replace(" ", "_"))

    print(filename)
    plt.savefig(filename, format='svg', bbox_inches='tight')

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