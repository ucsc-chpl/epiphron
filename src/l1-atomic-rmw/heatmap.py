# HEATMAP CODE
from ast import parse
from cmath import nan
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import re
import os
import math

def generate_heatmap(coordinates, title):
 # Create an empty data array of size 8x8
 data_array = np.zeros((9, 9))



 # Assign values to the data array based on coordinates
 for x, y, value, _ in coordinates:
    x_index = int(np.log2(x))
    y_index = int(np.log2(y))
    data_array[y_index, x_index] = value

 # Get the minimum and maximum values from the data_array
 data_min = math.floor(np.min(data_array))
 data_max = math.floor(np.max(data_array))

 # Set up the figure and axes
 fig, ax = plt.subplots()

 # Create the heatmap
 heatmap = ax.imshow(data_array, cmap='viridis', vmin=data_min, vmax=data_max)

 # Set appropriate axis labels
 plt.xlabel("Contention")
 plt.ylabel("Padding")

 # Set the tick locations and labels for the x-axis
 x_ticks = [2 ** i for i in range(9)]
 ax.set_xticks(np.arange(len(x_ticks)))
 ax.set_xticklabels(x_ticks)

 # Set the tick locations and labels for the y-axis
 y_ticks = [2 ** i for i in range(9)]
 ax.set_yticks(np.arange(len(y_ticks)))
 ax.set_yticklabels(y_ticks)

 # Add text annotations for data points
 for i in range(data_array.shape[0]):
    for j in range(data_array.shape[1]):
        # if ((j == 6 and i == 0) or (j == 7 and i == 0) or (j == 7 and i == 1)):
        #     text = ax.text(j, i, "N/A",
        #                ha="center", va="center", color="w", fontsize=8)
        #else:
            text = ax.text(j, i, int(data_array[i][j]),
                       ha="center", va="center", color="w", fontsize=8)

 # Customize the color bar range
 cbar = plt.colorbar(heatmap, fraction=0.046, pad=0.04, ticks=[data_min, math.floor(data_max/3), math.floor(data_max/2), data_max])
 cbar.set_label("Avg # of Operations per Millisecond")

 ax.invert_yaxis()  # Invert the y-axis

 description = title.split(", ")
 plt.text(-0.42, 1.1, description[0], transform=plt.gca().transAxes, fontsize=12, va='center')
 plt.title(description[1])

 save_folder = "heatmaps"
 os.makedirs(save_folder, exist_ok=True)

 # ... code to process and plot the data ...

 # Save the plot in the specified folder
 filename = os.path.join(save_folder, f"{title}.png")

 plt.savefig(filename)

 plt.close()

 #plt.show()

def extract_coordinates_from_file(filename):
    coordinates = []
    current_title = ""

    with open(filename, 'r') as file:
        for line in file:
            if re.match(r"\(\d+, \d+, \d+.\d+\)", line) or "inf" in line:
                parts = line.strip("()\n").split(", ")
                x = int(parts[0])
                y = int(parts[1])
                # value = 0.0
                # if (parts[2] == "inf"): # Special check for fa_relaxed
                #     value = 65536.0
                # else:
                value = float(parts[2])
                coordinates.append((x, y, value, current_title))
            else:
                current_title = line.strip() # get workgroup size here

    return coordinates

# File name
filename = "result.txt"
# Extract coordinates from the file
coordinates = extract_coordinates_from_file(filename)
titles = set(coord[3] for coord in coordinates)
for title in sorted(titles):
    graph_coordinates = [c for c in coordinates if c[3] == title]
    generate_heatmap(graph_coordinates, title)