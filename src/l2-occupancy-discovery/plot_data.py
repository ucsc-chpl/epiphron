import matplotlib.pyplot as plt
import numpy as np
import json
import sys

# Check if filename is provided
if len(sys.argv) < 2:
    print("Usage: script_name.py <filename>")
    sys.exit(1)

filename = sys.argv[1]

# Load data from the file
with open(filename, 'r') as file:
    data = json.load(file)

# Extract unique x and y axis values and sort them
x_values = sorted(list(set(item['localMemSize'] for item in data['results'])))
y_values = sorted(list(set(item['workgroupSize'] for item in data['results'])))

# Initialize empty matrices for maxOccupancyBound and CV
matrix_occupancy_bound = np.zeros((len(y_values), len(x_values)))
matrix_cv = np.zeros((len(y_values), len(x_values)))

# Fill in the matrices
for item in data['results']:
    x_index = x_values.index(item['localMemSize'])
    y_index = y_values.index(item['workgroupSize'])
    matrix_occupancy_bound[y_index, x_index] = item['maxOccupancyBound']
    matrix_cv[y_index, x_index] = item['CV']

# Create the heatmap for maxOccupancyBound
fig, ax = plt.subplots()
cax = ax.matshow(matrix_occupancy_bound, cmap='viridis')

# Add colorbar
cbar = fig.colorbar(cax)

# Set ticks
ax.set_xticks(np.arange(len(x_values)))
ax.set_yticks(np.arange(len(y_values)))

# Set tick labels
ax.set_xticklabels(x_values)
ax.set_yticklabels(y_values)

# Rotate x-axis labels for better readability
plt.setp(ax.get_xticklabels(), rotation=25, ha="center", rotation_mode="anchor", fontsize=6)

# Display actual values in the heatmap
for i in range(len(y_values)):
    for j in range(len(x_values)):
        ax.text(j, i, str(int(matrix_occupancy_bound[i, j])), ha='center', va='center', color='w', fontsize=6)

ax.set_xlabel('localMemSize')
ax.set_ylabel('workgroupSize')

ax.set_title(f"Occupancy Bound - {data['deviceName']}")

plt.tight_layout()

# Save plot for maxOccupancyBound
plot_filename_occupancy_bound = filename.split(".")[0] + "-occupancy-bound.png"
plt.savefig(plot_filename_occupancy_bound)
print(f"Plot saved as {plot_filename_occupancy_bound}")

# Create the heatmap for CV
fig, ax = plt.subplots()
cax = ax.matshow(matrix_cv, cmap='viridis')

# Add colorbar
cbar = fig.colorbar(cax)

# Set ticks
ax.set_xticks(np.arange(len(x_values)))
ax.set_yticks(np.arange(len(y_values)))

# Set tick labels
ax.set_xticklabels(x_values)
ax.set_yticklabels(y_values)

# Rotate x-axis labels for better readability
plt.setp(ax.get_xticklabels(), rotation=25, ha="center", rotation_mode="anchor", fontsize=6)

# Display actual values in the heatmap
for i in range(len(y_values)):
    for j in range(len(x_values)):
        ax.text(j, i, str(int(matrix_cv[i, j])), ha='center', va='center', color='w', fontsize=6)

ax.set_xlabel('localMemSize')
ax.set_ylabel('workgroupSize')

ax.set_title(f"Occupancy Bound coeff. vari. - {data['deviceName']}")

plt.tight_layout()

# Save plot for CV
plot_filename_cv = filename.split(".")[0] + "-cv.png"
plt.savefig(plot_filename_cv)
print(f"Plot saved as {plot_filename_cv}")
