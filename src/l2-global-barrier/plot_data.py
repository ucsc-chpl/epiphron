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

# Initialize an empty matrix
matrix = np.zeros((len(y_values), len(x_values)))

# Fill in the matrix
for item in data['results']:
    x_index = x_values.index(item['localMemSize'])
    y_index = y_values.index(item['workgroupSize'])
    matrix[y_index, x_index] = item['maxOccupancyBound']

# Create the heatmap
fig, ax = plt.subplots()
cax = ax.matshow(matrix, cmap='viridis')

# Add colorbar
cbar = fig.colorbar(cax)

# Set ticks
ax.set_xticks(np.arange(len(x_values)))
ax.set_yticks(np.arange(len(y_values)))

# Set tick labels
ax.set_xticklabels(x_values)
ax.set_yticklabels(y_values)

# Rotate x-axis labels for better readability
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Display actual values in the heatmap
for i in range(len(y_values)):
    for j in range(len(x_values)):
        ax.text(j, i, str(int(matrix[i, j])), ha='center', va='center', color='w')

ax.set_xlabel('localMemSize')
ax.set_ylabel('workgroupSize')


ax.set_title(f"Occupancy Bound - {data['deviceName']} - numWorkgroups = 4096")

plt.tight_layout()

plt.savefig('out.png')