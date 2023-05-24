import json
import matplotlib.pyplot as plt

# Load JSON data from file
with open('data/results.json') as f:
    data = json.load(f)

# Extract relevant data from JSON
results = data['vectorAddResults']['results']
vec_sizes = [result['vecSize'] for result in results]
avg_utils = [result['avgUtilPerTrial'] for result in results]
std_devs = [result['stdDev'] for result in results]

# Plot the data with error bars
plt.errorbar(vec_sizes, avg_utils, yerr=std_devs, fmt='o', capsize=5, color='blue')
plt.plot(vec_sizes, avg_utils, marker='o', linestyle='-', color='blue')
plt.xlabel('vecSize')
plt.xscale('log', base=2)
plt.ylabel('avgUtilPerTrial')
plt.title('Vector Size vs Average Utility per Trial')
plt.grid(True)
plt.savefig('out.png')