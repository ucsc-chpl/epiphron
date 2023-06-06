import json
import matplotlib.pyplot as plt

# Load JSON data from file
with open('data/results.json') as f:
    data = json.load(f)

# Vect Add
# Extract relevant data from JSON
vecAddResults = data['vectorAddResults']['results']
vec_sizes = [result['vecSize'] for result in vecAddResults]
avg_utils = [result['avgUtilPerTrial'] for result in vecAddResults]
std_devs = [result['stdDev'] for result in vecAddResults]

# Plot the data with error bars
plt.errorbar(vec_sizes, avg_utils, yerr=std_devs, fmt='o', capsize=5, color='blue')
plt.plot(vec_sizes, avg_utils, marker='o', linestyle='-', color='blue')
plt.xlabel('Dispatch Size')
plt.xscale('log', base=2)
plt.ylabel('Utilization')
plt.title('Vector (Dispatch) Size vs. Utilization')
plt.grid(True)
plt.savefig('vectAdd.png')
plt.clf()

# Fixed Dispatch
vecAddFixedDispatchResults = data['vectorAddFixedDispatchResults']['results']
kernelWorkloads = [result['kernelWorkload'] for result in vecAddFixedDispatchResults]
avg_utils = [result['avgUtilPerTrial'] for result in vecAddFixedDispatchResults]
std_devs = [result['stdDev'] for result in vecAddFixedDispatchResults]

plt.errorbar(kernelWorkloads, avg_utils, yerr=std_devs, fmt='o', capsize=5, color='red')
plt.plot(kernelWorkloads, avg_utils, marker='o', linestyle='-', color='red')
plt.xlabel('Kernel Workload')
plt.xscale('log', base=2)
plt.ylabel('Utilization')
plt.title('Kernel Workload vs Utilization (numWorkgroups=8)')
plt.grid(True)
plt.savefig('fixedDispatch.png')
plt.clf()