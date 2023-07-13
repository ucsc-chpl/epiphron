import os
import json
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('results_file', type=str, help='JSON file containing benchmark results')
args = parser.parse_args()

# Load JSON data from file
try:
    with open(args.results_file) as f:
        data = json.load(f)
except IOError as e:
    print("Failed to open test results JSON!")
    exit(1)

colors = ['red', 'green', 'blue']


fig, ax = plt.subplots()
# No Barrier
no_barrier_results = data["noBarrier"]["results"]
workgroup_sizes = [result["workgroupSize"] for result in no_barrier_results]
avg_times = [result["avgTime"] for result in no_barrier_results]
std_devs = [result["stdDev"] for result in no_barrier_results]
# plot
label = "no barrier"
color = 'red'
ax.errorbar(workgroup_sizes, avg_times, yerr=std_devs, fmt='o', capsize=5, color=color)
ax.plot(workgroup_sizes, avg_times, marker='o', linestyle='-', color=color, label=label)

# Local barrier
local_barrier_results = data["localWorkgroupBarrier"]["results"]
workgroup_sizes = [result["workgroupSize"] for result in local_barrier_results]
avg_times = [result["avgTime"] for result in local_barrier_results]
std_devs = [result["stdDev"] for result in local_barrier_results]
# plot
label = "local workgroup barrier"
color = 'green'
ax.errorbar(workgroup_sizes, avg_times, yerr=std_devs, fmt='o', capsize=5, color=color)
ax.plot(workgroup_sizes, avg_times, marker='o', linestyle='-', color=color, label=label)

# Global barrier
global_barrier_results = data["globalWorkgroupBarrier"]["results"]
workgroup_sizes = [result["workgroupSize"] for result in global_barrier_results]
avg_times = [result["avgTime"] for result in global_barrier_results]
std_devs = [result["stdDev"] for result in global_barrier_results]
# plot
label = "global workgroup barrier"
color = 'blue'
ax.errorbar(workgroup_sizes, avg_times, yerr=std_devs, fmt='o', capsize=5, color=color)
ax.plot(workgroup_sizes, avg_times, marker='o', linestyle='-', color=color, label=label)

# Subgroup local
local_sub_res = data["localSubgroupBarrier"]["results"]
workgroup_sizes = [result["workgroupSize"] for result in local_sub_res]
avg_times = [result["avgTime"] for result in local_sub_res]
std_devs = [result["stdDev"] for result in local_sub_res]
# plot
label = "local subgroup barrier"
color = 'orange'
ax.errorbar(workgroup_sizes, avg_times, yerr=std_devs, fmt='o', capsize=5, color=color)
ax.plot(workgroup_sizes, avg_times, marker='o', linestyle='-', color=color, label=label)

# Subgroup global
global_sub_res = data["globalSubgroupBarrier"]["results"]
workgroup_sizes = [result["workgroupSize"] for result in global_sub_res]
avg_times = [result["avgTime"] for result in global_sub_res]
std_devs = [result["stdDev"] for result in global_sub_res]
# plot
label = "global subgroup barrier"
color = 'purple'
ax.errorbar(workgroup_sizes, avg_times, yerr=std_devs, fmt='o', capsize=5, color=color)
ax.plot(workgroup_sizes, avg_times, marker='o', linestyle='-', color=color, label=label)



# for i, deviceRun in enumerate(data["deviceRuns"]):
#     # Varied dispatch
#     # Extract relevant data from JSON
#     varied_dispatch_results = deviceRun['variedDispatch']['results']
#     workgroup_counts = [result['dispatchSize'] for result in varied_dispatch_results]
#     avg_utils = [result['avgUtilPerTrial'] for result in varied_dispatch_results]
#     std_devs = [result['stdDev'] for result in varied_dispatch_results]

#     # Plot the data
#     color = colors[i]
#     label = deviceRun["deviceName"]
#     ax.errorbar(workgroup_counts, avg_utils, yerr=std_devs, fmt='o', capsize=5, color=color)
#     ax.plot(workgroup_counts, avg_utils, marker='o', linestyle='-', color=color, label=label)

ax.set_xlabel('Workgroup Size.')
# ax.set_xscale('log', base=2)
ax.set_ylabel('Time (\u03BCs)')
# workgroup_size = data['deviceRuns'][0]['variedDispatch']['workGroupSize']
device = data["noBarrier"]["deviceName"]
ax.set_title(f'numWorkgroups={data["numWorkgroups"]}, {device}')
ax.legend()
ax.grid(True)

output_filename = os.path.split(args.results_file)[1].split('.')[0]
fig.savefig(f'data/{output_filename}.png')

# plt.clf()
        
# fig, ax = plt.subplots()
# for i, deviceRun in enumerate(data["deviceRuns"]):
#     # Extract relevant data from JSON
#     varied_thread_workload = deviceRun['variedThreadWorkload']['results']
#     thread_workloads = [result['threadWorkload'] for result in varied_thread_workload]
#     avg_utils = [result['avgUtilPerTrial'] for result in varied_thread_workload]
#     std_devs = [result['stdDev'] for result in varied_thread_workload]

#     # Plot the data
#     color = colors[i]
#     label = deviceRun["deviceName"]
#     ax.errorbar(thread_workloads, avg_utils, yerr=std_devs, fmt='o', capsize=5, color=color)
#     ax.plot(thread_workloads, avg_utils, marker='o', linestyle='-', color=color, label=label)

# ax.set_xlabel('Thread workload')
# ax.set_xscale('log', base=2)
# ax.set_ylabel('Utilization')
# num_workgroups = data['deviceRuns'][0]['variedThreadWorkload']['numWorkgroups']
# workgroup_size = data['deviceRuns'][0]['variedThreadWorkload']['workGroupSize']
# ax.set_title(f'Thread Workload vs. Utilization (numWorkgroups={num_workgroups}, workGroupSize={workgroup_size})')
# ax.legend()
# ax.grid(True)
# fig.savefig('data/variedThreadWorkload.png')