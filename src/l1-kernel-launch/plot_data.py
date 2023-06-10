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
for i, deviceRun in enumerate(data["deviceRuns"]):
    # Varied dispatch
    # Extract relevant data from JSON
    varied_dispatch_results = deviceRun['variedDispatch']['results']
    workgroup_counts = [result['dispatchSize'] for result in varied_dispatch_results]
    avg_utils = [result['avgUtilPerTrial'] for result in varied_dispatch_results]
    std_devs = [result['stdDev'] for result in varied_dispatch_results]

    # Plot the data
    color = colors[i]
    label = deviceRun["deviceName"]
    ax.errorbar(workgroup_counts, avg_utils, yerr=std_devs, fmt='o', capsize=5, color=color)
    ax.plot(workgroup_counts, avg_utils, marker='o', linestyle='-', color=color, label=label)

ax.set_xlabel('Number of workgroups')
ax.set_xscale('log', base=2)
ax.set_ylabel('Utilization')
workgroup_size = data['deviceRuns'][0]['variedDispatch']['workGroupSize']
ax.set_title(f'Number of Workgroups vs. Utilization (workGroupSize={workgroup_size})')
ax.legend()
ax.grid(True)
fig.savefig('data/variedDispatch.png')

plt.clf()
        
fig, ax = plt.subplots()
for i, deviceRun in enumerate(data["deviceRuns"]):
    # Extract relevant data from JSON
    varied_thread_workload = deviceRun['variedThreadWorkload']['results']
    thread_workloads = [result['threadWorkload'] for result in varied_thread_workload]
    avg_utils = [result['avgUtilPerTrial'] for result in varied_thread_workload]
    std_devs = [result['stdDev'] for result in varied_thread_workload]

    # Plot the data
    color = colors[i]
    label = deviceRun["deviceName"]
    ax.errorbar(thread_workloads, avg_utils, yerr=std_devs, fmt='o', capsize=5, color=color)
    ax.plot(thread_workloads, avg_utils, marker='o', linestyle='-', color=color, label=label)

ax.set_xlabel('Thread workload')
ax.set_xscale('log', base=2)
ax.set_ylabel('Utilization')
num_workgroups = data['deviceRuns'][0]['variedThreadWorkload']['numWorkgroups']
workgroup_size = data['deviceRuns'][0]['variedThreadWorkload']['workGroupSize']
ax.set_title(f'Thread Workload vs. Utilization (numWorkgroups={num_workgroups}, workGroupSize={workgroup_size})')
ax.legend()
ax.grid(True)
fig.savefig('data/variedThreadWorkload.png')