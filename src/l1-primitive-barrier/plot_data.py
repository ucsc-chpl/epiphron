import os
import json
import argparse
import matplotlib.pyplot as plt
import sys

def plot_from_json(filename):
    # Load JSON data from file
    try:
        with open(filename) as f:
            data = json.load(f)
    except IOError as e:
        print("Failed to open test results JSON!")
        exit(1)

    colors = ['red', 'green', 'blue']


    fig, ax = plt.subplots()
    # No Barrier
    no_barrier_results = data["noBarrier"]
    workgroup_sizes = [result["workgroupSize"] for result in no_barrier_results]
    avg_times = [result["avgTime"] for result in no_barrier_results]
    std_devs = [result["stdDev"] for result in no_barrier_results]
    # plot
    label = "no barrier"
    color = 'red'
    ax.errorbar(workgroup_sizes, avg_times, yerr=std_devs, fmt='o', capsize=5, color=color)
    ax.plot(workgroup_sizes, avg_times, marker='o', linestyle='-', color=color, label=label)

    # Local barrier
    local_barrier_results = data["localWorkgroupBarrier"]
    workgroup_sizes = [result["workgroupSize"] for result in local_barrier_results]
    avg_times = [result["avgTime"] for result in local_barrier_results]
    std_devs = [result["stdDev"] for result in local_barrier_results]
    # plot
    label = "local workgroup barrier"
    color = 'green'
    ax.errorbar(workgroup_sizes, avg_times, yerr=std_devs, fmt='o', capsize=5, color=color)
    ax.plot(workgroup_sizes, avg_times, marker='o', linestyle='-', color=color, label=label)

    # Global barrier
    global_barrier_results = data["globalWorkgroupBarrier"]
    workgroup_sizes = [result["workgroupSize"] for result in global_barrier_results]
    avg_times = [result["avgTime"] for result in global_barrier_results]
    std_devs = [result["stdDev"] for result in global_barrier_results]
    # plot
    label = "global workgroup barrier"
    color = 'blue'
    ax.errorbar(workgroup_sizes, avg_times, yerr=std_devs, fmt='o', capsize=5, color=color)
    ax.plot(workgroup_sizes, avg_times, marker='o', linestyle='-', color=color, label=label)

    # Subgroup local
    local_sub_res = data["localSubgroupBarrier"]
    workgroup_sizes = [result["workgroupSize"] for result in local_sub_res]
    avg_times = [result["avgTime"] for result in local_sub_res]
    std_devs = [result["stdDev"] for result in local_sub_res]
    # plot
    label = "local subgroup barrier"
    color = 'orange'
    ax.errorbar(workgroup_sizes, avg_times, yerr=std_devs, fmt='o', capsize=5, color=color)
    ax.plot(workgroup_sizes, avg_times, marker='o', linestyle='-', color=color, label=label)

    # Subgroup global
    global_sub_res = data["globalSubgroupBarrier"]
    workgroup_sizes = [result["workgroupSize"] for result in global_sub_res]
    avg_times = [result["avgTime"] for result in global_sub_res]
    std_devs = [result["stdDev"] for result in global_sub_res]
    # plot
    label = "global subgroup barrier"
    color = 'purple'
    ax.errorbar(workgroup_sizes, avg_times, yerr=std_devs, fmt='o', capsize=5, color=color)
    ax.plot(workgroup_sizes, avg_times, marker='o', linestyle='-', color=color, label=label)

    ax.set_xlabel('Workgroup Size')
    # ax.set_xscale('log', base=2)
    ax.set_ylabel('Time (\u03BCs)')
    # workgroup_size = data['deviceRuns'][0]['variedDispatch']['workGroupSize']
    device = data["deviceName"]
    ax.set_title(f'numWorkgroups={data["numWorkgroups"]}, {device}')
    ax.legend()
    ax.grid(True)

    output_filename = os.path.split(filename)[1].split('.')[0]
    fig.savefig(f'data/{output_filename}.png')

def plots_exist_for_json(json_file):
    # Check if a corresponding plot already exists for the given JSON file
    output_filename = os.path.split(json_file)[1].split('.')[0]
    return os.path.exists(f'data/{output_filename}.png')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: plot_data.py <json_filename_or_directory>")
        sys.exit(1)
    
    path = sys.argv[1]
    
    # Check if path is a directory or a file
    if os.path.isdir(path):
        # Loop over each JSON file in the directory
        for json_file in os.listdir(path):
            if json_file.endswith(".json") and not plots_exist_for_json(json_file):
                full_path = os.path.join(path, json_file)
                plot_from_json(full_path)
    elif os.path.isfile(path) and path.endswith(".json"):
        if not plots_exist_for_json(path):
            plot_from_json(path)
    else:
        print(f"Invalid input: {path}. Provide either a JSON file or a directory.")
        sys.exit(1)