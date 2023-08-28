import json
import matplotlib.pyplot as plt
import sys

def plot_from_json(filename):
    # Load JSON data
    with open(filename, 'r') as f:
        data = json.load(f)

    # Extract avgTime and timeStdDev from the JSON data
    kernel_avg_time = data['kernelBarrierResults']['avgTime']
    kernel_time_stddev = data['kernelBarrierResults']['timeStdDev']
    
    global_avg_time = data['globalBarrierResults']['avgTime']
    global_time_stddev = data['globalBarrierResults']['timeStdDev']

    # Data for plotting
    categories = ['Kernel Barrier', 'Global Barrier']
    avg_times = [kernel_avg_time, global_avg_time]
    std_devs = [kernel_time_stddev, global_time_stddev]

    # Create the plot
    plt.bar(categories, avg_times, yerr=std_devs, color=['blue', 'green'], align='center', alpha=0.7, capsize=10)
    plt.ylabel('Average Time')
    plt.title('Comparison of Average Times with Error Bars')
    
    # Save the plot
    plot_filename = filename.split(".")[0] + ".png"
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: script_name.py <json_filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    plot_from_json(filename)
