import re, os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import scienceplots

def vendor_name(title):
    if "AMD" in title:
        return "AMD"
    elif "Intel" in title:
        return "Intel"
    elif "NVIDIA" in title:
        return "NVIDIA"
    elif "Apple" in title:
        return "Apple"

def extract_coordinates(filename):
    coordinates = []
    current_title = ""
    with open(filename, 'r') as file:
        for line in file:
            if re.match(r"\(\d+, \d+, \d+.\d+\)", line) or "inf" in line:
                parts = line.strip("()\n").split(", ")
                x = int(parts[0])
                value = float(parts[2])
                coordinates.append((x, value, current_title))
            else:
                current_title = line.strip()
    return coordinates

def main():
    plt.style.use(['science','no-latex'])
    plt.rcParams['xtick.minor.size'] = 0
    plt.rcParams['xtick.minor.width'] = 0
    plt.rcParams['ytick.minor.size'] = 0
    plt.rcParams['ytick.minor.width'] = 0

    # File name
    read_folder = "results/random_access/"
    for filename in filter(lambda p: p.endswith(".txt"), os.listdir(read_folder)):
        print(f"Processing '{filename}'...")
        # Extract coordinates from the file
        coordinates = extract_coordinates(os.path.join(read_folder, filename))
        titles = set(coord[2] for coord in coordinates)
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        lines_data = {}

        for title in sorted(titles):
            if "random_access:" in title:
                graph_coordinates = [c[0:2] for c in coordinates if c[2] == title]
                lines_data[vendor_name(title)] = graph_coordinates

        fig = plt.figure(figsize=(4, 4))  # Adjust figure size if needed

        for i, (title, graph_coordinates) in enumerate(lines_data.items()):
            contention_values = [coord[0] for coord in graph_coordinates]
            throughput_values = [coord[1] / graph_coordinates[0][1] for coord in graph_coordinates]
            plt.plot(contention_values, throughput_values, marker='o', linestyle='-', label=title, markersize=6)

        x_ticks = [2 ** i for i in range(0, 11)]  # Powers of 2 from 1 to 1024
        x_labels = [str(2 ** i) for i in range(0, 11)]
        plt.xscale('log')
        plt.xticks(x_ticks, x_labels)

        ax = fig.axes[0]

        if len(lines_data) > 1:
            plt.legend()
        plt.xlabel('# of Atomics')
        plt.ylabel('Relative atomic operation speedup')

        save_folder = "heatmaps"
        os.makedirs(save_folder, exist_ok=True)

        savedfilename = os.path.join(save_folder, filename.removesuffix(".txt") + ".svg")
        print(f"Saving '{savedfilename}'...")
        plt.savefig(savedfilename, format='svg', bbox_inches='tight')
        plt.savefig(savedfilename.removesuffix(".svg")+".png", bbox_inches="tight")

if __name__ == "__main__":
    main()