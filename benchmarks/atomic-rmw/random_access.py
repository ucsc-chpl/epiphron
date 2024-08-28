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
    elif "Samsung" in title:
        return "Samsung"
    elif "Mali" in title:
        return "ARM"

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

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def main():
    plt.style.use(['science','no-latex'])
    plt.rcParams['xtick.minor.size'] = 0
    plt.rcParams['xtick.minor.width'] = 0
    plt.rcParams['ytick.minor.size'] = 0
    plt.rcParams['ytick.minor.width'] = 0

    plt.rcParams["font.serif"] = ["Linux Libertine", "DejaVu Serif"]
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["mathtext.fontset"] = "cm"

    # File name
    read_folder = "results/random_access/"
    for filename in filter(lambda p: p.endswith(".txt"), os.listdir(read_folder)):
        print(f"Processing '{filename}'...")
        # Extract coordinates from the file
        coordinates = extract_coordinates(os.path.join(read_folder, filename))
        titles = [coord[2] for coord in coordinates]
        # Using scienceplots default colors, but in order of AMD/NVIDIA/Intel/Apple/Samsung/ARM
        colors = ['#FF2C00', '#00B945', '#78DBF9', '#CECECE', '#1515F4', '#0F85E6']
        lines_data = {}

        titles = f7(titles)

        for title in titles:
            if "random_access:" in title:
                graph_coordinates = [c[0:2] for c in coordinates if c[2] == title]
                lines_data[vendor_name(title)] = graph_coordinates

        fig = plt.figure(figsize=(4, 4))  # Adjust figure size if needed

        for i, (title, graph_coordinates) in enumerate(lines_data.items()):
            contention_values = [coord[0] for coord in graph_coordinates]
            throughput_values = [coord[1] / graph_coordinates[0][1] for coord in graph_coordinates]
            plt.plot(contention_values, throughput_values, marker='o', linestyle='-', label=title, markersize=6, color=colors[i])

        x_ticks = [2 ** i for i in range(0, 11)]  # Powers of 2 from 1 to 1024
        x_labels = [str(2 ** i) for i in range(0, 11)]
        plt.xscale('log')
        plt.xticks(x_ticks, x_labels)

        plt.legend()
        plt.xlabel('# of Atomics')
        plt.ylabel('Relative atomic operation speedup')

        save_folder = "graphs"
        os.makedirs(save_folder, exist_ok=True)
        
        svgfilename = os.path.join(save_folder, filename.removesuffix(".txt") + ".svg")
        pngfilename = os.path.join(save_folder, filename.removesuffix(".txt") + ".png")
        
        print(f"Saving '{svgfilename}'...")
        plt.savefig(svgfilename, format='svg', bbox_inches='tight')
        print(f"Saving '{pngfilename}'...")
        plt.savefig(pngfilename, format='png', bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    main()