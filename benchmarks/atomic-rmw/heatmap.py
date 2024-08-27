import re, os, math
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as cl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def generate_heatmap(coordinates, title, filename):
    # workgroup and title extraction
    title_information = title.split(":", 1)
    workgroup_information = title_information[0].split(",")
    workgroup_size = int(workgroup_information[0])
    workgroups = int(workgroup_information[1])
    device_name = title_information[1]
    final_size = 0
    if 'local' in title_information[1]:
        final_size = 1024#256
    elif "random_access" in title_information[1].split(", ")[1]:
        final_size = 4096
    elif workgroup_size * workgroups > 1024:
        final_size = 1024
    elif 'Ryzen' in device_name:
        final_size = 16
    else:
        final_size = workgroup_size * workgroups
    grid_scale = int(math.log2(final_size)) + 1 #4096 for global random
    
    contention = grid_scale
    padding = grid_scale
    if 'local' in title_information[1]:
        padding = 6 # scaling to 8
    elif 'Ryzen' in device_name:
        contention = 5
        padding = 5
    data_array = np.zeros((padding, contention-3)) #-1 for random

    # Assign values to the data array based on coordinates
    for x, y, value, _ in coordinates:
        x_index = int(np.log2(x)-3)
        y_index = int(np.log2(y))
        print(x_index, y_index)
        data_array[y_index, x_index] = value

    # Get the minimum and maximum values from the data_array
    data_min = math.floor(np.min(data_array))
    data_max = math.floor(np.max(data_array))

    # Set up the figure and axes
    fig_x = 10
    fig_y = 8
    if 'local' in title_information[1]:
        fig_x = 8
        fig_y = 6
    fig, ax = plt.subplots(figsize=(fig_x, fig_y), dpi=300)

    # Create the heatmap
    cmap = 'gray'
    if "Ryzen" in device_name:
        cmap = cl.LinearSegmentedColormap.from_list('', ['black', 'darkorange', 'white'])
    elif "AMD" in device_name:
        cmap = cl.LinearSegmentedColormap.from_list('', ['black', 'red', 'white'])
    elif "NVIDIA" in device_name:
        cmap = cl.LinearSegmentedColormap.from_list('', ['black', 'lawngreen', 'white'])
    elif "Intel" in device_name:
        cmap = cl.LinearSegmentedColormap.from_list('', ['black', 'deepskyblue', 'white'])
    elif "Apple" in device_name:
        cmap = cl.LinearSegmentedColormap.from_list('', ['black', 'lightsteelblue', 'white'])
    
    heatmap = None
    if 'local' in title_information[1]:
        heatmap = ax.imshow(data_array, cmap=cmap, vmin=data_min, vmax=data_max)
    else:
        norm = cl.LogNorm(vmin=data_min, vmax=data_max)
        heatmap = ax.imshow(data_array, cmap=cmap, norm=norm)

    # Set appropriate axis labels
    plt.xlabel("Threads", fontsize=20, labelpad=10)
    plt.ylabel("Padding", fontsize=20, labelpad=-6)
    if 'local' in title_information[1]:
        plt.ylabel("Padding", fontsize=20, labelpad=5)

    #  # Set the tick locations and labels for the x-axis
    x_ticks = [2 ** i for i in range(3, contention)]
    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_xticklabels(x_ticks, fontsize=10)
    if 'local' in title_information[1]:
        ax.set_xticklabels(x_ticks, fontsize=18)

    #  # Set the tick locations and labels for the y-axis
    y_ticks = [2 ** i for i in range(padding)]
    ax.set_yticks(np.arange(len(y_ticks)))
    ax.set_yticklabels(y_ticks, fontsize=18)

    # Add text annotations for data points
    # flag check here 
    #baseline = data_array[0][0]
    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            pass
            # temporarily disabling heatmap labels
            text = ax.text(j, i, int(data_array[i][j]),
              ha="center", va="center", color="w", fontsize=6, path_effects=[pe.withStroke(linewidth=3.0, foreground="black")], weight='bold')


    ax.invert_yaxis()  # Invert the y-axis

    description = title_information[1].split(", ")
   
    # Thread access pattern
    tmp = ""
    if 'cross_warp' in description[1]:
       tmp = "Cross Warp"
    elif "contiguous_access" in description[1]:
       tmp = "Contiguous Access"
    elif "branched" in description[1]:
       tmp = "Branched"
    elif "random_access" in description[1]:
        tmp = "Random Access"

    if 'local' in title_information[1]:
       tmp += " (Local Memory)"
   
    # Operation type
    if 'atomic_fa' in description[1]:
       tmp += ": atomic_fetch_add"
    elif 'cas' in description[1]:
       tmp += ": atomic_compare_exchange"
    else:
        tmp += " " + description[1][description[1].find(':'):]
    
    # Removing underscores from operation because TeX font included in Matplotlib doesn't have that character
    tmp = tmp.replace("_", " ")

    if 'Ryzen' not in device_name:
        plt.title(f'{description[0]}\n{tmp}\nWorkgroups: ({workgroup_information[0]}, 1) x {workgroup_information[1]}', fontsize=20)
    else:
        plt.title(f'{description[0]}\n{tmp}\nLogical Processors: 16', fontsize=20)

    if 'local' in title_information[1]:
        data_step = math.floor((data_max - data_min) / 6)
        cbar_ticks = [math.floor(n) for n in range(data_min, data_max, data_step)]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = plt.colorbar(heatmap, cax=cax, ticks=cbar_ticks)
        cbar.set_label('Atomic Operations per Microsecond', rotation=270, labelpad=24, fontsize=14)
        cbar.ax.tick_params(labelsize=13)
    else:
        # Add colorbar with logarithmic ticks
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = plt.colorbar(heatmap, cax=cax)
        cbar.set_label('Atomic Operations per Microsecond', rotation=270, labelpad=24, fontsize=20)
        
        # Set the colorbar ticks to logarithmic scale
        cbar_ticks = np.logspace(np.log10(data_min), np.log10(data_max), num=7)
        cbar.set_ticks(cbar_ticks)
        cbar.ax.set_yticklabels([f'{int(t):d}' if t >= 1 else f'{t:.1e}' for t in cbar_ticks])
        cbar.ax.tick_params(labelsize=13)

    save_folder = "graphs"
    os.makedirs(save_folder, exist_ok=True)

    # Save the plot in the specified folder
    savedfilename = os.path.join(save_folder, filename.removesuffix(".txt") + ".svg")
    print(f"Saving '{savedfilename}'...")
    plt.savefig(savedfilename, format='svg', bbox_inches='tight')

    savedfilename_png = os.path.join(save_folder, filename.removesuffix(".txt") + ".png")
    print(f"Saving '{savedfilename_png}'...")
    plt.savefig(savedfilename_png, format='png', bbox_inches='tight')


    plt.close()

def extract_coordinates_from_file(filename):
    coordinates = []
    current_title = ""

    with open("results1/" + filename, 'r') as file:
        for line in file:
            if re.match(r"\(\d+, \d+, [\d.eE+-]+\)", line):
                parts = line.strip("()\n").split(", ")
                x = int(parts[0])
                y = int(parts[1])
                value = float(parts[2])
                coordinates.append((x, y, value, current_title))
            else:
                current_title = line.strip()
    return coordinates

def main():
    plt.rcParams["font.serif"] = ["Linux Libertine", "DejaVu Serif"]
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["mathtext.fontset"] = "cm"
    # File name
    for filename in filter(lambda r: r.endswith(".txt"), os.listdir("results1/")):
        print(f"Processing '{filename}'...")
        # Extract coordinates from the file
        coordinates = extract_coordinates_from_file(filename)
        titles = set(coord[3] for coord in coordinates)
        for title in sorted(titles):
            #if "random_access" not in title:
            graph_coordinates = [c for c in coordinates if c[3] == title]
            generate_heatmap(graph_coordinates, title, filename)

if __name__ == "__main__":
    main()