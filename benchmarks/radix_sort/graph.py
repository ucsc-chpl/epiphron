import os
import matplotlib.pyplot as plt

def parse_data(file_path):
    data = {}
    current_label = ""
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if ":" in line:
                current_label = line.rstrip(":")
                data[current_label] = []
            else:
                values = line.strip("()\n").split(", ")
                num_items = int(values[0])
                throughput = float(values[1])
                print(throughput)
                data[current_label].append((num_items, throughput))
    
    return data

def plot_data(data, output_path):
    plt.figure(figsize=(10, 6))
    
    for label, values in data.items():
        num_items = [item[0] for item in values]
        throughput = [item[1] for item in values]  # Convert to Melements/sec
        plt.plot(num_items, throughput, marker='o', linestyle='-', label=label)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Elements (32-bit)', fontsize=12)
    plt.ylabel('Melements/sec', fontsize=12)
    plt.title('NVIDIA GeForce RTX 4070', fontsize=14) #Change this
    plt.legend(title="Legend", fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to the output file
    plt.savefig(output_path, format='png')
    plt.close()  # Close the plot to avoid display in case running in notebooks or other environments

if __name__ == "__main__":
    file_path = "result.txt"  # Update this to your file path
    
    # Extract the folder from file path and use it to save the plot
    folder = os.path.dirname(file_path)
    output_file = os.path.join(folder, "gpu_sort.png")
    
    # Parse the data and create the plot
    data = parse_data(file_path)
    plot_data(data, output_file)
    
    print(f"Graph saved to {output_file}")