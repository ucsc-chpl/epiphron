import networkx as nx
import matplotlib.pyplot as plt
import gzip
from tqdm import tqdm
from collections import Counter


# Function to get the number of lines in the compressed file for tqdm progress bar
def count_lines_gzip(filename):
    with gzip.open(filename, 'rt', encoding='utf-8') as file:
        return sum(1 for _ in file)


# Read the compressed graph text file and reconstruct the graph
compressed_file_path = 'musae_git_edges.txt.gz'  # Replace with your actual compressed file path
G = nx.Graph()
line_count = count_lines_gzip(compressed_file_path)

with gzip.open(compressed_file_path, 'rt', encoding='utf-8') as compressed_file:
    for line in tqdm(compressed_file, total=line_count, desc='Reading'):
        vertex1, vertex2 = line.strip().split()
        G.add_edge(vertex1, vertex2)

# Calculate the degree distribution
degree_sequence = [d for n, d in G.degree()]
degree_count = Counter(degree_sequence)
degree, count = zip(*sorted(degree_count.items()))

# Plot the degree distribution
plt.figure(figsize=(10, 6))
plt.loglog(degree, count, 'bo', markersize=5)
plt.title("Degree Distribution of musae-github")
plt.ylabel("# of Nodes")
plt.xlabel("Degree")
plt.grid(True)
filename = "degree_distribution.png"
plt.savefig(filename)

print("Saved plot to file " + filename)