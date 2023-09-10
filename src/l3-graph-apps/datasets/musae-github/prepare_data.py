import csv
import gzip
from tqdm import tqdm

# Function to get the number of lines in the CSV file for tqdm progress bar
def count_lines(filename):
    with open(filename, 'r') as file:
        return sum(1 for line in file)

# Read the CSV file and convert it to a text file format
csv_file_path = 'musae_git_edges.csv' 
text_file_content = ''
line_count = count_lines(csv_file_path)

with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader, None)  # Skip the first row (header)
    for row in tqdm(csv_reader, total=line_count, desc='Processing'):
        vertex1, vertex2 = row  # Assuming the two columns in each row represent vertices connected by an edge
        text_file_content += f"{vertex1} {vertex2}\n"

# Save the converted content into a text file and compress it using gzip
compressed_file_path = 'musae_git_edges.gz'  # Replace with your desired output compressed file path
with gzip.open(compressed_file_path, 'wt', encoding='utf-8') as compressed_file:
    compressed_file.write(text_file_content)

print(f"Compressed text file saved to {compressed_file_path}")