#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <stdint.h>
#include <set>


typedef struct Graph {
    uint32_t num_vertices;
    uint32_t num_edges;
    std::vector<uint32_t> vertices;
    std::vector<uint32_t> edges;
} Graph;


Graph generateRandomGraph(uint32_t num_vertices, uint32_t max_degree) {
    // Initialize random seed.
    std::srand(std::time(0));

    Graph g;
    g.num_vertices = num_vertices;

    // Keeps track of what vertices each vertex is connected to.
    std::vector<std::set<int>> adjacency_sets(num_vertices);

    int e_count = 0; // Keep track of how many edges we've generated.
    for (int i = 0; i < num_vertices; i++) {
        g.vertices.push_back(e_count);

        // Randomly decide the number of edges for the current vertex.
        int num_edges = std::rand() % (max_degree + 1);
        int added_edges = 0;

        while (added_edges < num_edges) {
            // Randomly decide on a vertex to connect to.
            int vertex_to_connect = std::rand() % num_vertices;

            // Skip self-edges.
            if (vertex_to_connect == i) {
                continue;
            }

              // Check if this edge already exists.
            if (adjacency_sets[i].find(vertex_to_connect) == adjacency_sets[i].end() &&
                adjacency_sets[vertex_to_connect].find(i) == adjacency_sets[vertex_to_connect].end()) {
                
                // Keep track of adjacency in both directions.
                // This means we only store the edge from i to j if i < j.
                adjacency_sets[i].insert(vertex_to_connect);
                adjacency_sets[vertex_to_connect].insert(i);

                g.edges.push_back(vertex_to_connect);
                e_count++;
                added_edges++;
            }
        }
    }

    // An additional sentinel value which makes retrieving the edges of the last
    // vertex easier.
    g.vertices.push_back(e_count);
    g.num_edges = e_count;

    return g;
}


void generateDOTFile(const Graph &g, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for creating the DOT file.\n";
        return;
    }

    // Start the DOT description.
    file << "graph G {\n"; 

    for (size_t i = 0; i < g.vertices.size() - 1; ++i) {
        int start_index = g.vertices[i];
        int end_index = g.vertices[i + 1];
        for (int j = start_index; j < end_index; ++j) {
            // Write an edge from vertex 'i' to vertex 'E_A[j]'
            file << "  " << i << " -- " << g.edges[j] << ";\n";
        }
    }

    file << "}\n"; // End the DOT description
    file.close();

}


int main() {
    Graph g = generateRandomGraph(8, 4);
    generateDOTFile(g, "graph.dot");
}