#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>


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

    int e_count = 0; // Keep track of how many edges we've generated.
    for (int i = 0; i < num_vertices; i++) {
        g.vertices.push_back(e_count);

        // Randomly decide the number of edges for the current vertex.
        int num_edges = std::rand() % (max_degree + 1);

        for (int j = 0; j < num_edges; j++) {
            // Randomly decide a vertex to connect to.
            int vertex_to_connect = std::random() % num_vertices;
            g_edges.push_back(vertex_to_connect);
            e_count++;
        }
    }

    // An additional sentinel value which makes retrieving the edges of the last
    // vertex easier.
    g_vertices.push_back(e_count);
    g.num_edges = e_count;

    return g;
}


int main() {
    std::cout << "Hello world!\n";

    Graph g = genderateRandomGraph(8, 4);


}