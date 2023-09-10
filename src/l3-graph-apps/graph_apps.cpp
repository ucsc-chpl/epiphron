#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <stdint.h>
#include <set>
#include <queue>
#include <cassert>
#include <cmath>

#include <easyvk.h>

#ifdef __ANDROID__
#define USE_VALIDATION_LAYERS false
#else
#define USE_VALIDATION_LAYERS true
#endif


typedef struct Graph {
    uint32_t num_vertices;
    uint32_t num_edges;
    std::vector<uint32_t> vertices;
    std::vector<uint32_t> edges;
} Graph;


Graph generateRandomGraph(uint32_t num_vertices, uint32_t max_degree, uint32_t seed) {
    // Initialize random seed.
    std::srand(seed);

    Graph g;
    g.num_vertices = num_vertices;

    // Keeps track of what vertices each vertex is connected to.
    std::vector<std::set<int>> adjacency_sets(num_vertices);

    int e_count = 0; // Keep track of how many edges we've generated.
    for (int i = 0; i < num_vertices; i++) {
        // std::cout << "Generating edges from vertex " << i << "\n";

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

                // g.edges.push_back(vertex_to_connect);
                // std::cout << i << " -- " << vertex_to_connect << "\n";
                e_count++;
                added_edges++;
            }
        }

        // std::cout << "\n\n";
    }

    // Convert the adjacency set to the packed representation.
    e_count = 0;
    for (int i = 0; i < adjacency_sets.size(); i++) {
        g.vertices.push_back(e_count);
        for (const auto elem : adjacency_sets[i]) {
            g.edges.push_back(elem);
            e_count++;
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
            // Our data structure stores both directions of the edge.
            // Onl display one.
            if (i < g.edges[j]) {
                // Write an edge from vertex 'i' to vertex 'E_A[j]'
                file << "  " << i << " -- " << g.edges[j] << ";\n";
            }
        }
    }

    file << "}\n"; // End the DOT description
    file.close();
}

// Performs DFS on the given graph from a source vertex.
// A cost array is returned which cost[i] represents the cost of the shortest path
// from the source vertex to vertex i.
std::vector<uint32_t> cpuBfs(const Graph &g, uint32_t source) {
    assert(source < g.num_vertices);

    std::vector<uint32_t> costs(g.num_vertices, UINT32_MAX);
    costs[source] = 0; // cost from source to itself is 0.

    std::set<uint32_t> visited;
    std::queue<uint32_t> unvisited; // Queue of nodes to be visited.

    unvisited.push(source);

    while (!unvisited.empty()) {
        // Choose a vertex to visit.
        uint32_t curr = unvisited.front();
        unvisited.pop();

        visited.insert(curr);

        // Enqueue all of the unvisited neighbors.
        for (int i = g.vertices[curr]; i < g.vertices[curr+1]; i++) {
            auto neighbor = g.edges[i];

            // Update cost of neighbor if this path is shorter.
            if (costs[curr] + 1 < costs[neighbor]) {
                costs[neighbor] = costs[curr] + 1;
            }

            // Enqueue the neighbor only if it has not been visited yet
            if (visited.find(neighbor) == visited.end()) {
                unvisited.push(neighbor);
            }
        }
    }

    return costs;
}

// Performs BFS on the given graph using the algorithm described in:
// Pawan Harish and P. J. Narayanan. Accelerating large graph algorithms on the GPU using CUDA.
std::vector<uint32_t> baselineBfs(const Graph &g, uint32_t source) {
    auto instance = easyvk::Instance(USE_VALIDATION_LAYERS);
    auto deviceIndex = 0;
    auto device = easyvk::Device(instance, instance.physicalDevices().at(deviceIndex));
    auto deviceName = device.properties.deviceName;
    std::cout << "Using device: " << deviceName << "\n";

    // Load shader code.
    std::vector<uint32_t> spvCode = 
    #include "build/baseline.cinit"
    ;
    auto entry_point = "baseline_bfs";

    // Set up kernel buffers.
    auto num_vertices_buf = easyvk::Buffer(device, 1, sizeof(uint32_t));
    num_vertices_buf.store<uint32_t>(0, g.num_vertices);
    auto curr_buf = easyvk::Buffer(device, 1, sizeof(uint32_t));
    curr_buf.store<uint32_t>(0, 0);
    auto costs_buf = easyvk::Buffer(device, g.num_vertices, sizeof(uint32_t));
    for (int i = 0; i < g.num_vertices; i++) {
        if (i == source) {
            costs_buf.store<uint32_t>(i, 0);
            continue;
        } 

        costs_buf.store<uint32_t>(i, UINT32_MAX);
    }
    auto vertices_buf = easyvk::Buffer(device, g.num_vertices+1, sizeof(uint32_t));
    for (int i = 0; i < g.num_vertices+1; i++) {
        vertices_buf.store<uint32_t>(i, g.vertices.at(i));
    }
    auto edges_buf = easyvk::Buffer(device, g.num_edges, sizeof(uint32_t));
    for (int i = 0; i < g.num_edges; i++) {
        edges_buf.store<uint32_t>(i, g.edges.at(i));
    }
    auto finished_buf = easyvk::Buffer(device, 1, sizeof(bool));
    finished_buf.store<bool>(0, false);

    std::vector<easyvk::Buffer> kernelInputs = {num_vertices_buf,
                                                curr_buf, 
                                                costs_buf,
                                                vertices_buf,
                                                edges_buf,
                                                finished_buf};
    
    auto program = easyvk::Program(device, spvCode, kernelInputs);
    // Launch a thread per vertex.
    auto workgroupSize = 256;
    auto numWorkgroups = std::ceil((double) (g.num_vertices) / workgroupSize);
    std::cout << "numWorkgroups: " << numWorkgroups << ", workgroupSize: " << workgroupSize << std::endl;
    std::cout << "Total work size: " << numWorkgroups * workgroupSize << "\n";
    program.setWorkgroups(numWorkgroups);
    program.setWorkgroupSize(workgroupSize);
    program.initialize(entry_point); // TODO: Make a change to easvk which ensures you do this.

    do {
        finished_buf.store<bool>(0, true);
        program.run();
        curr_buf.store<uint32_t>(0, curr_buf.load<uint32_t>(0) + 1);
    } while (!finished_buf.load<bool>(0));

    // Copy costs array.
    std::vector<uint32_t> costs;
    for (int i = 0; i < g.num_vertices; i++) {
        costs.emplace_back(costs_buf.load<uint32_t>(i));
    }

    finished_buf.teardown();
    num_vertices_buf.teardown();
    curr_buf.teardown();
    vertices_buf.teardown();
    edges_buf.teardown();
    costs_buf.teardown();
    program.teardown();
    device.teardown();
    instance.teardown();

    return costs;
}


int main() {
    // uint32_t seed = std::time(0);
    uint32_t seed = 0xcafebabe;
    Graph g = generateRandomGraph(512, 64, seed);
    // Run dot -Tpng graph.dot -o graph.png to visualize.
    generateDOTFile(g, "graph.dot");

    auto sourceVertex = 0;
    auto cpuCosts = cpuBfs(g, sourceVertex);
    auto baselineCosts = baselineBfs(g, sourceVertex);

    // Check that the results match the cpu results.
    for (int i = 0; i < g.num_vertices; i++) {
        assert(cpuCosts[i] == baselineCosts[i]);
    }

    // for (int i = 0; i < g.num_vertices; i++) {
    //     std::cout << i << ": " << costs[i] << "\n";
    // }

}