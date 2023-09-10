#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <stdint.h>
#include <set>
#include <unordered_map>
#include <queue>
#include <cassert>
#include <cmath>
#include <sstream>
#include <string>
#include <chrono>

#include <easyvk.h>
#include "json.h"
#include "common.h"

#ifdef __ANDROID__
#define USE_VALIDATION_LAYERS false
#else
#define USE_VALIDATION_LAYERS true
#endif

using ordered_json = nlohmann::ordered_json;
using namespace std::chrono;

typedef struct Graph {
    uint32_t num_vertices;
    uint32_t num_edges;
    std::vector<uint32_t> vertices;
    std::vector<uint32_t> edges;
} Graph;

/**
 * Calculates the total size in bytes of all the Buffer objects in a vector.
 *
 * Useful for debug purposes to check how much memory the kernel is using.
 *
 * @param buffers A vector of Buffer objects.
 * @return The total size in bytes of all the Buffer objects in the vector.
 */
int getTotalBufferSize(const std::vector<easyvk::Buffer>& buffers) {
    int totalSize = 0;
    
    for (const easyvk::Buffer& buffer : buffers) {
        totalSize += buffer.size();
    }
    
    return totalSize;
}


/**
 * Generates a graph given a text file. The text file is expected to be in the following format:
 * 
 * vx vy
 * vz vw
 * ...
 * 
 * e.g each line represents an edge from vx to vy. The input graph is assumed to be undirected.
*/
Graph generateGraphFromFile(const std::string &filePath) {
    std::ifstream textFile(filePath);
    if (!textFile.is_open()) {
        std::cerr << "Failed to open file.\n";
        exit(1);
    }

    Graph g;
    g.num_vertices = 0;
    g.num_edges = 0;
    std::string line;
    std::set<uint32_t> seen; // Vertices we've seen.
    // Map of vertices seen in file to our own internal repr.
    std::unordered_map<uint32_t, uint32_t> vertexMap; 
    // Keeps track of what vertices each vertex is connected to.
    std::vector<std::set<int>> adjacency_sets;

    while (std::getline(textFile, line)) {
        std::istringstream lineStream(line);
        std::string vertex1, vertex2;
        lineStream >> vertex1 >> vertex2;

        try {
            // Convert string to uint32_t
            uint32_t v1 = static_cast<uint32_t>(std::stoul(vertex1));
            uint32_t v2 = static_cast<uint32_t>(std::stoul(vertex2));

            // If we've never seen these vertices before, track them and map to our own id
            if (seen.find(v1) == seen.end()) {
                // We've never seen this vertex before.
                vertexMap[v1] = g.num_vertices; // Assign to a vertex.
                g.num_vertices++;                     // incr counter
                seen.insert(v1);                      // mark as seen
                adjacency_sets.emplace_back();        // Reserve space in the adjacency map
            } 

            if (seen.find(v2) == seen.end()) {
                // We've never seen this vertex before.
                vertexMap[v2] = g.num_vertices; // Assign to a vertex.
                g.num_vertices++;                     // Incr counter
                seen.insert(v2);                      // Mark as seen
                adjacency_sets.emplace_back();        // Reserve space in the adjacency map
            } 
            // Re-assign to our own internal id.
            v1 = vertexMap[v1];
            v2 = vertexMap[v2];

            adjacency_sets[v1].insert(v2);
            adjacency_sets[v2].insert(v1);

        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid argument: " << e.what() << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "Out of range: " << e.what() << std::endl;
        }
    }

    // Convert the adjacency set to the packed representation.
    uint32_t edgeCount = 0;
    for (int i = 0; i < adjacency_sets.size(); i++) {
        g.vertices.push_back(edgeCount);
        for (const auto elem : adjacency_sets[i]) {
            g.edges.push_back(elem);
            edgeCount++;
        }
    }

    // An additional sentinel value which makes retrieving the edges of the last
    // vertex easier.
    g.vertices.push_back(edgeCount);
    g.num_edges = edgeCount;
    g.num_vertices = adjacency_sets.size();

    return g;
}


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
ordered_json baselineBfs(size_t deviceIndex, const Graph &g, uint32_t source) {
    ordered_json results;
    auto instance = easyvk::Instance(USE_VALIDATION_LAYERS);
    auto device = easyvk::Device(instance, instance.physicalDevices().at(deviceIndex));
    auto deviceName = device.properties.deviceName;
    // std::cout << "Using device: " << deviceName << "\n";

    // Load shader code.
    std::vector<uint32_t> spvCode = 
    #include "build/baseline_bfs.cinit"
    ;
    auto entry_point = "baselineBfs";

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
    results["globalMemUsageInMb"] = getTotalBufferSize(kernelInputs) / (double) 1000000;
    
    auto startTime = high_resolution_clock::now(); // TODO: Where should we start timing from?
    auto program = easyvk::Program(device, spvCode, kernelInputs);
    // Launch a thread per vertex.
    auto workgroupSize = 512;
    auto numWorkgroups = std::ceil((double) (g.num_vertices) / workgroupSize);
    std::cout << "numWorkgroups: " << numWorkgroups << ", workgroupSize: " << workgroupSize << std::endl;
    program.setWorkgroups(numWorkgroups);
    program.setWorkgroupSize(workgroupSize);
    program.initialize(entry_point); 

    int kernelInvocations = 0;
    auto totalGpuTime = 0;
    do {
        finished_buf.store<bool>(0, true);
        auto gpuTime = program.runWithDispatchTiming();
        totalGpuTime += gpuTime;
        curr_buf.store<uint32_t>(0, curr_buf.load<uint32_t>(0) + 1);
        kernelInvocations++;
    } while (!finished_buf.load<bool>(0));
    auto totalCpuTime = duration_cast<milliseconds>(high_resolution_clock::now() - startTime).count();

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

    // Save results.
    results["costs"] = costs;
    results["kernelInvocations"] = kernelInvocations;
    results["totalGpuTimeInMs"] = (double) totalGpuTime / 1000000;
    results["avgGpuTimeInMs"] = ((double) totalGpuTime / kernelInvocations) / 1000000.0; // Convert to ms.
    results["totalCpuTimeInMs"] = totalCpuTime;

    return results;
}


ordered_json warpBfs(size_t deviceIndex, const Graph &g, uint32_t source) {
    ordered_json results;
    auto instance = easyvk::Instance(USE_VALIDATION_LAYERS);
    auto device = easyvk::Device(instance, instance.physicalDevices().at(deviceIndex));
    auto deviceName = device.properties.deviceName;
    // std::cout << "Using device: " << deviceName << "\n";

    // Load shader code.
    std::vector<uint32_t> spvCode = 
    #include "build/warp_bfs.cinit"
    ;
    auto entry_point = "warp_bfs";

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

    results["globalMemUsageInMb"] = getTotalBufferSize(kernelInputs) / (double) 1000000;
    
    auto startTime = high_resolution_clock::now(); // TODO: Where should we start timing from?
    auto program = easyvk::Program(device, spvCode, kernelInputs);
    // Each workgroup for a chunk of the vertices.
    auto numWorkgroups = std::ceil((double) (g.num_vertices) / CHUNK_SIZE);
    auto workgroupSize = 32;
    std::cout << "numWorkgroups: " << numWorkgroups << ", workgroupSize: " << workgroupSize << std::endl;
    program.setWorkgroups(numWorkgroups);
    program.setWorkgroupSize(workgroupSize);
    program.initialize(entry_point); 

    int kernelInvocations = 0;
    auto totalGpuTime = 0;
    do {
        finished_buf.store<bool>(0, true);
        auto gpuTime = program.runWithDispatchTiming();
        totalGpuTime += gpuTime;
        curr_buf.store<uint32_t>(0, curr_buf.load<uint32_t>(0) + 1);
        kernelInvocations++;
    } while (!finished_buf.load<bool>(0));
    auto totalCpuTime = duration_cast<milliseconds>(high_resolution_clock::now() - startTime).count();

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

    // Save results.
    results["costs"] = costs;
    results["kernelInvocations"] = kernelInvocations;
    results["totalGpuTimeInMs"] = (double) totalGpuTime / 1000000;
    results["avgGpuTimeInMs"] = ((double) totalGpuTime / kernelInvocations) / 1000000.0; // Convert to ms.
    results["totalCpuTimeInMs"] = totalCpuTime;

    return results;
}


int main() {
    size_t deviceIndex = 0;
    // uint32_t seed = std::time(0);
    uint32_t seed = 0xcafebabe;

    // Graph g = generateGraphFromFile("datasets/musae-github/musae_git_edges.txt");
    Graph g = generateGraphFromFile("datasets/roadNet-CA/roadNet-CA.txt");
    std::cout << "Vertex count: " << g.num_vertices << "\n";
    std::cout << "Edge count: " << g.num_edges << "\n\n";

    // Graph g = generateRandomGraph(1024, 256, seed);
    // Run dot -Tpng graph.dot -o graph.png to visualize.
    // generateDOTFile(g, "graph.dot");

    auto sourceVertex = 0;
    auto startTime = high_resolution_clock::now();
    // auto cpuCosts = cpuBfs(g, sourceVertex);
    auto cpuTime = duration_cast<milliseconds>(high_resolution_clock::now() - startTime).count();
    std::cout << "cpuBFS time: " << cpuTime << "ms\n";

    std::cout << "\n";

    std::cout << "baselineBfs results:\n";
    auto baselineResults = baselineBfs(deviceIndex, g, sourceVertex);
    std::cout << "Number of kernel invocations: " << baselineResults["kernelInvocations"] << "\n";
    std::cout << "Total time (host + device): " << baselineResults["totalCpuTimeInMs"] << "ms\n";
    std::cout << "Total GPU time (device only): " << baselineResults["totalGpuTimeInMs"] << "ms\n";
    std::cout << "Total GPU global memory usage: " << baselineResults["globalMemUsageInMb"] << "mb\n";

    std::cout << "\n";

    std::cout << "warpBfs results:\n";
    auto warpResults = warpBfs(deviceIndex, g, sourceVertex);
    std::cout << "Number of kernel invocations: " << warpResults["kernelInvocations"] << "\n";
    std::cout << "Total time: (host + device): " << warpResults["totalCpuTimeInMs"] << "ms\n";
    std::cout << "Total GPU time (device only): " << warpResults["totalGpuTimeInMs"] << "ms\n";
    std::cout << "Total GPU global memory usage: " << warpResults["globalMemUsageInMb"] << "mb\n";

    // Check that the results match the cpu results.
    for (int i = 0; i < g.num_vertices; i++) {
        assert(baselineResults["costs"][i] == warpResults["costs"][i]);
    }

    // Save costs to text file.
    // std::ofstream outFile;
    // outFile.open("costs.txt");
    // // Check if the file is successfully opened
    // if (!outFile.is_open()) {
    //     std::cerr << "Error opening file!" << std::endl;
    //     return 1; // Return an error code
    // }
    // outFile << "# vertexID     baselineBfsCost     warpBfsCost\n";
    // for (int i = 0; i < g.num_vertices; i++) {
    //     outFile << "    " << i << "            " << baselineResults["costs"][i]
    //         << "            " << warpResults["costs"][i] << "\n";
    // }

    // outFile.close();

    return 0;
}