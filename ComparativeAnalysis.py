from math import sqrt
import uuid

import networkx
import collections
import infomap
import numpy
import statistics

from networkx.algorithms.community.label_propagation import asyn_lpa_communities
from networkx.algorithms.community.quality import partition_quality
from networkx.algorithms.community import modularity

from timeit import default_timer as timer
from itertools import repeat, product


class Graph:
    def __init__(self, name):
        self.graph = networkx.Graph()
        self.name = name
        self.partition = []

    def getGraph(self):
        return self.graph
    
    def getName(self):
        return self.name
    
    def getPartition(self):
        return self.partition
    
    def setPartition(self, partition):
        self.partition = partition

    # Creates a partition as sets of edges from the community attribute in the graph set by the algorithms
    def updatePartition(self):
        self.partition = [[] for i in repeat(None,times=len(
        collections.Counter(list(networkx.get_node_attributes(self.graph, 'community').values())).keys()))]
        for k, v in networkx.get_node_attributes(self.getGraph(), 'community').items(): 
            self.partition[v].append(k)

    def createGraphFromEdgeList(self, filename):

        file = open(filename, 'r')    
        type = file.readline().split()

        for line in file.readlines():
            if type[0] == "d":
                self.graph = self.graph.to_directed()

            vertices = line.split()
            edge = (int(vertices[0]), int(vertices[1]))

            if type[1] == "w":
                self.graph.add_edge(*edge, weight = int(vertices[2]))
            else:
                self.graph.add_edge(*edge)

    def createGraphLFR(self, argumentsLFR):
            
        self.graph = networkx.LFR_benchmark_graph(argumentsLFR[0], argumentsLFR[1], argumentsLFR[2],
                                                    argumentsLFR[3], argumentsLFR[4], max_degree=30, max_iters=200000)
    
    def classify(self, report):
        analysis = "-----Analyses of the network \"" + self.getName()[6:-4] + "\"-----\n" +\
                        "Nodes: {}, Edges: {}, Self Loops: {}".format(self.graph.number_of_nodes(), self.graph.number_of_edges(), networkx.number_of_selfloops(self.graph)) + "\n" +\
                        "Graph Type: " + ("Directed and " if self.graph.is_directed() == True else "Undirected and ") +\
                        ("Weighted" if networkx.is_weighted(self.graph) == True else "Non-Weighted") + "\n"                            

        # Decide which ones stay after getting results
               
        # results.write("Number of connected components: {}".format(a.getNumberOfConnectedComponents(graph)))
        # results.write("Number of weakly connected components: {}".format(a.getNumberOfWeaklyConnectedComponents(graph)) if graph.is_directed() else "Weakly connected components not implemented for undirected case")
        # results.write("Number of Isolates: {}".format(a.getNumberOfIsolates(graph)))
        # results.write("Degree Centrality: {}".format(a.getDegreeCentrality(graph)))
        # results.write("Betweeness Centrality: {}".format(a.getBetweenessCentrality(graph)))
        # print(a.getNeighbours(graph,1))
        # for component in a.getConnectedComponents(graph):
        #     subgraph = Graph()
        #     for neighbours in component:
        #     print("Diameter of {} is: {}\n".format(component,"pass"))
        # results.write("Closeness centrality: {}".format(a.getClosenessCentrality(graph)))
        # results.write("Katz centrality: {}".format(a.getKatzCentrality(graph)))
        # results.write("Pagerank: {}".format(a.getPageRank(graph)))
        # results.write("Triangles: {}".format(a.getTriangles(graph)))
        # results.write("All Pairs Shortest Path: {}".format(a.getAllPairsShortestPath(graph)))
        # results.write("All Pairs Shortest Connectivity: {}".format(a.getAllPairsNodeConnectivity(graph)))
        # results.write("Network bridges: {}".format(a.getBridges(graph)))
        # results.write("All Connected Components: {}".format(a.getConnectedComponents(graph)))

        report.write(analysis)


class Analyser:
    def runTestSuite(self, infoMapArgumentsList, labelPropagationArgumentsList, analyser, graph, report):
        # InfoMap
        analysis = {}
        for infoMapArguments in infoMapArgumentsList:
            start = timer()
            analyser.InfoMap(graph, report, infoMapArguments)
            end = timer()

            report.write("----------InfoMap Stats----------\n" +
                        "InfoMap Parameters: " +
                         infoMapArguments +
                         "\nProcessing time: " +
                         str(end - start) +
                         "s" +
                         "\n")

            # Analyze partition quality
            analysis["infomap"] = analyser.ratePartition(graph, report)

        # Label Propagation 
        for labelPropagationArguments in labelPropagationArgumentsList:
            start = timer()
            analyser.LabelPropagation(graph, labelPropagationArguments)
            end = timer()

            report.write("------Label Propagation Stats------\n" +
                         "Processing time " +
                         str(end - start) +
                         "s" +
                         "\n")

            # Analyze partition quality
            analysis["label_prop"] = analyser.ratePartition(graph, report)

        report.write("\n")
        return analysis
        
    def InfoMap(self, graph, report, infoMapArguments):
        infomapWrapper = infomap.Infomap(infoMapArguments)

        for edge in graph.getGraph().edges():
            infomapWrapper.network.addLink(*edge)

        infomapWrapper.run()

        report.write("%d modules with codelength %f found\n" % (infomapWrapper.numTopModules(),
                                                                infomapWrapper.codelength))

        # Set the community of each node in the graph
        communities = {}
        for node in infomapWrapper.iterLeafNodes():
            communities[node.physicalId] = node.moduleIndex()

        networkx.set_node_attributes(graph.getGraph(), name='community', values=communities)
        
        # Make partition accessible by other methods
        graph.updatePartition()

    def LabelPropagation(self, graph, labelPropagationArguments):
        graph.setPartition(
            [
                list(s) for s in asyn_lpa_communities(
                    graph.getGraph(), labelPropagationArguments[0], labelPropagationArguments[1]
                )
            ]
        )
    
    def ratePartition(self, graph, report):
        sum_intra_density, sum_inter_density, diff_sum_intra_inter_densities, \
            expansion_mean, expansion_stdev = self.adapted_mancoridis_metric_and_expansion_metric(graph, report)
        print("     -> Mancoridis & Expansion Analyzed")
        coverage, performance = self.partition_quality(graph, report)
        print("     -> Coverage & Performance Analyzed")
        modularity = self.modularity(graph, report)
        print("     -> Modularity Analyzed")
        triangle_mean, triangle_stdev = self.triangle_participation_ratio(graph, report)
        print("     -> Triangle Participation Analyzed")

        return {
            "sum_intra_density": sum_intra_density,
            "sum_inter_density": sum_inter_density,
            "diff_sum_densities": diff_sum_intra_inter_densities,
            "expansion_mean": expansion_mean,
            "expansion_deviation": expansion_stdev,
            "modularity": modularity,
            "coverage": coverage,
            "performance": performance,
            "triangle_participation_mean": triangle_mean,
            "triangle_participation_deviation": triangle_stdev
        }

    def adapted_mancoridis_metric_and_expansion_metric(self, graph, report):
        sumIntraClusterDensity = 0
        sumInterClusterDensity = 0

        expansion = []

        for community in graph.getPartition():
            communityNodeNumber = len(community)

            # Single node communities not taken into account
            if len(community) == 1:
                continue                

            # Also known as Internal Density
            internalEdges = graph.getGraph().subgraph(community).number_of_edges()
            maxPossibleInternalEdges = communityNodeNumber * (communityNodeNumber - 1) / (1 if graph.getGraph().is_directed() == True else 2)
            sumIntraClusterDensity += internalEdges / maxPossibleInternalEdges

            # Also known as Cut Ration
            interClusterEdges = len(list(networkx.edge_boundary(graph.getGraph(), community, [x for x in graph.getGraph().nodes() if x not in community])))
            maxPossibleInterClusterEdges = communityNodeNumber * (graph.getGraph().size() - communityNodeNumber) * (2 if graph.getGraph().is_directed() == True else 1)
            sumInterClusterDensity += interClusterEdges / maxPossibleInterClusterEdges

            expansion += [interClusterEdges / communityNodeNumber]

        report.write(
            "Adapted Mancoridis metric: Total Intra Cluster Density {} - Total Inter Cluster Density {} = {}\n".
            format(sumIntraClusterDensity,
                    sumInterClusterDensity,
                    sumIntraClusterDensity - sumInterClusterDensity)
             )
        stdev = statistics.stdev(expansion)
        mean = statistics.mean(expansion)
        report.write("Community Expansion Average: {}\n".format(statistics.stdev(expansion)))
        report.write("Community Expansion Deviation: {}\n".format(statistics.mean(expansion)))
        return sumIntraClusterDensity, sumInterClusterDensity, sumIntraClusterDensity - sumInterClusterDensity, \
            mean, stdev

    def partition_quality(self, graph, report):
        quality = partition_quality(graph.getGraph(), graph.getPartition())
        report.write("Coverage: {}\n".format(quality[0]))
        report.write("Performance: {}\n".format(quality[1]))
        return quality[0], quality[1]

    def modularity(self, graph, report):
        mod = modularity(graph.getGraph(), graph.getPartition(), resolution=1)
        report.write("Modularity: {}\n".format(mod))
        return mod

    def triangle_participation_ratio(self, graph, report):
        triangleParticipationRatio = []

        for community in graph.getPartition():

            if len(community) < 4:
                triangleParticipationRatio += [0]
                continue     

            matrix = networkx.to_numpy_array(graph.getGraph().to_undirected().subgraph(community))
            if networkx.is_weighted(graph.getGraph()):
                matrix[matrix != 0] = 1

            triangleParticipationRatio += [numpy.trace(numpy.linalg.matrix_power(matrix, 3)) / 6]

        mean = statistics.mean(triangleParticipationRatio)
        stdev = statistics.stdev(triangleParticipationRatio)
        report.write("Triangle Participation Ratio Average: {}\n".format(mean))
        report.write("Triangle Participation Standard Deviation: {}\n".format(stdev))
        return mean, stdev


def main():
    report = open("report.txt", 'a')
    analyser = Analyser()
    
    edgeListModels = ["data/LesMiserables.txt"]

    infoMapArgumentsList = ["--two-level --directed"]
    labelPropagationArgumentsList = [[None, None]]

    # Tested options: n, tau1, tau2, mu, average_degree, min_community
    n = [150]# 2000, 4000, 6000, 8000]
    tau1 = [2, 3]
    tau2 = [1.05, 2]
    mu = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    average_degree = [7]
    
    argumentsLFR = list(list(combination) for combination in product(*[n, tau1, tau2, mu, average_degree]))
                        #if (combination[1] != 3 or combination[2] != 2))

    analysis = dict()

    for edgeListModel in edgeListModels:
        graph = Graph(edgeListModel)
        graph.createGraphFromEdgeList(edgeListModel)
        graph.classify(report)
        analysis[uuid.uuid4()] = analyser.runTestSuite(infoMapArgumentsList, labelPropagationArgumentsList, analyser, graph, report)

    for argumentLFR in argumentsLFR:
        graph = Graph("LFR")
        graph.createGraphLFR(argumentLFR)
        graph.classify(report)
        analyser.runTestSuite(infoMapArgumentsList, labelPropagationArgumentsList, analyser, graph, report)
        analysis[uuid.uuid4()] = analyser.runTestSuite(infoMapArgumentsList, labelPropagationArgumentsList, analyser, graph, report)

    print(analysis)
     

if __name__ == '__main__':
    main()
    