from operator import length_hint
from typing import Self
import networkx
import collections
import infomap

from networkx.algorithms.community.label_propagation import asyn_lpa_communities, label_propagation_communities
from networkx.algorithms.community.quality import partition_quality
import networkx.algorithms as algorithms
import networkx.algorithms.community.quality as measure
import matplotlib.pyplot as pyplot
import matplotlib.colors as colors

from timeit import default_timer as timer
from itertools import repeat


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
        self.partition = [[] for i in repeat(None, times=len(collections.Counter(list(networkx.get_node_attributes(self.graph, 'community').values())).keys()))]
        for k, v in networkx.get_node_attributes(self.getGraph(), 'community').items(): 
            self.partition[v].append(k)

    def createGraphFromEdgeList(self, filename):

        file = open(filename, 'r')

        for line in file.readlines():
            vertices = line.split()
            edge = (int(vertices[0]), int(vertices[1]))
            self.graph.add_edge(*edge)

    
    def createGraphLFR(self, n, tau1, tau2, mu, average_degree, min_degree, max_degree,
                       min_community, max_community, tol, max_iters, seed):
            
        self.graph = networkx.LFR_benchmark_graph(n, tau1, tau2, mu, average_degree, min_degree, max_degree,min_community, max_community, tol, max_iters, seed)

        return self.graph
    
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
    def InfoMap(self, graph, report, infoMapArguments):
        infomapWrapper = infomap.Infomap(infoMapArguments)

        for e in graph.getGraph().edges():
            infomapWrapper.network.addLink(*e)

        infomapWrapper.run()

        report.write("%d modules with codelength %f found\n" % (infomapWrapper.numTopModules(), infomapWrapper.codelength))

        # Set the community of each node in the graph
        communities = {}
        for node in infomapWrapper.iterLeafNodes():
            communities[node.physicalId] = node.moduleIndex()

        networkx.set_node_attributes(graph.getGraph(), name='community', values=communities)
        
        graph.updatePartition()

    def LabelPropagation(self, graph, labelPropagationArguments):
        graph.setPartition([list(s) for s in asyn_lpa_communities(graph.getGraph(), labelPropagationArguments[0], labelPropagationArguments[1])])
    
    def ratePartition(self, graph, report):
        self.adaptedMancoridisMetric(graph, report)
        self.partition_quality(graph, report)

    def adaptedMancoridisMetric(self, graph, report):
        sumIntraClusterDensity = 0
        sumInterClusterDensity = 0

        for community in graph.getPartition():
            communityNodeNumber = len(community)

            # Also known as Internal Density
            internalEdges = graph.getGraph().subgraph(community).number_of_edges()
            maxPossibleInternalEdges = communityNodeNumber * (communityNodeNumber - 1) / (1 if graph.getGraph().is_directed() == True else 2)
            sumIntraClusterDensity += internalEdges / maxPossibleInternalEdges

            # Also known as Cut Ration
            interClusterEdges = len(list(networkx.edge_boundary(graph.getGraph(), community, [x for x in graph.getGraph().nodes() if x not in community])))
            maxPossibleInterClusterEdges = communityNodeNumber * (graph.getGraph().size() - communityNodeNumber) * (2 if graph.getGraph().is_directed() == True else 1)
            sumInterClusterDensity += interClusterEdges / maxPossibleInterClusterEdges

        report.write("Adapted Mancoridis metric: sumIntraClusterDensity {} - sumInterClusterDensity {} = {}\n".
                     format(sumIntraClusterDensity, sumInterClusterDensity, sumIntraClusterDensity - sumInterClusterDensity))
    
    def partition_quality(self, graph, report):
        quality = partition_quality(graph.getGraph(), graph.getPartition())
        report.write("Coverage: {}\n".format(quality[0]))
        report.write("Performance: {}\n".format(quality[1]))
        

def main():
    report = open("report.txt", 'a')
    analyser = Analyser()
    
    edgeListModels = ["data//club.txt"]

    infoMapArgumentsList = ["--two-level --directed"]
    labelPropagationArgumentsList = [[None, None]]

    #TODO Use createGraphLFR to create graphs

    for edgeListModel in edgeListModels:
        graph = Graph(edgeListModel)
        graph.createGraphFromEdgeList(edgeListModel)
        graph.classify(report)

        #TODO Redirect infomap stdout to report
        # InfoMap
        for infoMapArguments in infoMapArgumentsList:
            start = timer()
            analyser.InfoMap(graph, report, infoMapArguments)
            end = timer()
            report.write("----------InfoMap Stats----------\n" +\
                        "InfoMap Parameters: " + infoMapArguments + "\nProcessing time: "+ str(end - start) + "s" + "\n")
            analyser.ratePartition(graph, report)

        # Label Propagation 
        for labelPropagationArguments in labelPropagationArgumentsList:
            start = timer()
            analyser.LabelPropagation(graph, labelPropagationArguments)
            end = timer()
            report.write("------Label Propagation Stats------\n" + "Processing time "+ str(end - start) + "s" + "\n")
            analyser.ratePartition(graph, report)

        report.write("\n")
     

if __name__ == '__main__':
    main()
    