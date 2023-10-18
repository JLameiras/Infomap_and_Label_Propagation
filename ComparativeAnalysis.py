from typing import Self
import networkx
import collections
import infomap

from networkx.algorithms.community import asyn_lpa_communities
import networkx.algorithms as algorithms
import networkx.algorithms.community.quality as measure
import matplotlib.pyplot as pyplot
import matplotlib.colors as colors

from timeit import default_timer as timer


class Graph:
    def __init__(self, name):
        self.graph = networkx.Graph()
        self.name = name

    def getGraph(self):
        return self.graph
    
    def getName(self):
        return self.name

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
        analysis = "Analyses of the network " + self.getName() + "\n" +\
                        "Nodes: {}, Edges: {}, Self Loops: {}".format(self.graph.number_of_nodes(), self.graph.number_of_edges(), networkx.number_of_selfloops(self.graph)) + "\n" +\
                        "Graph Type: " + ("Directed And " if self.graph.is_directed() == True else "Undirected And ") +\
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
    def InfoMap(self, G,report):
        infomapWrapper = infomap.Infomap("--two-level --directed") # TODO Tweak this to improve solution after getting results

        for e in G.edges():
            infomapWrapper.network.addLink(*e)

        infomapWrapper.run()

        report.write("%d modules with codelength %f found\n" % (infomapWrapper.numTopModules(), infomapWrapper.codelength))

        # Set the community of each node in the graph
        communities = {}
        for node in infomapWrapper.iterLeafNodes():
            communities[node.physicalId] = node.moduleIndex()

        networkx.set_node_attributes(G, name='community', values=communities)

        return infomapWrapper.numTopModules()   
    
    def LabelPropagation(self, G, weight, seed):
        return asyn_lpa_communities(G, weight, seed)
    
    def ratePartition(self, G, report):
        report.write(self.sumDiff(G))
    
    def sumDiff(self, G):
        communities = collections.defaultdict(lambda: list())
        for k, v in networkx.get_node_attributes(G, 'community').items():
            communities[v].append(k)

        intra_density = collections.defaultdict(lambda: list())
        inter_density = collections.defaultdict(lambda: list())
        sum_diffs = 0
        vertex_count = len(G.nodes())
        for community in communities.keys():
            nc = len(communities[community])
            intra_density[community] = measure.intra_community_edges(G, communities[community])
            inter_density[community] = measure.inter_community_edges(G, [{node} for node in communities[community]])
            sum_diffs = intra_density[community] / (nc * (nc - 1)) * 2 -\
                        inter_density[community] / (nc * (vertex_count - nc))

        return ("Sum of intra-inter community edge density differences " + sum_diffs)


def main():
    report = open("report.txt", 'a')
    analyser = Analyser()
    
    edgeListModels = ["data//club.txt"]
    #TODO Use createGraphLFR to create graphs

    for edgeListModel in edgeListModels:         
        graph = Graph(edgeListModel) 
        graph.createGraphFromEdgeList(edgeListModel)
        graph.classify(report)

        # InfoMap - More stats in the stdout
        start = timer()
        infoMapPartition = analyser.InfoMap(graph.getGraph(), report)
        end = timer()

        report.write("InfoMap Stats:\n" + "Processing time: "+ str(end - start) + "s" + "\n")
        analyser.ratePartition(graph, infoMapPartition)

        # Label Propagation
        start = timer()
        labelPropagationPartition = analyser.LabelPropagation(graph.getGraph(), None, None)
        end = timer()

        report.write("Label Propagation Stats:\n" + "Processing time "+ str(end - start) + "s" + "\n")
        analyser.ratePartition(report, labelPropagationPartition)

        report.write("\n")
     

if __name__ == '__main__':
    main()