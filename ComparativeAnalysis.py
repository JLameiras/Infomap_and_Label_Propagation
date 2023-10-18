from typing import Self
import networkx
import collections
import infomap

from networkx.algorithms.community import asyn_lpa_communities
import matplotlib.pyplot as pyplot
import matplotlib.colors as colors

import infomap
import networkx.algorithms as algorithms
import networkx.algorithms.community.quality as measure
import collections


class Graph:
    def __init__(self):
        self.graph = networkx.Graph()

    def setName(self, name):
        self.name = name

    def createGraphFromEdgeList(self, filename):
        
        networkx.from_edgelist(open(filename, 'r'), self.graph)

        return self.graph
    
    def createGraphLFR(self, n, tau1, tau2, mu, average_degree, min_degree, max_degree,
                       min_community, max_community, tol, max_iters, seed):
            
        self.graph = networkx.LFR_benchmark_graph(n, tau1, tau2, mu, average_degree, min_degree, max_degree,min_community, max_community, tol, max_iters, seed)

        return self.graph
    
    def classify(self, report):
        analysis = "Analyses of the network " + self.name + "\n" +\
                        "Nodes: {}, Edges: {}, Self Loops: {}".format(self.graph.number_of_nodes(), self.graph.number_of_edges(), networkx.number_of_selfloops(self.graph)) + "\n" +\
                        "Graph Type: " + "Directed And " if self.graph.is_directed() == True else "Undirected And " +\
                        "Weighted" if networkx.is_weighted(self.graph) == True else "Non-Weighted" + "\n"                            

        # The rest might be too costly. Not all parameters matter
               
        # results.write("Number of connected components: {}".format(a.getNumberOfConnectedComponents(graph)))
        # results.write("\n")
        # results.write("Number of weakly connected components: {}".format(a.getNumberOfWeaklyConnectedComponents(graph)) if graph.is_directed() else "Weakly connected components not implemented for undirected case")
        # results.write("\n")
        # results.write("Number of Isolates: {}".format(a.getNumberOfIsolates(graph)))
        # results.write("\n")
        # results.write("Degree Centrality: {}".format(a.getDegreeCentrality(graph)))
        # results.write("\n")
        # results.write("Betweeness Centrality: {}".format(a.getBetweenessCentrality(graph)))
        # print(a.getNeighbours(graph,1))
        # for component in a.getConnectedComponents(graph):
        #     subgraph = Graph()
        #     for neighbours in component:
        #     print("Diameter of {} is: {}\n".format(component,"pass"))
        # results.write("\n")
        # results.write("Closeness centrality: {}".format(a.getClosenessCentrality(graph)))
        # results.write("\n")
        # results.write("Katz centrality: {}".format(a.getKatzCentrality(graph)))
        # results.write("\n")
        # results.write("Pagerank: {}".format(a.getPageRank(graph)))
        # results.write("\n")
        # results.write("Triangles: {}".format(a.getTriangles(graph)))
        # results.write("\n")
        # results.write("All Pairs Shortest Path: {}".format(a.getAllPairsShortestPath(graph)))
        # results.write("\n")
        # results.write("All Pairs Shortest Connectivity: {}".format(a.getAllPairsNodeConnectivity(graph)))
        # results.write("\n")
        # results.write("Network bridges: {}".format(a.getBridges(graph)))
        # results.write("\n")
        # results.write("All Connected Components: {}".format(a.getConnectedComponents(graph)))

        report.write(analysis)


class InfoMap:
    def __init__(self, G):
        self.graph = G

    def findCommunities(self, G):
        infomapWrapper = infomap.Infomap("--two-level --directed") # TODO Tweak this to improve solution

        for e in G.edges():
            infomapWrapper.network.addLink(*e)

        infomapWrapper.run()

        print("Found %d modules with codelength: %f" % (infomapWrapper.numTopModules(), infomapWrapper.codelength))

        # Set the community of each node in the graph
        communities = {}
        for node in infomapWrapper.iterLeafNodes():
            communities[node.physicalId] = node.moduleIndex()

        networkx.set_node_attributes(G, name='community', values=communities)

        return infomapWrapper.numTopModules()

    
class LabelPropagation:
    def __init__(self, G):
        self.graph = G

    def findCommunities(self, G, weight, seed):
        return asyn_lpa_communities(G, weight, seed)
    

class Analyser:
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

        print("Sum of intra-inter community edge density differences ", sum_diffs)

    # TODO: fix this mess
    def outputCommunities(self, G):
        self.findCommunities(G)
        communities = collections.defaultdict(lambda: list())
        for k, v in networkx.get_node_attributes(G, 'community').items():
            communities[v].append(k)
        communitie_sort = sorted(communities.values(), key=lambda b: -len(b))
        count = 0
        for communitie in communitie_sort:
            count += 1
            print(f'count{count},community{communitie}', end='\n')
        print(self.cal_Q(communities.values()))


def main():
    graph = Graph()
    report = open("report.txt", 'a')

    dataModels = ["data//club.txt"]
    #TODO Use createGraphLFR to create graphs

    for dataModel in dataModels: 
        graph.setName(dataModel) 
        graph.createGraphFromEdgeList(dataModel)
        graph.classify(report)

        # InfoMap
        infomap = InfoMap(graph)
        # Measure performance here
        infomap.findCommunities(graph)

        # Label Propagation
        labelpropagation = LabelPropagation(graph)
        # Measure performance here
        infomap.findCommunities(graph)
     

def __main__():
    main()