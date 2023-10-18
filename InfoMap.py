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

# The findCommunities and drawNetwork of the InfoMap class are adaptations of the following example:
# github.com/chrisbloecker/infomap-bipartite/blob/master/examples/python/infomap-examples.ipynb
# Please note that the original code was not up to date with the InfoMap's python module.


class Graph:
    def __init__(self):
        self.graph = networkx.DiGraph()

    def createGraphFromEdgeList(self, filename):
        
        networkx.from_edgelist(open(filename, 'r'), self.graph)

        return self.graph
    
    def createGraphLFR(self, n, tau1, tau2, mu, average_degree, min_degree, max_degree,
                       min_community, max_community, tol, max_iters, seed):
            
        self.graph = networkx.LFR_benchmark_graph(n, tau1, tau2, mu, average_degree, min_degree, max_degree,min_community, max_community, tol, max_iters, seed)

        return self.graph


class InfoMap:
    def __init__(self, G):
        self.graph = G

    def findCommunities(self, G):
        infomapWrapper = infomap.Infomap("--two-level --directed") # Test

        print("Building Infomap network from a NetworkX graph...")
        for e in G.edges():
            infomapWrapper.network.addLink(*e)

        print("Find communities with Infomap...")
        infomapWrapper.run()

        print("Found %d modules with codelength: %f" % (infomapWrapper.numTopModules(), infomapWrapper.codelength))

        # Set the community of each node in the graph
        communities = {}
        for node in infomapWrapper.iterLeafNodes():
            communities[node.physicalId] = node.moduleIndex()

        networkx.set_node_attributes(G, name='community', values=communities)

        return infomapWrapper.numTopModules()

    # Only useful for small graphs
    def drawNetwork(self, G):
        # position map
        pos = networkx.spring_layout(G)
        # community ids
        communities = [v for k, v in networkx.get_node_attributes(G, 'community').items()]
        numCommunities = max(communities) + 1
        # color map from http://colorbrewer2.org/
        cmapLight = colors.ListedColormap(['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6'],
                                     'indexed', numCommunities)
        cmapDark = colors.ListedColormap(['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a'],
                                    'indexed', numCommunities)

        # Draw edges
        networkx.draw_networkx_edges(G, pos)

        # Draw nodes
        nodeCollection = networkx.draw_networkx_nodes(G,
                                                           pos=pos,
                                                           node_color=communities,
                                                           cmap=cmapLight
                                                           )
        # Set node border color to the darker shade
        darkColors = [cmapDark(v) for v in communities]
        nodeCollection.set_edgecolor(darkColors)

        # Draw node labels
        for n in G.nodes():
            pyplot.annotate(n,
                            xy=pos[n],
                            textcoords='offset points',
                            horizontalalignment='center',
                            verticalalignment='center',
                            xytext=[0, 0],
                            color=cmapDark(communities[n - 1])
                            )

        pyplot.axis('off')
        pyplot.savefig("graph_draw.png")
        pyplot.show()

    # Everything cleaned up until here!

    def output(self, G):
        self.findCommunities(G)

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


    def visualize(self, G):
        self.findCommunities(G)
        self.drawNetwork(G)

    def getNumberOfConnectedComponents(self, G):
        return algorithms.number_connected_components(G)

    def getNumberOfCliques(self, G):
        return algorithms.number_of_cliques(G)

    def getNumberOfStronglyConnectedComponents(self, G):
        return algorithms.number_strongly_connected_components(G)

    def getNumberOfWeaklyConnectedComponents(self, G):
        return algorithms.number_weakly_connected_components(G)

    def getNumberOfIsolates(self, G):
        return algorithms.number_of_isolates(G)

    def getDegreeCentrality(self, G):
        return algorithms.degree_centrality(G)

    def getBetweenessCentrality(self, G):
        return algorithms.betweenness_centrality(G)

    def getAllPairsShortestPath(self, G):
        return algorithms.all_pairs_shortest_path(G)

    def getAllPairsNodeConnectivity(self, G):
        return algorithms.all_pairs_node_connectivity(G)

    def getClosenessCentrality(self, G):
        return algorithms.closeness_centrality(G)

    def getBridges(self, G):
        return algorithms.bridges(G)

    def getConnectedComponents(self, G):
        return algorithms.connected_components(G)

    def getDiameter(self, G):
        return algorithms.diameter(G)

    def getKatzCentrality(self, G):
        return algorithms.katz_centrality

    def getPageRank(self, G):
        return algorithms.pagerank(G)

    def getTriangles(self, G):
        return algorithms.triangles(G)

    def getNeighbours(self, G, vertex):
        neighbourList = []
        for neighbour in G:
            neighbourList.append(neighbour)
        return neighbourList
    
class LabelPropagation:
    def __init__(self, G):
        self.graph = G

    def findCommunities(self, G, weight, seed):
        return asyn_lpa_communities(G, weight, seed)


# results = open("results3.txt", 'a')
obj = Graph()
graph = networkx.karate_club_graph()
# graph = obj.createGraph("data//google.txt")
# graph = obj.createGraph("data//OpenFlights.txt")
# results.write("Network info:")
# results.write("\n")
# results.write("Nodes:{}, Edges:{}, Self loops:{}".format(graph.number_of_nodes(), graph.number_of_edges(), graph.number_of_selfloops()))
# results.write("\n")
# results.write("Graph type: " + "undirected" if graph.is_directed() == False else "directed")
# results.write("\n")
# results.write("Is multigraph? - {}".format(graph.is_multigraph()))
# results.write("\n")

a = InfoMap(graph)
# a.findCommunities(graph)
a.visualize(graph)
a.output(graph)
# a.printCom(graph)
#

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