import networkx as nx
from heapq import heappush,heappop, heapify

__author__ = 'navin'



debug=False
def p(x):
    if debug:
        print(x)

class UndirectedGraph(nx.Graph):

    def equalGraphs(self, G2):
        nodeSet1=set(self.nodes())
        nodeSet2=set(G2.nodes())
        if nodeSet1 != nodeSet2:
            return False
        edgeSet1=set(self.edges())
        edgeSet2 = set(G2.edges())
        if edgeSet1 != edgeSet2:
            return False
        return True

    def maximumSpanningTree(self):
        assert isinstance(self, nx.Graph)
        edges = []
        valid_edges = []
        node = self.nodes()[0]
        cloudSet = set()
        print(self[node].items())
        for dest, edge in self[node].items():
            edges.append((-edge["wt"], node, dest))
        while len(edges) != 0:
            ndwTriplet = heappop(edges)
            dest = ndwTriplet[2]
            if dest in cloudSet:
                continue
            cloudSet.add(dest)
            valid_edges.append((ndwTriplet[1], dest))
            for nbr, edge in self[dest].items():
                edges.append((dest, nbr, edge["wt"]))
        self.remove_edges_from(self.edges())
        for edge in valid_edges:
            self.add_edge(edge[0], edge[1])

    def maxCardinalitySearch(self, actionNum, triangulateGraph=False):
        #actionNum 0 just checks for triangularity
        #actionNum 1 returns clique size
        #actionNum 2 just returns the junction tree

        #triangulateGraph will just not delete the new edges
        from pgmpy.MarkovModel.JunctionTree import JunctionTree

        jt=JunctionTree()
        jtnode = 0
        L = set()
        L_nbrs = {}
        for node in self.nodes():
            L_nbrs[node]=[]

        U=[(-0,node) for node in self.nodes()]
        heapify(U)
        new_edges = []
        max_clique_size = 0
        firstIt=True
        prevNode = (None, None)
        while len(U)!=0:
            top = heappop(U)
            curr_node = top[1]
            if curr_node in L:
                continue
            L.add(curr_node)
            p("Working with "+str(curr_node))
            for nbr in self.neighbors(curr_node):
                if nbr not in L:
                    L_nbrs[nbr].append(curr_node)
                    heappush(U,(-len(L_nbrs[nbr]),nbr))
            cliqueCheckReqd = L_nbrs[curr_node]
            p(str(cliqueCheckReqd))
            new_edges_temp = self.makeClique(cliqueCheckReqd)
            new_edges.extend(new_edges_temp)

            if actionNum==0 and new_edges_temp:
                print("Not a triangulated graph")
                return False

            C_curr = len(L_nbrs[curr_node])

            if firstIt or C_curr >= prevNode[1] +1:
                p(prevNode[1])
                #Dont form a clique now
            else:
                #add a clique node to the junction Tree
                jtnode+=1
                jt.add_node(jtnode)
                cliqueNodes = L_nbrs[prevNode[0]]+[prevNode[0]]
                jt.node[jtnode]["clique_nodes"]=cliqueNodes
                print("making a clique with "+str(cliqueNodes))
                max_clique_size=max(max_clique_size,len(cliqueNodes))
            prevNode = (curr_node, C_curr)
            p("prev "+str(prevNode))
            firstIt=False

        jtnode+=1
        jt.add_node(jtnode)
        cliqueNodes = L_nbrs[prevNode[0]]+[prevNode[0]]
        jt.node[jtnode]["clique_nodes"]=cliqueNodes
        print("making a clique with "+str(cliqueNodes))
        max_clique_size=max(max_clique_size,len(cliqueNodes))


        if actionNum==0:
            return True
        if not triangulateGraph:
            for edge in new_edges:
                self.remove_edge(edge[0],edge[1])
        elif actionNum==1:
            #delete all edges and return the junction Tree
            return max_clique_size
        elif actionNum==2:
            jt.addJTEdges()
            return jt



    def isTriangulated(self):
        ret = self.maxCardinalitySearch(0)
        return ret





    def print_graph(self, s):
        print("Printing the graph " + s + "<<<")
        for node in self.nodes():
            print(str(node) + " : " + str(self[node]))
        print(">>>")

    def checkClique(self, nodes):
        for i in range(len(nodes)):
            for j in range(i+1,len(nodes)):
                if not self.has_edge(nodes[i],nodes[j]):
                    return False
        return True

    def makeClique(self, clique_nodes):
        new_edges=[]
        for i in range(len(clique_nodes)):
            a = clique_nodes[i]
            for j in range(i + 1, len(clique_nodes)):
                b = clique_nodes[j]
                if not self.has_edge(a, b):
                    p("Edge added b/w " + a + " and " + b)
                    self.add_edge(a, b)
                    new_edges.append((a,b))
        return new_edges


    def junctionTree1(self, returnJunctionTree=True, triangulateGraph=False, f=None):
        """


        Parameters
        ----------
        node  :  Graph Node
                Node for which the order needs to be changed

        states : List
                List of the states of node in the order in which
                CPD will be entered

        See Also
        --------
        get_rule_for_states
        get_rule_for_parents
        set_rule_for_parents

        Example
        -------
        >>> from pgmpy import BayesianModel as bm
        >>> G = bm.BayesianModel([('diff', 'grade'), ('diff', 'intel')])
        >>> G.set_states({'diff': ['easy', 'hard'],
        ...               'intel': ['dumb', 'smart']})
        >>> G.get_rule_for_states('diff')
        ['easy', 'hard']
        >>> G.set_rule_for_states('diff', ['hard', 'easy'])
        >>> G.get_rule_for_states('diff')
        ['hard', 'easy']
        """
        from pgmpy.MarkovModel.JunctionTree import JunctionTree
        jt = JunctionTree()
        jtnode = 0
        nodes = [(f(self,node),node) for node in self.nodes()]
        heapify(nodes)
        p(str(nodes))
        max_clique_size= 0
        triangulated_nodes = set()
        new_edges = []
        while len(nodes)!=0:
            min_el = heappop(nodes)
            curr_node = min_el[1]
            p("Popped "+str(curr_node))
            if curr_node in triangulated_nodes:
                p("continued")
                continue

            triangulated_nodes.add(curr_node)
            flag = False
            set2 = set(self.neighbors(curr_node))
            for nbr in self.neighbors(curr_node):
                if nbr in triangulated_nodes:
                    set1 = set(self.neighbors(nbr))
                    if set2 - set1 is None:
                        flag = True
                        break
            if flag:
                break
            #actual triangulation begins here
            #Take all the neighbours and connect them. Done
            jtnode += 1
            jt.add_node(jtnode)
            clique_nodes = []
            for node in self.neighbors(curr_node):
                if node not in triangulated_nodes:
                    clique_nodes.append(node)
            jt.node[jtnode]["clique_nodes"] = clique_nodes
            p(str(clique_nodes))
            max_clique_size=max(max_clique_size,len(clique_nodes))
            p("Working with "+str(curr_node))
            new_edges_temp = self.makeClique(clique_nodes)
            new_edges.extend(new_edges_temp)
            for node in clique_nodes:
                heappush(nodes,(f(self, node), node))
        if not triangulateGraph:
            for edge in new_edges:
                self.remove_edge(edge[0],edge[1])
        if not returnJunctionTree:
            return max_clique_size
        jt.addJTEdges()
        return jt

    def junctionTreeFromTriangulatedGraph(self, returnJunctionTree):
        return self.maxCardinalitySearch(2,False)


    def junctionTreeOptimal(self, returnJunctionTree, triangulateGraph):
        pass


    def minFillHeu(self,graph,node):
        nbrs = graph.neighbors(node)
        num_edges=0
        for i in range(len(nbrs)):
            for j in range(i+1,len(nbrs)):
                if(graph.has_edge(nbrs[i],nbrs[j])):
                    num_edges+=1
        val = ((len(nbrs)*(len(nbrs)-1))/2) - num_edges
        return val


    def bestTriangulationHeuristic(self):
        i=2
        minCliqueSize = int("inf")
        minCliqueTechnique = -1
        while True:
            size = self.junctionTreeCliqueSize(i)
            if size==False:
                break
            if size < minCliqueSize:
                minCliqueTechnique=i
                minCliqueSize=size
        return minCliqueTechnique




    def junctionTreeTechniques(self, triangulation_technique, returnJunctionTree, triangulateGraph):
        if triangulation_technique==0:
            if(self.isTriangulated()):
                ret = self.junctionTreeFromTriangulatedGraph(returnJunctionTree)
            else:
                raise Exception(" Graph Is Not Triangulated ")
        if triangulation_technique==1:
            ret=self.junctionTreeOptimal(returnJunctionTree, triangulateGraph)
        if triangulation_technique==2:
            f = (lambda graph,node:len(graph.neighbors(node)))
            ret=self.junctionTree1(returnJunctionTree,triangulateGraph,f)
        elif triangulation_technique==3:
            f = (lambda graph,node:-len(graph.neighbors(node)))
            ret = self.junctionTree1(returnJunctionTree, triangulateGraph,f)
        elif triangulation_technique==4:
            f = (lambda  graph, node: self.minFillHeu(graph,node) )
            ret = self.junctionTree1(returnJunctionTree, triangulateGraph,f)
        elif triangulation_technique==5:
            f=(lambda graph,node: 5)
            ret = self.junctionTree1(returnJunctionTree, triangulateGraph,f)
        else:
            ret = False
        return ret

    def junctionTreeCliqueSize(self, triangulation_technique):
        return self.junctionTreeTechniques(triangulation_technique,False, False)

    def makeJunctionTree(self, triangulation_technique):
        return self.junctionTreeTechniques(triangulation_technique, True, False)



    def readSimpleFormatGraph(self, filename):
        """
        Read the graph from a file assuming a very simple graph reading format

        Parameters
        ----------
        filename  :  String
                The file which has the graph data

        Example
        -------
        >>> from pgmpy import UndirectedGraph as ug
        >>> G = ug.UndirectedGraph()
        >>> G.readSimpleFormatGraph("graph)
        """
        file = open(filename,"r")
        num_nodes = int(file.readline())
        for i in range(num_nodes):
            self.add_node(str(i))
        print("nodes"+str(num_nodes))
        file.readline()
        edge = ""
        while True:
            edge = file.readline()
            if not edge:
                break
            nodes = edge.split()
            self.add_edge(nodes[0],nodes[1])



