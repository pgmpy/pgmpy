from heapq import heappush,heappop, heapify
import networkx as nx

debug=False
def p(x):
    if debug:
        print(x)

class UndirectedGraph(nx.Graph):

    def equalGraphs(self, G2):
        """
        Just to check if the current graph is same as G2 in terms of nodes and edges

        Parameters
        ----------
        G2:
            The other graph

        """
        nodeSet1=set(self.nodes())
        nodeSet2=set(G2.nodes())
        if nodeSet1 != nodeSet2:
            return False
        edgeSet1=set(self.edges())
        edgeSet2 = set(G2.edges())
        if edgeSet1 != edgeSet2:
            return False
        return True



    def maxCardinalitySearch(self, actionNum):
        """
        Applying the maximum cardinality search algo. Refer to Algo 9.3 (Koller)
        Used to check for triangularity and to find the max-cliques if the graph is
        already triangulated

        Parameters
        -----------
        actionNum:
                #actionNum 0 just checks for triangularity
                #actionNum 1 returns clique size
                #actionNum 2 just returns the junction tree
        """
        from pgmpy.MarkovModel.JunctionTree import JunctionTree

        jt=JunctionTree()
        jtnode = 0
        L = set()
        L_nbrs = {}
        for node in self.nodes():
            L_nbrs[node]=[]

        U=[(-0,node) for node in self.nodes()]
        heapify(U)
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

            if not self.checkClique(cliqueCheckReqd):
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
        p("making a clique with "+str(cliqueNodes))
        max_clique_size=max(max_clique_size,len(cliqueNodes))


        if actionNum==0:
            return True
        elif actionNum==1:
            return max_clique_size
        elif actionNum==2:
            jt.addJTEdges()
            return jt



    def isTriangulated(self):
        """
        Just checks if the graph is triangulated

        Parameters
        ----------

        See Also
        --------
        maxCardinalitySearch

        Example
        --------
        >>> from pgmpy import MarkovModel as mm
        >>> G = mm.UndirectedGraph()
        >>> G.readSimpleFormatGraph("graph")
        >>> G.junctionTreeTechniques(2,False,True)
        4
        >>> G.isTriangulated()
        True
        """
        ret = self.maxCardinalitySearch(0)
        return ret





    def print_graph(self, s):
        """
        Prints the graph in a particular fashion.
        Useful for debugging

        Parameters
        ----------
        nodes  :  List of nodes
                The list of nodes which are to be checked for clique property

        See Also
        --------
        make_clique

        Example
        -------
        >>> from pgmpy import MarkovModel as mm
        >>> G = mm.UndirectedGraph()
        >>> G.readSimpleFormatGraph("graph")
        >>> G.print_graph("Test Printing")
        Printing the graph Test Printing<<<
        10( {} ) : {'8': {}, '5': {}, '7': {}}
        1( {} ) : {'0': {}, '2': {}, '4': {}, '8': {}}
        0( {} ) : {'1': {}, '8': {}, '3': {}}
        3( {} ) : {'9': {}, '0': {}, '8': {}}
        2( {} ) : {'1': {}, '4': {}, '7': {}, '6': {}}
        5( {} ) : {'9': {}, '8': {}, '10': {}}
        4( {} ) : {'1': {}, '8': {}, '2': {}, '7': {}}
        7( {} ) : {'10': {}, '2': {}, '4': {}, '6': {}}
        6( {} ) : {'2': {}, '7': {}}
        9( {} ) : {'3': {}, '5': {}}
        8( {} ) : {'10': {}, '1': {}, '0': {}, '3': {}, '5': {}, '4': {}}
        >>>

        """

        print("Printing the graph " + s + "<<<")
        for node in self.nodes():
            print(str(node) + "( " + str(self.node[node]) +" ) : "+str(self[node]))
        print(">>>")

    def checkClique(self, nodes):
        """
        Check if a given set of nodes form a clique

        Parameters
        ----------
        nodes  :  List of nodes
                The list of nodes which are to be checked for clique property

        See Also
        --------
        make_clique

        Example
        -------
        >>> from pgmpy import MarkovModel as mm
        >>> G = mm.UndirectedGraph()
        >>> G.readSimpleFormatGraph("graph")
        >>> G.checkClique([1,2,3])
        False

        """
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


    def _junctionTree1(self, returnJunctionTree=True, triangulateGraph=False, f=None):
        """
        Applies the basic junction creation algorithms.
        Refer to Algorithm 9.4 in PGM (Koller)

        Parameters
        ----------
        returnJunctionTree  :  boolean
                returns the junction tree if yes, size of max clique if no

        triangulateGraph :
                Removes the edges added while triangulation if False
        f:
                The function that will be used to pick up the node at each stage

        See Also
        --------

        Example
        -------
        private function
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
        """

        """
        if returnJunctionTree:
            ind = 2
        else:
            ind=1
        return self.maxCardinalitySearch(ind)


    def junctionTreeOptimal(self, returnJunctionTree, triangulateGraph):
        """
        Exponential strategy to find the optimal junction tree creation strategy

        Parameters
        ----------
        returnJunctionTree  :  boolean
                returns the junction tree if yes, size of max clique if no

        triangulateGraph :
                Removes the edges added while triangulation if False

        Example
        -------

        """
        pass


    @staticmethod
    def minFillHeu(graph,node):
        """
        Minimum-fill heuristic
        """
        nbrs = graph.neighbors(node)
        num_edges=0
        for i in range(len(nbrs)):
            for j in range(i+1,len(nbrs)):
                if(graph.has_edge(nbrs[i],nbrs[j])):
                    num_edges+=1
        val = ((len(nbrs)*(len(nbrs)-1))/2) - num_edges
        return val


    def bestTriangulationHeuristic(self):
        """
        Tries every triangulation heuristic defined in junctionTreeTechniques and finds
        the best one

        Parameters
        ----------
        None

        See Also
        --------
        junctionTreeTechniques

        Example
        -------
        >>> from pgmpy import MarkovModel as mm
        >>> G = mm.UndirectedGraph()
        >>> G.readSimpleFormatGraph("graph")
        >>> G.bestTriangulationHeuristic()
        2
        """
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
        """
        Returns the junction tree or the max clique size depending on a triangulation technique
        Takes option about whether to retain the edges added during triangulation

        Parameters
        ----------
        triangulation_technique  :  integer
                The index of the triangulation technique to use
                0 : THe graph is already triangulated. Just make the junction tree
                1 : Optimal Triangulation. Exponential. Use if you have ample time
                2 : Min-neighbour
                3 : Min =fill
                4 : Max-neighbour

        returnJunctionTree : boolean
                Returns the junction tree if true, returns the triangulated graph otherwise

        triangulateGraph : boolean
                Retains the edges added during triangulation, if yes. Deletes the edges
                otherwise

        See Also
        --------
        junctionTree1
        junctionTreeOptimal
        junctionTreeFromTriangulatedGraph

        Example
        -------
        >>> from pgmpy import MarkovModel as mm
        >>> G = mm.UndirectedGraph()
        >>> G.readSimpleFormatGraph("graph")
        >>> G.junctionTreeCliqueSize(2)
        4
        """


        if triangulation_technique==0:
            ret = self.junctionTreeFromTriangulatedGraph(returnJunctionTree)
            if not ret:
                raise Exception(" Graph Is Not Triangulated ")
        if triangulation_technique==1:
            ret=self.junctionTreeOptimal(returnJunctionTree, triangulateGraph)
        if triangulation_technique==2:
            f = (lambda graph,node:len(graph.neighbors(node)))
            ret=self._junctionTree1(returnJunctionTree,triangulateGraph,f)
        elif triangulation_technique==3:
            f = (lambda  graph, node: UndirectedGraph.minFillHeu(graph,node) )
            ret = self._junctionTree1(returnJunctionTree, triangulateGraph,f)
        elif triangulation_technique==4:
            f = (lambda graph,node:-len(graph.neighbors(node)))
            ret = self._junctionTree1(returnJunctionTree, triangulateGraph,f)
        elif triangulation_technique==5:
            f=(lambda graph,node: 5)
            ret = self._junctionTree1(returnJunctionTree, triangulateGraph,f)
        else:
            ret = False
        return ret

    def junctionTreeCliqueSize(self, triangulation_technique):
        """
        Returns the max-clique size that a triangulation technique will return
        It removes all the edges it added while triangulating it and hence doesn't affect the graph

        Parameters
        ----------
        triangulation_technique  :  integer
                The index of the triangulation technique to use
                See the documentation of junctionTreeTechniques function to see the
                index corresponding to each heuristic

        See Also
        --------
        junctionTreeTechniques

        Example
        -------
        >>> from pgmpy import MarkovModel as mm
        >>> G = mm.UndirectedGraph()
        >>> G.readSimpleFormatGraph("graph")
        >>> G.junctionTreeCliqueSize(2)
        4
        """

        return self.junctionTreeTechniques(triangulation_technique,False, False)

    def makeJunctionTree(self, triangulation_technique):
        """
        Return the junction Tree after triangulating it using the triangulation_technique.
        It removes all the edges it added while triangulating it and hence doesn't affect the graph

        Parameters
        ----------
        triangulation_technique  :  integer
                The index of the triangulation technique to use
                See the documentation of junctionTreeTechniques function to see the
                index corresponding to each heuristic

        See Also
        --------
        junctionTreeTechniques

        Example
        -------
        >>> from pgmpy import MarkovModel as mm
        >>> G = mm.UndirectedGraph()
        >>> G.readSimpleFormatGraph("graph")
        >>> jt=G.makeJunctionTree(2)
        >>> jt
        """

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
        >>> from pgmpy import MarkovModel as mm
        >>> G = mm.UndirectedGraph()
        >>> G.readSimpleFormatGraph("graph")
        """
        file = open(filename,"r")
        num_nodes = int(file.readline())
        for i in range(num_nodes):
            self.add_node(str(i))
        #print("nodes"+str(num_nodes))
        file.readline()
        edge = ""
        while True:
            edge = file.readline()
            if not edge:
                break
            nodes = edge.split()
            self.add_edge(nodes[0],nodes[1])




