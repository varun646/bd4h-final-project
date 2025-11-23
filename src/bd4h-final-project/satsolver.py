from collections import defaultdict, deque

neg = '~'

#  directed graph class
#  adapted from:
#  src: https://www.geeksforgeeks.org/generate-graph-using-dictionary-python/
class dir_graph:
    def __init__(self):
        # create an empty directed graph, represented by a dictionary
        #  The dictionary consists of keys and corresponding lists
        #  Key = node u , List = nodes, v, such that (u,v) is an edge
        self.graph = defaultdict(set)
        self.nodes = set()

    # Function that adds an edge (u,v) to the graph
    #  It finds the dictionary entry for node u and appends node v to its list
    # performance: O(1)
    def add_edge(self, u, v):
        self.graph[u].add(v)
        self.nodes.add(u)
        self.nodes.add(v)

    # Function that outputs the edges of all nodes in the graph
    #  prints all (u,v) in the set of edges of the graoh
    # performance: O(m+n) m = #edges , n = #nodes
    def print(self):
        edges = []
        # for each node in graph
        for node in self.graph:
            # for each neighbour node of a single node
            for neighbour in self.graph[node]:
                # if edge exists then append
                edges.append((node, neighbour))
        return edges

"""
    2-CNF class
    Class for the boolean formula in Conjuctive Normal form
    This means that the size of literals is at most 2
    We represent this as a list of lists. The Inner lists are disjunctions of literals
      and the outer lists are conjunctions
    Negation is represented with ~
"""
class two_cnf:
    # Prob is a dictionary mapping variables (medications) to probabilities
    def __init__(self, prob):
        self.conj = []
        self.prob = prob

    def add_clause(self, clause):
        if len(clause) > 2:
            print("error: clause has more than 2 literals")
            return

        self.conj.append(clause)

    def get_variables(self):
        all_vars = set()
        for clause in self.conj:
            for lit in clause:
                all_vars.add(lit)

        return all_vars

    def print(self):
        print(self.conj)

## This is a helper func to resolve all instances of multiple negatives
#      i.e. ~~
def resolve_neg(formula):
    return formula.replace(neg + neg, '')

"""
The following methods comprise getting the strongly connected components (SCCs)
of a graph using Kosaraju's algorithm. This is returned as a list of lists.
Each list is a SCC containing nodes of the original graph G.
Adapted from: https://www.geeksforgeeks.org/dsa/kosarajus-algorithm-in-python/
"""

# DFS algorithm
def DFS(graph, node, visited, stack):
    visited.append(node)
    for neighbor in graph.graph[node]:
        if neighbor not in visited:
            DFS(graph, neighbor, visited, stack)
    stack.append(node)

def fill_order(graph, visited, stack):
    for node in graph.nodes:
        if node not in visited:
            DFS(graph, node, visited, stack)

def transpose(graph):
    new_graph = dir_graph()
    for node in graph.graph:
        for neighbor in graph.graph[node]:
            new_graph.add_edge(node, neighbor)
    return new_graph

def kosaraju_scc(graph):
    stack = deque()
    fill_order(graph, [], stack)

    transposed_graph = transpose(graph)
    visited = []
    sccs = []

    while stack:
        curr_node = stack.pop()
        if curr_node not in visited:
            component = []
            DFS(graph, curr_node, visited, component)
            sccs.append(component)

    return sccs
