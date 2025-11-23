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

# Depth-First-Search of the graph
def DFS(graph, node, visited, stack):
    visited.append(node)
    for neighbor in graph.graph[node]:
        if neighbor not in visited:
            DFS(graph, neighbor, visited, stack)
    stack.append(node)

# This fills the stack using a DFS from each of the graph nodes
def fill_order(graph, visited, stack):
    for node in graph.nodes:
        if node not in visited:
            DFS(graph, node, visited, stack)

# Tranpose the graph to reverse graph edges
def transpose(graph):
    new_graph = dir_graph()
    for node in graph.graph:
        for neighbor in graph.graph[node]:
            new_graph.add_edge(neighbor, node)
    return new_graph

# Runs Kosaraju's algorithm to get strongly connected components
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
            DFS(transposed_graph, curr_node, visited, component)
            sccs.append(component)

    return sccs


"""
This is the heuristic the paper describes for doing a topological sort that maximizes
agreement with the probabilities of the drugs listed.
"""
def heuristic_alg(last_graph, sccs, prob):
    scc_node_idx = {}
    for idx, component in enumerate(sccs):
        for node in component:
            scc_node_idx[node] = idx

    scc_graph = dir_graph()
    for node, neighbor_list in last_graph.graph.items():
        for neighbor in neighbor_list:
            scc_graph.add_edge(scc_node_idx[node], scc_node_idx[neighbor])

    def node_score(v):
        if v in prob:
            return prob[v]
        else:
            return 1 - prob[v[1:]]

    sort_sccs = []
    while scc_graph.nodes:
        best_node, best_score = None, None
        for node in scc_graph.nodes:
            in_deg = False
            for node_2 in scc_graph.nodes:
                if node in scc_graph.graph[node_2]:
                    in_deg = True
                    break

            if not in_deg:
                score = min(node_score(comp_node) for comp_node in sccs[node])
                if best_score is None or score < best_score:
                    best_node = node
                    best_score = score

        assert best_node is not None
        sort_sccs.append(sccs[best_node])

        scc_graph.nodes.remove(best_node)
        for node in scc_graph.nodes:
            if best_node in scc_graph.graph[node]:
                scc_graph.graph[node].remove(best_node)

    return sort_sccs

"""
Returns True if there is a contradiction in the literals within
one of the strongly connected components.
"""
def contradiction(sccs):
    for comp in sccs:
        for u in comp:
            for v in comp[comp.index(u):]:
                if v == resolve_neg(neg + u):
                    return True, sccs

    return False, sccs


"""
Overall 2-SAT Solver
First the graph is constructed from each of the formulas provided.
For ex, given a ^ b, we add an edge ~a -> b and an edge ~b -> a.
Given just a, we add ~a -> a
"""
def two_sat_solver(formula):
    sat_graph = dir_graph()
    for conjs in formula.conj:
        if len(conjs) == 2:
            u = conjs[0]
            v = conjs[1]
            sat_graph.add_edge(resolve_neg(neg + u), v)
            sat_graph.add_edge(resolve_neg(neg + v), u)
        else:
            u = conjs[0]
            sat_graph.add_edge(resolve_neg(neg + u), u)
    sccs = kosaraju_scc(sat_graph)

    sccs = heuristic_alg(sat_graph, sccs, formula.prob)

    res = contradiction(sccs)

    # 2SAT Satisfiable
    if not res[0]:
        sat_dict = {}
        sccs = res[1]
        for comp in sccs:
            for node in comp:
                if resolve_neg(neg + node) not in sat_dict.keys() and node not in sat_dict.keys():
                    if '~' not in node:
                        sat_dict[node] = 0
                    else:
                        sat_dict[resolve_neg(neg + node)] = 1
        return sat_dict
    else:
        # Not 2SAT Satisfiable
        return None

# ======= 2-SAT example =======
# Example from the paper repo shows that the works correctly
if __name__ == '__main__':

    formula = two_cnf({'a': 0.6, 'b': 0.9})
    formula.add_clause(['~b', '~a'])
    out_dict = two_sat_solver(formula)
    print(out_dict)  # pos: {b} (higher prob), neg: {a} (lower prob)

    formula = two_cnf({'a': 0.9, 'b': 0.6, 'c': 0.8, 'd': 0.6})
    formula.add_clause(['~a', '~b'])
    formula.add_clause(['~b', '~d'])
    formula.add_clause(['~c', '~d'])
    out_dict = two_sat_solver(formula)
    print(out_dict)  # pos: {a,c} (higher probs), neg: {b, d} (lower probs)
