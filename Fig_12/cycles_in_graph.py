import networkx as nx
import numpy as np
import itertools
from itertools import product
import matplotlib.pyplot as plt



## This files gives the functions used to generate Fig 12 in Appendix E.

def generate_grid_graph(n):
    """
    This function creates a grid graph of length n.
    The nodes are numbered as integers.
    """
    graph = nx.grid_graph([n, n])
    graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering='default', label_attribute=None)
    return graph

def generate_deleted_edges_graph(n, p):
    """
    This function generates a graph where each edge exists with a probability p.
    """
    graph = generate_grid_graph(n)
    number_of_edges = len(graph.edges())
    random_values_for_deletion = np.random.binomial(1, p, size=number_of_edges)
    edges = list(graph.edges())
    for i in range(number_of_edges):
        if random_values_for_deletion[i] == 0:
            graph.remove_edge(edges[i][0], edges[i][1])
    return graph

def remove_bottom_edges_of_cycles(graph, n, alice, bob):
    """
    This function takes a graph, grid size and Alice's and Bob's location.
    It outputs a graph wherein the bottom edges of four cycles have been removed if Alice and Bob do not belong to the node.
    """
    edges_to_remove = []
    edges = list(graph.edges())
    for i in range(n**2):
        possible_edges = [(i, i+1), (i+1, i+1+n), (i+n, i+n+1), (i, i+n)]
        check = list(set(possible_edges) - set(edges))
        if alice not in [i, i+n] and bob not in [i, i+n] and not check:
            graph.remove_edge(i, i+n)
    return graph


def remove_edges_from_alice_or_bob(graph, node):
    """
    This function takes a graph, and a node. 
    It outputs a graph wherein the edges belonging to the node have been removed.
    """
    edges = list(graph.edges(node))
    if edges:
        for i in range(len(edges)):
            graph.remove_edge(edges[i][0], edges[i][1])
    return graph


def generate_cycles_to_consider(n, p, alice, bob):
    
    """
    This function takes in grid size, link prob and alice and bob. 
    It outputs a graph with four cycles removed and the cycles in the graph.
    
    """
    graph = generate_deleted_edges_graph(n, p)
    graph = remove_bottom_edges_of_cycles(graph, n, alice, bob)
    cycles = nx.cycle_basis(graph)
    return graph, cycles


def get_right_hand_corner(n, cycle):
    """
    This function takes in the grid size n and the cycles. 
    It outputs the righht hand corner of the cycle. 
    
    """
    
    highest_number = max(cycle)
    base = my_round(highest_number, n)
    list2 = [max(j-base, -1) for j in cycle]
    list3 = [i for i in list2 if i >= 0]
    right_hand_corner_node = cycle[list2.index(min(list3))]
    return (right_hand_corner_node, right_hand_corner_node-n)


def my_round(x, base):

    return x // base * base


def dict_cycles(n, p, alice, bob):
    """This function returns a diictionary with keys as the cycle length and the values as the list of nodes involved in the cycle, and the generated graph. 
    """
    graph,cycles = generate_cycles_to_consider(n, p, alice, bob)
    dict1 = {i: [] for i in range(4, n**2+1, 2)}

    # Remove 4-cycles by deleting the right-hand corner edge
    for cycle_four in cycles:
        if len(cycle_four) == 4:
            remove_edge = get_right_hand_corner(n, cycle_four)
            if alice not in remove_edge and bob not in remove_edge:
                graph.remove_edge(*remove_edge)

    # Update the dictionary with remaining cycles
    cycles = nx.cycle_basis(graph)
    for i in cycles:
        dict1[len(i)].append(i)


    # Remove even-length cycles until there are no more
    for i in range(6, n**2+1, 2):
        cycles_in = dict1.get(i, []).copy()
        cycle_inter = []

        while cycles_in:
            cycle1 = cycles_in.pop(0)
            cycle_inter.append(cycle1)
            remove_edge = get_right_hand_corner(n, cycle1)
            if alice not in remove_edge and bob not in remove_edge:
                graph.remove_edge(*remove_edge)
            cycles = nx.cycle_basis(graph)
            cycles_in = [cycles_aga for cycles_aga in cycles if len(cycles_aga) <= i]

            for cycle_check in cycle_inter:
                if cycle_check in cycles_in:
                    cycles_in.remove(cycle_check)

        
            for i2 in range(i+2,n**2+1,2):
                dict1[i2] = [cycles1 for cycles1 in cycles_in if len(cycles1) == i2]

    for j in range(4,n**2+1,2):
        [dict1[j].remove(pos_cycle) for pos_cycle in dict1[j] if alice in pos_cycle or bob in pos_cycle]
        [dict1[j].remove(pos_cycle) for pos_cycle in dict1[j] if bob in pos_cycle or alice in pos_cycle]


    return dict1, graph


def final_function_frac_cycles(n,n1): 
    """This function gives a dictionary of dictionary. The keys in the first level of the dictionary are the p_link 
    in the graph. The key values for the second level are the cycle length. The values for this level are the fraction of 
    cycles. 
    
    n: grid size 
    n1: Number of tries. 
    
    """
    
    
    dict_final = {}

    key_List = [i for i in range(6, n**2)]
    key_List.remove(3+n)

    keyList = [i for i in range(4, n**2+1, 2)]

    for p_link1 in range(55, 101, 1):
        p_link = p_link1/100
        dict_val = {i: 0 for i in keyList}

        for j1 in key_List:
            for j in range(n1):
                dict_obtained, _ = dict_cycles(n, p_link, 3, j1)

                for k in keyList:
                    if dict_obtained[k]:
                        dict_val[k] += 1

        for i in keyList:
            dict_val[i] /= n1 * len(key_List)
            
    dict_final[p_link] = dict_val

    return dict_final