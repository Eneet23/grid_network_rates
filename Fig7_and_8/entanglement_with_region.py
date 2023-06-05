import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
import random
import itertools
from itertools import product
import numpy as np
import math
import copy
import random as rnd
from tqdm import tqdm
import multiprocess as mp
from operator import add


# Graph initialization

def common_member(a, b):
    "this function takes in two lists and returns back the common elements in no particular order"
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return(list(a_set & b_set))
    else:
        return([])
    
def graph_initial(n):
    """This function creates a grid graph of length n. The nodes are numbered as integers."""
    G = nx.grid_graph([n,n])
    H = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)
    return H
def flatten(l):
     """This function creates a flattened list."""
     return [item for sublist in l for item in sublist]

def graph_reset(n):
    """This function gives us the original grid graph of size n**2. Each node has four subnodes to reflect the number of 
    systems being held. """
    G = nx.Graph()
    G.clear()
    G.clear()
    nodes = [i for i in range(1,4*n**2+1)]
    G.add_nodes_from(nodes)
    edges = [(2+4*i , 1+4*(i+n)) for i in range(0,(n)**2-n)]
    edgevert = [(4*i , 4*i+3) for i in range(1,(n)**2+1) if i%n!=0]
    
    G.add_edges_from(edges)
    G.add_edges_from(edgevert)
    
    return G


        

def deleted_edges_graph(n,p):
    """This function gives a graph where each edge exists with a probability p"""
    graph = graph_reset(n)
    number_of_edges = len(graph.edges())
    rand_for_del = []
    edges = list(graph.edges())
    for j in range(0,number_of_edges):
        rand_for_del.append(np.random.binomial(1,p))
    for i in range(0,number_of_edges):
        if rand_for_del[i] ==0:
            graph.remove_edge(edges[i][0],edges[i][1])
    return graph


def region_nodes(n,n1,n2):
    G = graph_initial(n)
    all_paths =[list1 for list1 in nx.all_shortest_paths(G, n1, n2)]
    shortest_path_nodes = set(flatten(all_paths))
    list_nodes_region = [shortest_path_nodes]
    while len(shortest_path_nodes)!=n**2:
        shortest_path_nodes = set(flatten([flatten(list(G.edges(i))) for i in shortest_path_nodes]))
        list_nodes_region.append(shortest_path_nodes)
    return list_nodes_region
        
    
    
def region_graph(n,n1,n2,G):
    """G is the input of deleted egde graph. The output creates different regions for a graph according to nodes 
    involved in the shortest path length , second shortest and so on. """
    list_nodes = region_nodes(n,n1,n2)
    graph_list = []
    for i in range(0,len(list_nodes)):
        graph = G.copy()
        remove_node_list = [[4*i1+1,4*i1+2,4*i1+3,4*i1+4] for i1 in range(0,n**2) if i1 not in list_nodes[i]]
        [graph.remove_node(j) for j in flatten(remove_node_list)]
        [graph.add_node(j) for j in flatten(remove_node_list)]
        graph_list.append(graph)
    return graph_list
         
    
    
    
def empty_graph(n):
    """this creates an empty graph with n^2 nodes   """
    nodes = [i for i in range(0,n**2)]
    G_new = nx.Graph()
    G_new.add_nodes_from(nodes)
    return G_new
    
def possible(n1,n2,graph):
    """inputs n1, n2 : the starting and end nodes
                graph: the underlying grid graph
        output: 1. the connected global nodes to n1.
                2. the edges connected to n1 in terms of local nodes
                3. the connected local nodes to n1
    """
    edges = []
    next_node = []
    nodes_involved=[]
    edges = flatten([list(graph.edges(i)) for i in [4*n1+1,4*n1+2,4*n1+3,4*n1+4] if len(list(graph.edges(i)))!=0])
    for k in flatten(edges):
        if k not in [4*n1+1,4*n1+2,4*n1+3,4*n1+4]:
            next_node.append((k-1)//4)
        if k in [4*n1+1,4*n1+2,4*n1+3,4*n1+4]:
            nodes_involved.append(k)
 
    return next_node,edges,nodes_involved
    
    
# functions for determining the intermediary states

def swap_list(n,n1,n2,graph):
    """inputs
    n: grid size
    n1,n2: start and end nodes
    p: prob of each link 
    With this process, we can have atmost 4 independent states existing between the nodes n1 and n2. We need to 
    obtain parameters for each of this process.
    
    output:
    new_final_list: The outer length corresponds to the number of states existing between the nodes n1 and n2. 
                    The inner list corresponds to the local nodes involved in the big state.
    j_final_list: The outer length corresponds to the number of states existing between the nodes n1 and n2. 
                    The inner list corresponds to the global nodes involved in the big state.
    graph_list: This is state of graph during each swap.
    graph: The initial graph. """
    
    """the idea behind the code is as follows: 
    1. start with the an edge node connected to n1. We keep track of the nodes attached to the big state 
        in each swap. For the first swap we start with the neighboring node. For the second swap, we choose a new nodes
        n' such that n' is not equal to n1 or n2. We keep on doing this till only nodes from n1 and n2 are left
        in the new state and no other connections are left. At this point we get an Index error and break out of the 
        loop. Next, we check whether both n1 and n2 are included in the final state. If not, the process is not addded
        to the output. If this process fails, we start with a new neighbouring node to n1. """
   
    
    
    
    next_node,edges,nodes_involved = possible(n1,n2,graph)
    new_final1 = [0]
    new_final_list =[]
    j_final_list =[]
    graph_list_out = []
    new_list2 = []
    node_local_n1 = [4*n1+1,4*n1+2,4*n1+3,4*n1+4]
    node_local_n2 = [4*n2+1,4*n2+2,4*n2+3,4*n2+4]
    num_list1 =[n1,n2]
    for l in range(0,len(next_node)):#start the loop over the parameters of independent state
        graph_list = []
        new_final = []
        if nodes_involved[l] not in new_final1:#start over a new loop of swapping if the final state doesn't contain the node_involved
            #initiate the values
            j = next_node[l]
            final_j = 0
            G = graph.copy()
            j_list = []
            while final_j !='end':
                try:#do this till we reach node n2
                    remove_edge_list1 = []
                    j_list.append(j)
                    list5 = []
                    list6 = []
                    remove_edge_list = [4*j+1,4*j+2,4*j+3,4*j+4]
                    remove_edge_list1.append(remove_edge_list)
                    list2=flatten([list(nx.node_connected_component(G, i)) for i in remove_edge_list])
                    [list6.append(k) for k in list2 if k not in list6]
                    list2=list6
                    [list2.remove(k) for k in remove_edge_list]
                    list4 = common_member(list2,list5)
                    [list2.remove(e) for e in list4]
                    [list5.remove(e) for e in remove_edge_list if e in list5]
                    list5 = list5 + list2
                    list7 = list5.copy()
                    new_final.append(list7)
                    edge1=[f for e,f in G.edges(remove_edge_list)]
                    edge2=[e for e in G.edges(remove_edge_list)]
                    edge_list = [(edge1[k],edge1[k+1]) for k in range(0,len(edge1)-1)]
                    edge_list.append((edge1[0],edge1[-1]))
                    G.add_edges_from(edge_list)
                    G.remove_edges_from(edge2)
                    G.remove_nodes_from(remove_edge_list)
                    G.add_nodes_from(remove_edge_list)
                    a = max(nx.connected_components(G), key=len)
                    new_list2.append(len(a))
                    node_removed = j
                    G.remove_nodes_from(a)
                    c1=list(a)
                    edge_list_conn = [(c1[k],c1[k+1]) for k in range(0,len(c1)-1)]
                    edge_list_conn.append((c1[0],c1[-1]))
                    G.add_edges_from(edge_list_conn)
                    internal1 = new_final[0][0]
                    initial_edge = list(graph.edges(internal1))
                    j = (new_final[-1][0]-1)//4
                    i=0
                    graph_list.append(G.copy())
                    while j in num_list1:
                        i = i+1
                        j = (new_final[-1][i]-1)//4

                except IndexError:
                    new_final1 = new_final[-1]
                    break

            weed1 = common_member(new_final[-1],node_local_n1)
            weed2 = common_member(new_final[-1],node_local_n2)
#             print(weed1)

            if weed1!=[] and weed2!=[]:
                  new_final_list.append(initial_edge +new_final.copy())
                  j_final_list.append(j_list.copy())
                  graph_list_out.append(graph_list)
        
        
    return new_final_list,j_final_list,graph_list_out,graph
            
    
    
    
    
    
def swap_list_fun(n1,n2,n,graph):
    """Inputs: 
    n1, n2 : starting and endind nodes.
    n: grid size
    p: prob of link"""
    
    """Outputs:
    list of local nodes of the big state during each swap process.
    local nodes involved in the big state
    local nodes involved in the final state
    the local nodes removed list during each swap."""
    
    """We start with the list of local nodes of the big state and the list of gloabl swap nodes.
    We seperate out the intermediate and the final local nodes. We create the list of local nodes involved in the 
    swap. We now obtain the local swap nodes from the big state. That is done by a1, and swap_list1. 
    """
    new_final_list,j_final_list,graph_list,graph = swap_list(n,n1,n2,graph)
    big_state_list = []
    final_state = []
    out_big_state = []
    out_swap_nodes = []
    out_final_state = []
    out_remove_edge = []
    for j in range(0,len(j_final_list)):
        big_state_list = []
        final_state = []
        big_state_list = new_final_list[j][:-1]
        final_state = new_final_list[j][-1]
        remove_node_edge_list = j_final_list[j].copy()
        remove_node_edge_local = [[4*k+1,4*k+2,4*k+3,4*k+4] for k in remove_node_edge_list]
        a1=[common_member(remove_node_edge_local[i], big_state_list[i]) for i in range(0,len(remove_node_edge_local))]
        swap_list1=[[big_state_list[j].index(i) for i in a1[j]] for j in range(0,len(a1))]
        out_big_state.append(big_state_list)
        out_swap_nodes.append(swap_list1)
        out_final_state.append(final_state)
        out_remove_edge.append(remove_node_edge_local)
        
    return out_swap_nodes,out_big_state, out_final_state,out_remove_edge, graph, graph_list



def local_geo(out_swap_nodes,out_big_state, out_final_state,out_remove_edge,graph):
    """Outputs:
    list of local nodes of the big state during each swap process.
    local nodes involved in the big state
    local nodes involved in the final state
    the local nodes removed list during each swap.
    original_graph. 
    We get a node path exists if there is no path connecting n1 and n2. """
    
    """This code gives us the number of two-length states involved in the each swap, the length of the big state in each swap, 
    and the length of the final state.
    If the length of big state during swap process is 2, then the number of two-length states is subtracted by one. """
    
    number_of_two_states_list = []
    size_of_big_state_list = []
    final_state_len_list = []
    for k in range(0,len(out_final_state)):
        remove_edge_list_int1 = out_remove_edge[k].copy()

        new_list1 = []
        new_list2 = []
        G = graph.copy()
        for j in range(0,len(out_swap_nodes[k])):
            list2=[nx.node_connected_component(G, i) for i in remove_edge_list_int1[j]]
            list3 = [len(i) for i in list2]
            new_list1.append(list3) #this creates the length of connected components at global node j in graoh G.
            edge1=[f for e,f in G.edges(remove_edge_list_int1[j])]
            edge2=[e for e in G.edges(remove_edge_list_int1[j])]
            edge_list = [(edge1[k],edge1[k+1]) for k in range(0,len(edge1)-1)]
            edge_list.append((edge1[0],edge1[-1]))
            G.add_edges_from(edge_list)
            G.remove_edges_from(edge2)
            G.remove_nodes_from(remove_edge_list_int1[j])
            G.add_nodes_from(remove_edge_list_int1[j])
            a = max(nx.connected_components(G), key=len)
            new_list2.append(len(a))
            G.remove_nodes_from(a)
            c1=list(a)
            edge_list_conn = [(c1[k],c1[k+1]) for k in range(0,len(c1)-1)]
            edge_list_conn.append((c1[0],c1[-1]))
            G.add_edges_from(edge_list_conn)
        number_of_two_states =[]
        for i in range(len(new_list1)):
            internal1 = new_list1[i].count(2)
            if max(new_list1[i])==2:
                number_of_two_states.append(internal1-1)
            else: 
                number_of_two_states.append(internal1)
        size_of_big_state = [2]+new_list2[:-1] #we always start with a bipartite pair therefore [2] is added
        number_of_two_states_list.append(number_of_two_states.copy())
        size_of_big_state_list.append(size_of_big_state.copy())
        final_state_len_list.append(new_list2[-1])
    if final_state_len_list == []:
        print('no path exists')
        
        
    return number_of_two_states_list,size_of_big_state_list,final_state_len_list


## Performing X measurement

def bina(digit, n):
    """It takes the integer n and expands it into a binary number of length digit. Example bina(4,2) gives 0010"""
    arr = []
    for i in range(digit):
        arr.append(n%2)
        n = n>>1
    bina_nums = []   
    for i in range(digit-1, -1, -1):
        bina_nums.append(str(arr[i]))
    bina_nums = ''.join(bina_nums)
   
    return bina_nums
       
    
def b4r_list(n):
    """This takes in a number n, and gives a list of binary strings of length 2^n. Example b4r_list(2)->[00,01,10,11] """
    b4 = [bina(n,i) for i in range(int(pow(2, n)))]
    return [i for i in b4]  

def mod_string(a):
    """This function takes in the binary string and outputs the mode. Example : 000-> 111, 001->110"""
    b = [str(int(i)^1) for i in a]
    c=''.join(b)
    return c
    
def str_binary(b):
    """takes in a binary string and outputs the corresponding integer."""
    return int(b,2)

def xvar_remove(remove_element, length_second_last_state):
    """Given the remove_element:"r" and the length of the state"n", we will have the final state of length = n-1. 
    The X list will be of length = n-2. This gives us the pattern for contributing elements. (See overleaf for details)
    
    """
    length_final_state= length_second_last_state-1 #len of final state
    b = b4r_list(length_final_state-1) #len of X list
    remove_element1 = length_final_state+1-remove_element
    if remove_element == 0:
        c1 = [[str_binary(b[i]+'0'),str_binary(mod_string(b[i])+'1')] for i in range(0,len(b))]
        #c1 = [[b[i]+'0',mod_string(b[i])+'1'] for i in range(0,len(b))]
    if remove_element!=0:
        c1 = [[int(b[i][0:remove_element1-1]+'0'+b[i][remove_element1-1:],2),int(b[i][0:remove_element1-1]+'1'+b[i][remove_element1-1:],2)] for i in range(0,len(b))]
        #c1 = [[b[i][0:remove_element1-1]+'0'+b[i][remove_element1-1:],b[i][0:remove_element1-1]+'1'+b[i][remove_element1-1:]] for i in range(0,len(b))]
    return c1
    

def possible_matches_remove(xvar_remove1,remove_element):
    if remove_element ==0:
        possible_matches_int = [[[0,xvar_remove1[i][0]],[0,xvar_remove1[i][1]]] for i in range(0,len(xvar_remove1))]
        possible_matches_int1 = [[[1,xvar_remove1[i][0]],[1,xvar_remove1[i][1]]] for i in range(0,len(xvar_remove1))]
    if remove_element !=0:
        possible_matches_int = [[[0,xvar_remove1[i][0]],[0,xvar_remove1[i][1]]] for i in range(0,len(xvar_remove1))]
        possible_matches_int1 = [[[1,xvar_remove1[i][0]],[1,xvar_remove1[i][1]]] for i in range(0,len(xvar_remove1))]
        
    return flatten([possible_matches_int,possible_matches_int1])

def prob_new_remove(matches,total_prob):
    """This takes in as input the diagonal elements of the mixed state and the pattern for the new elements. It outputs
    the new distribution according to the pattern"""
    prob_new_remove_int = []
    for i in range(0,len(matches)):
        prob_int_2 = 0
        for j in range(0,len(matches[0])):
            x,y = matches[i][j]
            prob_int_2 = prob_int_2 + total_prob[x][y]
        prob_new_remove_int.append(prob_int_2)
    return prob_new_remove_int

def remove_state(remove_element,length_second_last_state,total_prob):
    """In this case, we assume that if we obtain the result =1, we apply the Pauli operators to the state to obtain 
    a state we would have gotten had we obtained the result=0. The resultant state is thereform normalized."""
    xvar_remove1 = xvar_remove(remove_element, length_second_last_state)
    matches = possible_matches_remove(xvar_remove1,remove_element)
    prob_new_after_removing = prob_new_remove(matches,total_prob)
    a=len(prob_new_after_removing)
    new_list_int = [prob_new_after_removing[0:a//2],prob_new_after_removing[a//2:]]
    return new_list_int




# Patterns for swapping 


#Let us assume that the first node of the big state always has Z
def number(i,b4r,number_of_state,swap_states,length_of_big_state):
    "number of states: no. of states being swapped"
    "swap_states : the node number being swapped"
    "length_of_big_states : the length of the big state included in the swap. Let it be m"
    #For our purposes we will only we using the case where one node is being swapped for now. 
    b = b4r[i][::-1][0:length_of_big_state-2][::-1] #this gives the leftmost m-1 elements. 
    swap_nodes_new = [length_of_big_state-i-1 for i in swap_states] 
    if len(swap_states)==1:
        xnew1 = b[0:swap_nodes_new[0]]+'0'+b[swap_nodes_new[0]:]
    return int(xnew1,2)

    
    
def number1(i,b4r,number_of_state,swap_states,length_of_big_state):
    "number of states: no. of states being swapped"
    "swap_states : the node number being swapped"
    "length_of_big_states : the length of the big state included in the swap. Let it be m"
    #For our purposes we will only we using the case where one node is being swapped for now. 
    b = b4r[i][::-1][0:length_of_big_state-2][::-1]#this gives the leftmost m-1 elements. 
    swap_nodes_new = [length_of_big_state-i-1 for i in swap_states] 
    if len(swap_states)==1:
        xnew1 = b[0:swap_nodes_new[0]]+'1'+b[swap_nodes_new[0]:]
    return int(xnew1,2)

def number_with_zero_new(i,b4r,length_of_big_state):
    """b4r[i] is of length of final_state-1. """
    b = b4r[i][::-1][0:length_of_big_state-2][::-1]+'0'
    #since we have only one state that is swapped, we skip everything below
    return int(b,2)

def number1_with_zero_new(i,b4r,length_of_big_state):
    b = mod_string(b4r[i][::-1][0:length_of_big_state-2])[::-1]+'1'
    #swap_nodes_new = [length_of_big_state-i for i in swap_states]
    #since we have only one state that is swapped, we skip everything below
    return int(b,2)

def xvar(number_of_states,measures_nodes_list_big_state,finaln1,length_of_big_state):
    xgen = finaln1-1
    b4r = [bina(xgen,i) for i in range(int(pow(2, xgen)))]
    if 0 not in measures_nodes_list_big_state: 
        x_new_var = [[[int(b4r[i][j],2) for j in range(number_of_states-1)]+[number(i,b4r,number_of_states,measures_nodes_list_big_state,length_of_big_state)],[int(b4r[i][j],2)^1 for j in range(number_of_states-1)]+[number1(i,b4r,number_of_states,measures_nodes_list_big_state,length_of_big_state)]] for i in range(2**(xgen))]
    else:
        #measures_nodes_list_big_state.remove(0)
        x_new_var = [[[int(b4r[i][j],2) for j in range(number_of_states-1)]+[number_with_zero_new(i,b4r,length_of_big_state)],[int(b4r[i][j],2)^1 for j in range(number_of_states-1)]+[number1_with_zero_new(i,b4r,length_of_big_state)]] for i in range(2**(xgen))]
    return x_new_var

def zpossible(number_of_states):
    """This generates the possible patterns for Z^i1, where i1 = 0 and 1."""
    a=[i for i in product([0,1],repeat=number_of_states)]
    zpossible1 = [i for i in a if sum(i)%2==0]
    zpossible2 = [i for i in a if sum(i)%2==1]
    zpossible12 = [zpossible1, zpossible2]
    return zpossible12

def possible_matches(zpossible12,xnew1):
    """Intermediate function"""
    possible_matches = [(zpossible12[i],xnew1[j]) for i in np.arange(0,2)for j in np.arange(0,len(xnew1),1) ]
    return possible_matches

def nth_element_indexes(possible_matches1,nth,number_of_states,length_of_big_state):
    """This function generates a list. Each element of the list, tells us the contributing elements of the initial state. 
    The 0th element would tell us the contributing elements of the input state to the 0th diagonal of the final state."""
    prob_rows = [(possible_matches1[nth][0][i],possible_matches1[nth][1][j]) for i in np.arange(0,len(possible_matches1[0][0])) for j in range(2)]
    if number_of_states == 2:
        list1 = [[[prob_rows[i][0][0],prob_rows[i][1][0]],[prob_rows[i][0][1],prob_rows[i][1][1]]] for i in range(len(prob_rows))]
    elif number_of_states == 3:
        list1 = [[[prob_rows[i][0][0],prob_rows[i][1][0]],[prob_rows[i][0][1],prob_rows[i][1][1]],[prob_rows[i][0][2],prob_rows[i][1][2]]] for i in range(len(prob_rows))]
    elif number_of_states == 4:
        list1 = [[[prob_rows[i][0][0],prob_rows[i][1][0]],[prob_rows[i][0][1],prob_rows[i][1][1]],[prob_rows[i][0][2],prob_rows[i][1][2]],[prob_rows[i][0][3],prob_rows[i][1][3]]] for i in range(len(prob_rows))]
    return list1


def con_prob(number_of_states,length_of_big_state,prob,prob_new,measured_nodes):
    """
    number_of_states: This corresponds to the number of states that are being swapped
    length_of_big_state : This gives the length of the state having more than 2 qubits
    

    """
    n = number_of_states #total number of states that are being swapped
    number_of_modes = [length_of_big_state]+[2]*(number_of_states-1)
    number_of_modes_final = sum(number_of_modes)-measured_nodes
    finaln = number_of_modes_final
    prob_con = []
    [prob_con.append(prob) for i in range(number_of_states-1)]
    prob_con.append(prob_new)
    return prob_con
    
#The final state will have Z^j with j=0 and j=1. The two lists represents the components contributing to the lists. 

def mod(i):
    return (i+1)%2

def finaln(number_of_states,length_of_big_state,measured_nodes):
    #returns the length of the final state
    n = number_of_states #total number of states that are being swapped
    number_of_modes = length_of_big_state+2*(number_of_states-1)
    number_of_modes_final = number_of_modes-(measured_nodes+number_of_states-1)
    finaln1 = number_of_modes_final
    return finaln1



def new_prob(prob1,list1):
    """prob1 is a concatenated probability matrix. The outer layer corresponds to different state. The bipartite state
    comes first and then the big state. In the probability matrix of one state, we have two lists. The first 
    list corresponds to Z=0, and the second to Z=1. In this list, the list element corresponds to its X vector binary number."""
    prob_new_state=[]
    probnew1 = 0
    for j in range(0,len(list1)):
        probnew2=1
        for k in range(len(list1[j])):
            z2,z3 = list1[j][k]
            probnew2=prob1[k][z2][z3]*probnew2
        probnew1 = probnew1+probnew2
    return probnew1

def new_state_prob(number_of_states,length_of_big_state,prob,prob_new,measures_nodes_list_big_state):
    #debugging done
    measured_nodes = len(measures_nodes_list_big_state)
    list_print = []
    
    zpossible12 = zpossible(number_of_states)
    #print(zpossible12)
    finaln2 = finaln(number_of_states,length_of_big_state,measured_nodes)
    xnew1 = xvar(number_of_states,measures_nodes_list_big_state,finaln2,length_of_big_state)
    possible_matches1 = possible_matches(zpossible12,xnew1)
    
    prob1 = con_prob(number_of_states,length_of_big_state,prob,prob_new,measured_nodes)
    new_state_prob_list = []
    #print('***')
    #print(number_of_states,length_of_big_state,measured_nodes)
    for nth in range(2**(finaln2)):
        list2 = nth_element_indexes(possible_matches1,nth,number_of_states,length_of_big_state)
        var1 = new_prob(prob1,list2)
        new_state_prob_list.append(var1)
        list_print.append(list2)
        a=len(new_state_prob_list)
        new = [new_state_prob_list[0:a//2],new_state_prob_list[a//2:]]
    return new



def final_state(n,n1,n2,p,prob_link,graph):
    """Inputs: 
    n: grid size
    n1&n2 : start and end points
    p: prob of initial Werner states
    prob_link: prob of link existing between nodes. """
    """Outputs the diagonal matrix elements of the state, graph, and graph_lists"""
    new_final_list,j_final_list,graph_list_out,graph = swap_list(n,n1,n2,graph)
    out_swap_nodes,out_big_state, out_final_state,out_remove_edge,graph,graph_list=swap_list_fun(n1,n2,n,graph)
    number_of_two_states_list,size_of_big_state_list,final_state_len_list=local_geo(out_swap_nodes,out_big_state, out_final_state,out_remove_edge,graph)

    prob_final = []
    for l in range(0,len(out_swap_nodes)): #for each independent state in the loop run a loop. 
        number_of_states = number_of_two_states_list[l]
        inter_big_state = out_big_state[l]
        swapped_node_big_state = out_swap_nodes[l]
        number_of_iter_swap = len(out_remove_edge[l])
        prob = [[p,(1-p)/3],[(1-p)/3,(1-p)/3]]
        prob_new = [[p,(1-p)/3],[(1-p)/3,(1-p)/3]]
        
        for j in range(0,number_of_iter_swap):#this is the swapping loop
            # for each swap, you encounter, first more than one node of the big state involved
            #in the swap. In this case, remove till one is left. 
            number_of_states_int = number_of_states[j]+1
            len_of_big_state_int = len(inter_big_state[j])
            measures_nodes_list_big_state = swapped_node_big_state[j]


            while len(measures_nodes_list_big_state)!=1:#this brings down the number of swapping element to 1
                prob_new_int = prob_new.copy()
                remove_element = measures_nodes_list_big_state[0]
                
                prob_new = remove_state(remove_element, len_of_big_state_int,prob_new_int)
                len_of_big_state_int= len_of_big_state_int-1
                measures_nodes_list_big_state.pop(0)
            if number_of_states_int!=1:#this performs swapping
                prob_new = new_state_prob(number_of_states_int,len_of_big_state_int,prob,prob_new,measures_nodes_list_big_state)
            if number_of_states_int==1:#if only one state is there, remove elements
                prob_new = remove_state(measures_nodes_list_big_state[0],len_of_big_state_int,prob_new)
                len_of_big_state_int = len_of_big_state_int-1
        prob_final.append(prob_new)
    return prob_final,graph,graph_list,out_final_state




# Now we start with the function for calculating the entropy of the marginal. 

def state_marginals_node_list(node_list,n1,n2):
    """This function takes in as input the list of local nodes for the final state 
    Output1: local nodes of big state on which the  nodes of n1 are situated.
    Output2 : local nodes of big state on which the nodes of n2 are situated.  """
    node_list1 = node_list.copy()
    list_int_1 = [4*n1+1,4*n1+2,4*n1+3,4*n1+4]
    list_int_2 = [4*n2+1,4*n2+2,4*n2+3,4*n2+4]
    
    if node_list1[0] in list_int_1:
        index_big_state_marg = [node_list1.index(i) for i in node_list1 if i in list_int_1]
        index_big_state_marg1 = [node_list1.index(i) for i in node_list1 if i in list_int_2]
    if node_list1[0] in list_int_2:
        index_big_state_marg = [node_list1.index(i) for i in node_list1 if i in list_int_2]
        index_big_state_marg1 = [node_list1.index(i) for i in node_list1 if i in list_int_1]
    return index_big_state_marg,index_big_state_marg1

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def del_dups(seq):
    new_del_list = []
    
    for i in seq:
        if i not in new_del_list:
            new_del_list.append(i)
    return new_del_list
    

def pattern_X_coherent_info(n1,n2,n,node_list):
    """This returns the pattern for the x_list. """
    list_int_1 = [4*n1+1,4*n1+2,4*n1+3,4*n1+4]
    list_int_2 = [4*n2+1,4*n2+2,4*n2+3,4*n2+4]
    state_marginals_node,position_not_nodes = state_marginals_node_list(node_list,n1,n2)
    x_list1 = []
    x_list2 = []
    len_final_list = len(node_list)
    len_marginal_state = len(state_marginals_node)
    len_removed_states = len(node_list)-len_marginal_state
    a1 =[]
    possible_string = b4r_list(len_final_list) # creates a list of binary number of length final_list-1
    possible_list = [list(i) for i in possible_string] # creates a list of the binary strings
    for i in possible_list:
        for j in position_not_nodes:
            i[j] = '0'
    position_node_nodes_new = del_dups(possible_list)

    
    if len_marginal_state!=1:
        for i in position_node_nodes_new:
            a1.append([i.copy() for j in range(0,2**(len_removed_states))])
    flatten(a1)
    replace2 = b4r_list(len_removed_states) #creates the pattern for interpersing
    replace1 = [list(i) for i in replace2]
    pattern = []
    pattern_int = []
    pattern2 = []
    for i in a1:
        for j in range(0,len(replace1)):
            new_list_for_final = i[j].copy()
            for k in range(0,len_removed_states):
                new_list_for_final[position_not_nodes[k]] = replace1[j][k]
    
            new_list_for_final.pop(0)
            len_new_list = len(new_list_for_final)
            
            pattern.append(new_list_for_final.copy())
            pattern_int.append(''.join(new_list_for_final.copy()))    
    
    length_of_list = len(pattern_int.copy())
    pattern2 = pattern_int[:length_of_list//2]
    pattern3 = [int(var1[::-1],2) for var1 in pattern2]
            
    pattern_final = list(split(pattern3, 2**(len_marginal_state-1)))
    
    return pattern_final


def matches_coh_info(pattern_final):
    #this function generates the pattern required for the selecting the probabilities
    matches = []
    for i in range(0,len(pattern_final)):
        a1 = [[0]+[pattern_final[i]],[1]+[pattern_final[i]]]
        matches.append(a1)
    
    return matches+matches

def coh_info_prob(pattern1,prob_marg):
    prob_new = []
    for i in range(0,len(pattern1)):
        prob_int = 0
        for j in range(0,2):
            
            z_var = pattern1[i][j][0]
            for k in range(0,len(pattern1[0][0][1])):
                x_var = pattern1[i][j][1][k]
                prob_int = prob_int +prob_marg[z_var][x_var]/2

        prob_new.append(prob_int)
    return prob_new
    

def conditional_entropy(state_prob,n1,n2,node_list,n):
    
    if len(node_list)!=2:
        prob_marg = state_prob
        pattern_final = pattern_X_coherent_info(n1,n2,n,node_list)
        pattern1 = matches_coh_info(pattern_final)
        prob_cond= coh_info_prob(pattern1,prob_marg)
    else:
        prob_int_entropy =flatten(state_prob)
        prob_cond = [sum(prob_int_entropy)/2,sum(prob_int_entropy)/2]    
    entropy_cond = sum([-j*log2_mod(j) for j in prob_cond])
    
    return entropy_cond

def entropy(state_prob):
    prob_int_entropy =flatten(state_prob)
    entropy_cond = sum([-j*log2_mod(j) for j in  prob_int_entropy])
    return entropy_cond


def matches_coh_info(pattern_final):
    #this function generates the pattern required for the selecting the probabilities
    matches = []
    for i in range(0,len(pattern_final)):
        a1 = [[0]+[pattern_final[i]],[1]+[pattern_final[i]]]
        matches.append(a1)
    
    return matches+matches


def log2_mod(i):
    if i==0:
        x_int = 0
    else:
        x_int = math.log2(i)
    return x_int
    
    
    
#----------- FINAL FUNCTION FOR THE REGION PLOTS IN FIG 7 and 8-------------------
def region_final(n,alice,bob,link_prob):
    """This function takes in n: grid size, alice and bob's location, and link probability. 
    Output: list of lists. Each sublists corresponds to a list of coherent information for a given link fidelity, and 
    the list of coherent information is with respect to different regions. 
    
    """

    n1= alice
    n2 = bob
    p_list = np.arange(.95, 1.0, 0.001)
    prob_link_list = [link_prob]
    list_disconnected = []
    coh_iter_link = []
    for prob_link in prob_link_list:
        if prob_link ==1:
            repeat_number = 1
        else: 
            repeat_number = 100
        G_list = [deleted_edges_graph(n,prob_link) for g33 in range(0,repeat_number)]
        region_list1 = region_graph(n,n1,n2,G_list[0])
        coh_iter_werner = []
        for j1 in range(0,len(p_list)):
            p = p_list[j1]
            coh_information_region_list =[]

            for k1 in range(0,len(region_list1)):

                final_coh_iter = 0
                for iter1 in range(0,repeat_number):
                    print(prob_link,j1,k1,iter1)
                    region_list = region_graph(n,n1,n2,G_list[iter1])
                    graph = region_list[k1]
                    prob_final,graph,graph_list,out_final_state = final_state(n,n1,n2,p,prob_link,graph)
                    final_coh_information = 0

    #                 print('**')
    #                 print(prob_final)
                    for prob_length in range(0,len(prob_final)):

                        #print(sum(flatten(flatten(prob_final))))
                        try:
                            pattern_final = pattern_X_coherent_info(n1,n2,n,out_final_state[prob_length])
                            pattern1 = matches_coh_info(pattern_final)
                            coh_info_prob(pattern1,prob_final[prob_length])
                            a1111=conditional_entropy(prob_final[prob_length],n1,n2,out_final_state[prob_length],n)
                            a222=entropy(prob_final[prob_length])
                            final_coh_information = max(a1111-a222,0)+final_coh_information
                            print(final_coh_information)
                        except:  
                            IndexError
                    final_coh_iter = final_coh_information + final_coh_iter
                final_coh_iter = final_coh_iter/repeat_number
                coh_information_region_list.append(final_coh_iter)
            coh_iter_link.append(coh_information_region_list)
        
    return coh_iter_link