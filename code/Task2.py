#################################################################################################
### Author: R. Padmavathi Iyer [NetID: RI928946]
### This file contains the code for Task 2 of the project.
#################################################################################################



####################### Imports #######################
import numpy as np
import pandas as pd
import networkx as nx
import time
from datetime import datetime
import matplotlib.pyplot as plt
import math
from decimal import Decimal
from statistics import median
import GenerateGraph as GG



####################### Returns the ego-network for a given node and graph H #######################
def getEgoNet(H, node):
	return nx.ego_graph(H,node).to_undirected()
	
####################### Returns the total weight of a given graph I #######################
def getTotalWeight(I):
	return sum([k['weight'] for i,j,k in I.edges(data=True)])
	
####################### Returns the principal eigenvalues of weighted adjacency matrix for a given graph I #######################
def getPrinEigVal(I):
	wAdjMat = (nx.adjacency_matrix(I)).todense()
	return np.real(max(np.linalg.eigvals(wAdjMat)))

####################### Returns the out-of-the-norm score of a node #######################	
def getOutOfNorm(origEdges, powLaw):
	return (max(powLaw,origEdges)/min(powLaw,origEdges)) * (math.log10(abs(origEdges-powLaw) + 1))



### G is the email correspondence network
G = GG.GraphGenerator()


### Calculating list of all egonets for each node
EgoNet_list = list(map(getEgoNet, [G]*len(G.nodes()), list(G.nodes())))

### Calculating number of nodes and number of edges in egonets for each node in the graph
EgoNet_nodesLength_list = list(map(len, EgoNet_list))
EgoNet_edgesLength_list = list(map(nx.number_of_edges, EgoNet_list))
### Removing nodes with zero degree
EgoNet_nodesLength_list = [n for i,n in enumerate(EgoNet_nodesLength_list) if EgoNet_edgesLength_list[i]]
### Calculating total weight and principal eigen values in egonets for each node in the graph
EgoNet_totalWeight_list = list(map(getTotalWeight, EgoNet_list))
### Removing total-weight entries corresponding to zero degree nodes
EgoNet_totalWeight_list = [n for i,n in enumerate(EgoNet_totalWeight_list) if EgoNet_edgesLength_list[i]]
EgoNet_PrinEigVal_list = list(map(getPrinEigVal, EgoNet_list))
### Removing principal eigenvalues corresponding to zero degree nodes
EgoNet_PrinEigVal_list = [n for i,n in enumerate(EgoNet_PrinEigVal_list) if EgoNet_edgesLength_list[i]]
### Removing edges entries of zero degree nodes
EgoNet_edgesLength_list = [n for n in EgoNet_edgesLength_list if n]


fig = plt.figure()
ax = plt.gca()

### UNCOMMENT THIS COMMENETED BLOCK TO SEE THE OUTPUT FOR EDGES VS. NODES GRAPH
'''
### Fitting line for edges vs. nodes graph
z = np.polyfit([math.log10(x) for x in EgoNet_nodesLength_list], [math.log10(y) for y in EgoNet_edgesLength_list], 1)
powlawEgoedges = (10**z[1])*(EgoNet_nodesLength_list**z[0])

### Calculating the out-of-the-norm score of each node and sorting the scores of all nodes in descending order
OutOfNorm_list = list(map(getOutOfNorm, EgoNet_edgesLength_list, powlawEgoedges))
OutOfNorm_list = np.argsort(-np.array(OutOfNorm_list))

### Performing logarithmic binning
#bins = np.logspace(math.log10(min(EgoNet_nodesLength_list)), math.log10(max(EgoNet_nodesLength_list)+0.1), 21)
bins = np.array([2,3,13,113,1113,11113])
arr = np.digitize(np.array(EgoNet_nodesLength_list), bins)
li = list((np.where(arr==x)[0]).tolist() for x in np.unique(arr))
EgoNet_nodes_logbin = [[EgoNet_nodesLength_list[j] for j in i] for i in li]
EgoNet_edges_logbin = [[EgoNet_edgesLength_list[j] for j in i] for i in li]

### Performing the plots for edges vs. nodes graph
ax.scatter(np.array(EgoNet_nodesLength_list)[OutOfNorm_list[20:]].tolist(), np.array(EgoNet_edgesLength_list)[OutOfNorm_list[20:]].tolist(), c='c', marker='o', label='Eu Vs. Vu')
ax.scatter(np.array(EgoNet_nodesLength_list)[OutOfNorm_list[:20]].tolist(), np.array(EgoNet_edgesLength_list)[OutOfNorm_list[:20]].tolist(), c='k', marker='^', label='Eu Vs. Vu [Top 20 Out-Of-Norm]')
ax.plot([1,max(EgoNet_nodesLength_list)], [1,max(EgoNet_nodesLength_list)], c='r', label='Star')
ax.plot([1,max(EgoNet_nodesLength_list)], [1,max(EgoNet_nodesLength_list)**2], c='g', label='Clique') ### E = V^2

### Fitting line for logarithmic bins of edges vs. nodes graph
zz1 = np.polyfit([math.log10(np.mean(x)) for x in EgoNet_nodes_logbin], [math.log10(np.mean(y)) for y in EgoNet_edges_logbin], 1)
zz2 = np.polyfit([math.log10(median(x)) for x in EgoNet_nodes_logbin], [math.log10(median(y)) for y in EgoNet_edges_logbin], 1)
### Performing the plots for fitting lines of edges vs. nodes graph
X_axis = [np.mean(x) for x in EgoNet_nodes_logbin]
plt.plot(EgoNet_nodesLength_list, powlawEgoedges, c='y', label='Power Law [a=%s;b=%s]' %(round(Decimal(z[0]),3),round(Decimal(z[1]),3)))
### Taking mean of x-axis and mean of y-axis.
plt.plot(X_axis, (10**zz1[1])*(X_axis**zz1[0]), c='m', label='Logarithmic Binning [Using mean]')
### Taking median of x-axis and median of y-axis.
plt.plot(X_axis, (10**zz2[1])*(X_axis**zz2[0]), c='b', label='Logarithmic Binning [Using median]')
ax.set_yscale('log')
ax.set_xscale('log')

plt.xlabel('Vu')
plt.ylabel('Eu')
plt.legend()
plt.show()
'''



### Fitting line for principal eigenvalue vs. total weight plot
z = np.polyfit([math.log10(x) for x in EgoNet_totalWeight_list], [math.log10(y) for y in EgoNet_PrinEigVal_list], 1)
powlawEgoedges = (10**z[1])*(EgoNet_totalWeight_list**z[0])
### Calculating out-of-the-norm scores for principal eigenvalue vs. total weight graph and sorting them
OutOfNorm_list = list(map(getOutOfNorm, EgoNet_PrinEigVal_list, powlawEgoedges))
OutOfNorm_list = np.argsort(-np.array(OutOfNorm_list))
### Performing all the plots for principal eigenvalue vs. total weight graph
ax.scatter(np.array(EgoNet_totalWeight_list)[OutOfNorm_list[20:]].tolist(), np.array(EgoNet_PrinEigVal_list)[OutOfNorm_list[20:]].tolist(), c='c', marker='o', label='Lambda(w,u) Vs. Wu')
ax.scatter(np.array(EgoNet_totalWeight_list)[OutOfNorm_list[:20]].tolist(), np.array(EgoNet_PrinEigVal_list)[OutOfNorm_list[:20]].tolist(), c='k', marker='^', label='Lambda(w,u) Vs. Wu [Top 20 Out-Of-Norm]')
plt.plot(EgoNet_totalWeight_list, powlawEgoedges, c='y', label='Power Law [a=%s;b=%s]' %(round(Decimal(z[0]),3),round(Decimal(z[1]),3)))
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('Wu')
plt.ylabel('Lambda(w,u)')
plt.legend()
plt.show()















