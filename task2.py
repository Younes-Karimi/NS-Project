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




def getEgoNet(H, node):
	return nx.ego_graph(H,node).to_undirected()
	
def getTotalWeight(I):
	return sum([k['weight'] for i,j,k in I.edges(data=True)])
	
def getPrinEigVal(I):
	wAdjMat = (nx.adjacency_matrix(I)).todense()
	return np.real(max(np.linalg.eigvals(wAdjMat)))
	
def getOutOfNorm(origEdges, powLaw):
	return (max(powLaw,origEdges)/min(powLaw,origEdges)) * (math.log10(abs(origEdges-powLaw) + 1))



'''
orig_data = pd.read_csv("532projectdataset.txt", delimiter=" ", header=None)
data = np.array(orig_data)


G = nx.DiGraph()
from_emails = (data[:,1]).tolist()
from_emails.extend((data[:,2]).tolist())

G.add_nodes_from(from_emails)

print("Number of nodes === ", len(G.nodes()))

print("Shape of dataset (before removing weekends) === ", np.shape(data))
data[:,0] = np.asarray(data[:,0], dtype='datetime64[s]')
vfunc = np.vectorize(lambda t: t.weekday())
data[:,0] = vfunc(data[:,0])
print("Shape of dataset (before removing weekends and after updating the 0th column) === ", np.shape(data))
print("Data  === \n", data)
data = np.delete(data, np.where((data[:,0]==5) | (data[:,0]==6)), axis=0)
print("\nShape of dataset (after removing weekends)", np.shape(data))
print("Data === \n", data)


edge_data = data[:,[1,2]]
df = pd.DataFrame(edge_data,columns=['c1','c2'])
print("\nTotal sum after grouping === ", np.sum(np.array(df.groupby(['c1','c2']).size())))
dg = pd.DataFrame({'count' : df.groupby(['c1', 'c2']).size()}).reset_index()
edge_weight_data = np.array(dg)
print("\nTotal number of edges after grouping === ", np.shape(edge_weight_data)[0])
print("\n Self loops overlap === ", len(np.where(edge_weight_data[:,0]==edge_weight_data[:,1])[0]))
edge_weight_data = np.delete(edge_weight_data, np.where(edge_weight_data[:,0]==edge_weight_data[:,1]), axis=0)

G.add_weighted_edges_from(list(map(tuple, edge_weight_data)))
print("Total # of edges after removing self-loops == ", len(G.edges()))
print("football@cnnsi.com === ", G['football@cnnsi.com'])

## Check self-loops
print("\n Self-loops === ", G.number_of_selfloops())
'''


G = GG.GraphGenerator()
print("\n************************ GRAPH DETAILS ********************************\n")
print("Number of nodes in G === ", G.number_of_nodes())
print("Number of edges in G === ", G.number_of_edges())
print("Number of Self-loops === ", G.number_of_selfloops())




### Calculating list of all egonets for each node
EgoNet_list = list(map(getEgoNet, [G]*len(G.nodes()), list(G.nodes())))

### Calculating number of nodes and number of edges in egonets for each node in the graph

EgoNet_nodesLength_list = list(map(len, EgoNet_list))
EgoNet_edgesLength_list = list(map(nx.number_of_edges, EgoNet_list))
### Removing nodes with zero degree
EgoNet_nodesLength_list = [n for i,n in enumerate(EgoNet_nodesLength_list) if EgoNet_edgesLength_list[i]]
### Calculating total weight and principal eigen values in egonets for each node in the graph
EgoNet_totalWeight_list = list(map(getTotalWeight, EgoNet_list))
EgoNet_totalWeight_list = [n for i,n in enumerate(EgoNet_totalWeight_list) if EgoNet_edgesLength_list[i]]
EgoNet_PrinEigVal_list = list(map(getPrinEigVal, EgoNet_list))
EgoNet_PrinEigVal_list = [n for i,n in enumerate(EgoNet_PrinEigVal_list) if EgoNet_edgesLength_list[i]]
EgoNet_edgesLength_list = [n for n in EgoNet_edgesLength_list if n]

print("\nNumber of zero edges === ", len([x for x in EgoNet_edgesLength_list if x==0]))
print("Number of non-zero edges === ", len([x for x in EgoNet_edgesLength_list if x]))
print("Number of zero nodes === ", len([x for x in EgoNet_nodesLength_list if x==0]))
print("Number of non-zero nodes === ", len([x for x in EgoNet_nodesLength_list if x]))

print("\nNumber of zero weights === ", len([x for x in EgoNet_totalWeight_list if x==0]))
print("Number of non-zero weights === ", len([x for x in EgoNet_totalWeight_list if x]))
print("Number of zero eigen values === ", len([x for x in EgoNet_PrinEigVal_list if x==0]))
print("Number of non-zero eigen values === ", len([x for x in EgoNet_PrinEigVal_list if x]))



fig = plt.figure()
ax = plt.gca()

'''
z = np.polyfit([math.log10(x) for x in EgoNet_nodesLength_list], [math.log10(y) for y in EgoNet_edgesLength_list], 1)
print("\nPolyfit Coefficients  ================  ", z)
powlawEgoedges = (10**z[1])*(EgoNet_nodesLength_list**z[0])
print("Number of non-zero power-law edges === ", len([x for x in powlawEgoedges if x]))


OutOfNorm_list = list(map(getOutOfNorm, EgoNet_edgesLength_list, powlawEgoedges))
OutOfNorm_list = np.argsort(-np.array(OutOfNorm_list))
print("len(OutOfNorm_list[:20])+len(OutOfNorm_list[20:]) === ", len(OutOfNorm_list[:20])+len(OutOfNorm_list[20:]))


bins = np.logspace(math.log10(min(EgoNet_nodesLength_list)), math.log10(max(EgoNet_nodesLength_list)+0.1), 21)
arr = np.digitize(np.array(EgoNet_nodesLength_list), bins)
li = list((np.where(arr==x)[0]).tolist() for x in np.unique(arr))
EgoNet_nodes_logbin = [[EgoNet_nodesLength_list[j] for j in i] for i in li]
EgoNet_edges_logbin = [[EgoNet_edgesLength_list[j] for j in i] for i in li]



#plt.hist(EgoNet_nodesLength_list, bins=np.logspace(math.log10(min(EgoNet_nodesLength_list)-0.1), math.log10(max(EgoNet_nodesLength_list)+0.1), 5))
#plt.loglog(EgoNet_nodesLength_list, EgoNet_edgesLength_list, basex=10, basey=10, c='b', label='Eu Vs. Vu')
#plt.loglog([1,max(EgoNet_nodesLength_list)], [1,max(EgoNet_nodesLength_list)], basex=10, basey=10, c='r', label='Star')
#plt.loglog([1,max(EgoNet_nodesLength_list)], [1,max(EgoNet_nodesLength_list)**2], basex=10, basey=10, c='g', label='Clique')

ax.scatter(np.array(EgoNet_nodesLength_list)[OutOfNorm_list[20:]].tolist(), np.array(EgoNet_edgesLength_list)[OutOfNorm_list[20:]].tolist(), c='c', marker='o', label='Eu Vs. Vu')
ax.scatter(np.array(EgoNet_nodesLength_list)[OutOfNorm_list[:20]].tolist(), np.array(EgoNet_edgesLength_list)[OutOfNorm_list[:20]].tolist(), c='k', marker='^', label='Eu Vs. Vu [Top 20 Out-Of-Norm]')
ax.plot([1,max(EgoNet_nodesLength_list)], [1,max(EgoNet_nodesLength_list)], c='r', label='Star')
ax.plot([1,max(EgoNet_nodesLength_list)], [2,max(EgoNet_nodesLength_list)*2], c='g', label='Clique')

### Taking mean of x-axis and mean of y-axis.
ax.plot([np.mean(x) for x in EgoNet_nodes_logbin], [np.mean(x) for x in EgoNet_edges_logbin], c='m', label='Logarithmic Binning')
### Taking median of x-axis and median of y-axis.
#ax.plot([median(x) for x in EgoNet_nodes_logbin], [median(x) for x in EgoNet_edges_logbin], c='m', label='Logarithmic Binning')

plt.plot(EgoNet_nodesLength_list, powlawEgoedges, c='y', label='Power Law [a=%s;b=%s]' %(round(Decimal(z[0]),3),round(Decimal(z[1]),3)))
ax.set_yscale('log')
ax.set_xscale('log')

plt.xlabel('Vu')
plt.ylabel('Eu')
'''





z = np.polyfit([math.log10(x) for x in EgoNet_totalWeight_list], [math.log10(y) for y in EgoNet_PrinEigVal_list], 1)
print("\nPolyfit Coefficients  ================  ", z)
powlawEgoedges = (10**z[1])*(EgoNet_totalWeight_list**z[0])
print("Number of non-zero power-law edges === ", len([x for x in powlawEgoedges if x]))


OutOfNorm_list = list(map(getOutOfNorm, EgoNet_PrinEigVal_list, powlawEgoedges))
OutOfNorm_list = np.argsort(-np.array(OutOfNorm_list))
print("len(OutOfNorm_list[:20])+len(OutOfNorm_list[20:]) === ", len(OutOfNorm_list[:20])+len(OutOfNorm_list[20:]))
#plt.loglog(EgoNet_totalWeight_list, np.real(EgoNet_PrinEigVal_list), basex=10, basey=10)
#plt.loglog(EgoNet_totalWeight_list, np.imag(EgoNet_PrinEigVal_list), basex=10, basey=10)
ax.scatter(np.array(EgoNet_totalWeight_list)[OutOfNorm_list[20:]].tolist(), np.array(EgoNet_PrinEigVal_list)[OutOfNorm_list[20:]].tolist(), c='c', marker='o', label='Lambda(w,u) Vs. Wu')
ax.scatter(np.array(EgoNet_totalWeight_list)[OutOfNorm_list[:20]].tolist(), np.array(EgoNet_PrinEigVal_list)[OutOfNorm_list[:20]].tolist(), c='k', marker='^', label='Lambda(w,u) Vs. Wu [Top 20 Out-Of-Norm]')

#plt.loglog(EgoNet_totalWeight_list, EgoNet_PrinEigVal_list, basex=10, basey=10)

plt.plot(EgoNet_totalWeight_list, powlawEgoedges, c='y', label='Power Law [a=%s;b=%s]' %(round(Decimal(z[0]),3),round(Decimal(z[1]),3)))
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('Wu')
plt.ylabel('Lambda(w,u)')
'''
EgoNet_totalWeight_list = [n for i,n in enumerate(EgoNet_totalWeight_list) if n]
EgoNet_PrinEigVal_list = [n for n in EgoNet_PrinEigVal_list if n]
z = np.polyfit([math.log10(x) for x in EgoNet_totalWeight_list], [math.log10(y) for y in EgoNet_PrinEigVal_list], 1)
print("Polyfit Coefficients  ================  ", z)
plt.loglog(EgoNet_totalWeight_list, (10**z[1])*(EgoNet_totalWeight_list**z[0]), basex=10, basey=10, c='y', label='Wu Vs. Lambda(w,u)[a=%s;b=%s]' %(round(Decimal(z[0]),3),round(Decimal(z[1]),3)))
'''

plt.legend()
plt.show()
















