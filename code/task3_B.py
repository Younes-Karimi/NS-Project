#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 15:07:42 2018

@author: Emre_Ugurlu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 13:33:45 2018

@author: Emre_Ugurlu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:13:49 2018

@author: Emre_Ugurlu
"""

import numpy as np
import pandas as pd
import networkx as nx
import time
import datetime
from copy import deepcopy
import matplotlib.pyplot as plt 

def graphSnapShots(dateArray, data):
    df = pd.DataFrame(data,columns=['c1','c2','c3'])
    dg = df.loc[df['c1']== dateArray]
    var = np.array(dg)
    edge_data = var[:,[1,2]]
    df = pd.DataFrame(edge_data,columns=['c1','c2'])
    dg = pd.DataFrame({'count' : df.groupby(['c1', 'c2']).size()}).reset_index()
    edge_weight_data = np.array(dg)
    # Removing self-loops
    edge_weight_data = np.delete(edge_weight_data, np.where(edge_weight_data[:,0]==edge_weight_data[:,1]), axis=0)
    # Converting numpy 2-D array into list of tuples and adding the weighted edges to the graph G
    G = nx.DiGraph()
    G.add_weighted_edges_from(list(map(tuple, edge_weight_data)))
    return G

def getWeakComponents(graph):
    g = sorted(nx.weakly_connected_components(graph), key=len, reverse=True)[0]
    return len(g)

def getStrongComponents(graph):
    g = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)[0]
    return len(g)

def density(graph):
    edges = graph.size()
    nodes = len(graph)
    g = (2*edges)/((nodes)*(nodes-1))
    return g
def avr_clust(graph):
    a = nx.to_undirected(graph)
    g = nx.average_clustering(a)
    return g

# Importing dataset
orig_data = pd.read_csv("532projectdataset.txt", delimiter=" ", header=None)

# Converting into numpy array
data = np.array(orig_data)
data1 = deepcopy(data)

# Removing the rows in the dataset that correspond to weekends
data[:,0] = np.asarray(data[:,0], dtype='datetime64[s]')
data1[:,0] = np.asarray(data1[:,0], dtype='datetime64[s]')

#print("\n Unique values === ", len(np.unique(data[:,0])))
vfunc = np.vectorize(lambda t: t.weekday())
data1[:,0] = vfunc(data1[:,0])
data = np.delete(data, np.where((data1[:,0]==5) | (data1[:,0]==6)), axis=0)



vfunc1 = np.vectorize(lambda t: t.date())
data[:,0] = vfunc1(data[:,0])
G_unique = np.unique(data[:,0]).tolist()
print("\n Unique values === ", len(G_unique))

#print(G_unique)

graphSnapList = list(map(graphSnapShots, G_unique, [data]*len(G_unique)))
print(graphSnapList[0].nodes())
print(graphSnapList[0].edges())


daylist = np.linspace(1, len(graphSnapList), len(graphSnapList))

'''
weakCompList = list(map(getWeakComponents, graphSnapList))
strongCompList = list(map(getStrongComponents, graphSnapList))
#Density of graph
Density = list(map(density, graphSnapList))
'''
#Average Clustering Coefficent 

avr_cl_coefficient = list(map(avr_clust, graphSnapList))
'''
plt.plot(daylist,weakCompList)
plt.title('Evolution of the Largest Weakly Connected Components')
plt.xlabel('DAY')
plt.ylabel('Size of Weakly Connected Comp.')
plt.show()

plt.plot(daylist,strongCompList)
plt.title('Evolution of the Largest Strongly Connected Components')
plt.xlabel('DAY')
plt.ylabel('Size of Strongly Connnected Comp.')
plt.show()

plt.plot(daylist, Density)
plt.title('Density of the Graph')
plt.xlabel('DAY')
plt.ylabel('Density')
plt.show()
'''
plt.plot(daylist, avr_cl_coefficient)
plt.title('Average Clustering Coefficient of Each Gτ')
plt.xlabel('DAY')
plt.ylabel('Average Clustering Coefficient')
plt.show()