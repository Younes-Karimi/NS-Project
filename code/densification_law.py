#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 13:48:51 2018

@author: Emre_Ugurlu
"""

import numpy as np
import pandas as pd
import networkx as nx
import time
import datetime
from copy import deepcopy
import matplotlib.pyplot as plt 
import math
from decimal import Decimal

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

graphSnapList = list(map(graphSnapShots, G_unique, [data]*len(G_unique)))

fig = plt.figure()
ax = plt.gca()

# Find the slope and densification
a = [len(g.nodes()) for g in graphSnapList]
b = [len(g.edges()) for g in graphSnapList]
plt.scatter(a, b)
z = np.polyfit([math.log10(x) for x in a],[math.log10(y) for y in b],1) # Fitting slope
#Plot the slope based on densification law
plt.plot(a, (10**z[1])*(a**z[0]), c='y', label='Regression Line [a=%s;b=%s]' %(round(Decimal(z[0]),3),round(Decimal(z[1]),3)))

ax.set_yscale('log')
ax.set_xscale('log')
plt.title('Densification Law')
plt.xlabel('Number of Nodes')
plt.ylabel('Number of Edges')
plt.legend()
plt.show()
daylist = np.linspace(1, len(graphSnapList), len(graphSnapList))