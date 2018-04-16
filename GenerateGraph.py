import numpy as np
import pandas as pd
import networkx as nx
import time
from datetime import datetime




# Importing dataset
orig_data = pd.read_csv("532projectdataset.txt", delimiter=" ", header=None)
# Converting into numpy array
data = np.array(orig_data)


# G is a Directed Graph
G = nx.DiGraph()
from_emails = (data[:,1]).tolist()
from_emails.extend((data[:,2]).tolist())
# Add all the email addresses to the graph G
G.add_nodes_from(from_emails)

# Removing the rows in the dataset that correspond to weekends
data[:,0] = np.asarray(data[:,0], dtype='datetime64[s]')
vfunc = np.vectorize(lambda t: t.weekday())
data[:,0] = vfunc(data[:,0])
data = np.delete(data, np.where((data[:,0]==5) | (data[:,0]==6)), axis=0)

# Grouping dataset to get the weight of sent/received edge
edge_data = data[:,[1,2]]
df = pd.DataFrame(edge_data,columns=['c1','c2'])
dg = pd.DataFrame({'count' : df.groupby(['c1', 'c2']).size()}).reset_index()
edge_weight_data = np.array(dg)

# Removing self-loops
edge_weight_data = np.delete(edge_weight_data, np.where(edge_weight_data[:,0]==edge_weight_data[:,1]), axis=0)

# Converting numpy 2-D array into list of tuples and adding the weighted edges to the graph G
G.add_weighted_edges_from(list(map(tuple, edge_weight_data)))


print("\n************************ GRAPH DETAILS ********************************\n")
print("Number of nodes in G === ", G.number_of_nodes())
print("Number of edges in G === ", G.number_of_edges())
print("Number of Self-loops === ", G.number_of_selfloops())
