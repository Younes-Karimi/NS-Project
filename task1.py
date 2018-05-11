import GenerateGraph as GG
import networkx as nx
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import collections
# import math
from scipy.stats import expon
plt.style.use('seaborn-whitegrid')


G = GG.GraphGenerator()


# Task 1-1 ===============================================================

min_in = 1000000
min_out = 1000000
min_total = 1000000
max_in = 0
max_out = 0
max_total = 0

Average_in = G.number_of_edges()/G.number_of_nodes()
Average_out = G.number_of_edges()/G.number_of_nodes()
Average_total = (2*G.number_of_edges())/G.number_of_nodes()

for i in G.nodes:

	in_degree = G.in_degree(i)
	out_degree = G.out_degree(i)
	degree = G.degree(i)

	if (in_degree < min_in):
		min_in = in_degree
	if (out_degree < min_out):
		min_out = out_degree
	if (degree < min_total):
		min_total = degree
	if (in_degree > max_in):
		max_in = in_degree
	if (out_degree > max_out):
		max_out = out_degree
	if (degree > max_total):
		max_total = degree
	
bidirectional_edges = 0.5 * len([ 1 for (u,v) in G.edges() if u in G[v] ])

# Getting the diameter of the graph
UDG = nx.Graph()
UDG = G.to_undirected()
print (nx.info(UDG))
connected_components = nx.connected_component_subgraphs(UDG)

component_node_count = 0
largest_components = [0]
diameter = 0
# for component in connected_components:
# 	if len(component.nodes()) >= component_node_count:
# 		component_node_count = len(component.nodes())
# 		largest_components.append(component_node_count)

Gtmp = nx.Graph()
weakly_connected_components = nx.weakly_connected_component_subgraphs(G)
for component in weakly_connected_components:
	print ("len: ", len(component.edges()))
	Gtmp = component.to_undirected()
	print ("diameter", nx.diameter(Gtmp))


# for component in connected_components:
# 	print ("len: ", len(component.edges()))
# 	print ("diameter", nx.diameter(component))


# for x in range(len(largest_components)):
# 	print ("largest_components: ", largest_components[x])

# for component in connected_components:
# 	if len(component.nodes()) == 78524:
# 		diameter = nx.diameter(component)
# 	elif len(component.nodes()) > 20:
# 		diameter = 100
# 	else:
# 		diameter = 2
	# for i in range(len(largest_components)):
	# 	if len(component.nodes()) == largest_components[i]:
	# 		tmp = nx.diameter(component)
	# 		print ("tmp",tmp)
	# 	if tmp > diameter:
	# 		diameter = tmp

# [print(len(component.nodes())) for component in connected_components]
# diameter = [max(nx.diameter(component)) for component in connected_components]
# diameter = nx.diameter(connected_components)
# connected_components = nx.strongly_connected_component_subgraphs(G)
# for component in connected_components:
# 	print (nx.diameter(component))
# 	print (nx.info(component))
# for component in connected_components:
# 	print (component.nodes())

# diameter = [max(nx.diameter(component)) for component in connected_components]

print("\n************************ Task1-1 ************************\n")
print("00. Number of self-loops === ", G.number_of_selfloops())
print("01. Number of nodes in G === ", G.number_of_nodes())
print("02. Number of edges in G === ", G.number_of_edges())
print("03. Number of bidirectional edges === ", bidirectional_edges)
print("04. Minimum number of incoming correspondings === ", min_in)
print("05. Minimum number of outgoing correspondings  === ", min_out)
print("06. Minimum number of total correspondings  === ", min_total)
print("07. Maximum number of incoming correspondings  === ", max_in)
print("08. Maximum number of outgoing correspondings  === ", max_out)
print("09. Maximum number of total correspondings  === ", max_total)
print("10. Average number of incoming correspondings  === ", Average_in)
print("11. Average number of outgoing correspondings  === ", Average_out)
print("12. Average number of total correspondings  === ", Average_total)
print("13. Diameter of the network === ", diameter)

# Task 1-2 ===============================================================

# Degree Distribution
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
degree_count = collections.Counter(degree_sequence)
deg, deg_cnt = zip(*degree_count.items())
plt.plot(deg, deg_cnt, 'gs', label='degree')

# Degree Regression
deg = np.asarray(deg, dtype=float)
# Removing zero-degree element
deg = deg[:-1]
deg_cnt = np.asarray(deg_cnt, dtype=float)
# Removing zero-degree element
deg_cnt = deg_cnt[:-1]
deg_log = np.log10(deg)
deg_cnt_log = np.log10(deg_cnt)
deg_coef = np.polyfit(deg_log,deg_cnt_log,1)
labl = 'Degree', deg_coef[0], deg_coef[1]
plt.plot(deg, [10 ** ((deg_coef[0] - 0.09) * x + deg_coef[1] + 0.25) for x in deg_log], linestyle='-', color='k', label=labl)


# In-degree Distribution
in_degree_sequence = sorted([d for n, d in G.in_degree()], reverse=True)
in_degree_count = collections.Counter(in_degree_sequence)
in_deg, in_cnt = zip(*in_degree_count.items())
# plt.plot(in_deg, in_cnt, 'b^', label='in degree')

# In-degree Regression
in_deg = np.asarray(in_deg, dtype=float)
# Removing zero-degree element
in_deg = in_deg[:-1]
in_cnt = np.asarray(in_cnt, dtype=float)
# Removing zero-degree element
in_cnt = in_cnt[:-1]
deg_log = np.log10(in_deg)
deg_cnt_log = np.log10(in_cnt)
in_deg_coef = np.polyfit(deg_log,deg_cnt_log,1)
labl = 'In-degree', in_deg_coef[0], in_deg_coef[1]
# plt.plot(in_deg, [10 ** ((in_deg_coef[0] - 0.09) * x + in_deg_coef[1] + 0.15) for x in deg_log], linestyle='--', color='c', label=labl)


# Out-degree Distribution
out_degree_sequence = sorted([d for n, d in G.out_degree()], reverse=True)
out_degree_count = collections.Counter(out_degree_sequence)
out_deg, out_cnt = zip(*out_degree_count.items())
# plt.plot(out_deg, out_cnt, 'ro', label='out degree')

# Out-degree Regression
# Degree Regression
out_deg = np.asarray(out_deg, dtype=float)
# Removing zero-degree element
out_deg = out_deg[:-1]
out_cnt = np.asarray(out_cnt, dtype=float)
# Removing zero-degree element
out_cnt = out_cnt[:-1]
deg_log = np.log10(out_deg)
deg_cnt_log = np.log10(out_cnt)
out_deg_coef = np.polyfit(deg_log,deg_cnt_log,1)
labl = 'Out-degree', out_deg_coef[0], out_deg_coef[1]
# plt.plot(out_deg, [10 ** ((out_deg_coef[0] + 0.03) * x + out_deg_coef[1]) for x in deg_log], linestyle='-.', color='m', label=labl)


plt.title("Degree Distribution")
plt.ylabel("Count (log)")
plt.xlabel("Degree (log)")
plt.xscale('log')
plt.yscale('log')
plt.legend()
# plt.show()


# Task 1-3 ===============================================================

# exponential distribution
plt.plot(deg, 10 * 10 ** np.exp(-1 * np.log10(deg)),'r-', lw=3, alpha=0.9, label='exponential')
# log-normal distrobution
mu = +1
sigma = 0.8
x_coef = -1
y = 10000 * 10 ** (1 / (np.sqrt(2 * np.pi * np.power(sigma, 2)))) * \
    (np.power(np.e, -(np.power((x_coef * deg_log - mu), 2) / (2 * np.power(sigma, 2)))))
plt.plot(10 ** deg_log, y, label='log-normal');

plt.legend()
plt.show()
# print ("deg: \n\n", deg)
# print ("deg_cnt: \n\n", deg_cnt)
# print ("deg_log: \n\n", deg_log)
# print ("deg_cnt_log: \n\n", deg_cnt_log)