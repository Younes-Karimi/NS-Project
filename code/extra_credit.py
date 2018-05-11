###################################################################################################################
### Author: R. Padmavathi Iyer [NetID: RI928946]
### This file contains the code for the extra-credit part of the project.
### This code generates the level, confidence and timing for each of the detected significant changes.
###################################################################################################################


#################### Package imports ####################
import numpy as np
import pandas as pd
import networkx as nx
import time
import datetime
from copy import deepcopy
import matplotlib.pyplot as plt
from decimal import Decimal


#################### This method generates the temporal graph snapshot for each day ####################
#################### Inputs ####################
### dateArray: the day of the snapshot
### data: dataset of email correspondences
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
	
	
#################### This method returns number of nodes in the largest strongly connected component for a given graph ####################
def getStrongComponents(graph):
	g = sorted(nx.strongly_connected_components(graph),key=len,reverse=True)[0]
	return len(g)


#################### This method returns the values for CUSUM chart for given data points ####################	
def getCUSUM(comp):
	mean_val = np.mean(comp)
	S = [0]*(len(comp)+1)
	for i in list(range(1,(len(comp)+1))):
		S[i] = S[i-1] + (comp[i-1]-mean_val)
		
	return S
	

#################### This method performs change-point analysis iteratively ####################	
def changePointAnalysis(strongCompList, dayList, level):
	strongCompCopy = deepcopy(strongCompList)
	cusumList = getCUSUM(strongCompList)
	Sdiff0 = max(cusumList) - min(cusumList)
	### Plots the CUSUM chart for initial ordering of data
	plt.plot(dayList, cusumList[1:], label='Initial Ordering')
	Sdiff = []
	### range(1000) for 1000 bootstrap samples
	for i in range(1000):
		np.random.shuffle(strongCompCopy)
		boot_cusumList = getCUSUM(strongCompCopy)
		Sdiff.append(max(boot_cusumList) - min(boot_cusumList))
		### Plots the CUSUM chart for each bootstrap sample
		plt.plot(dayList, boot_cusumList[1:])
	Sdiff = np.array(Sdiff)
	plt.xlabel('Days')
	plt.ylabel('CUSUM (Strongly connected components)')
	plt.title('Comparison of CUSUM charts of data in original order vs. bootstrap samples')
	plt.legend()
	plt.show()
	
	confidence = round(Decimal(len(np.where(Sdiff<Sdiff0)[0])/len(Sdiff)),3)
	print("\nConfidence === ", confidence)
	if (confidence >= 0.9): ### Means there is some significant change and we want to find it
		### Find the timing and other details of the change
		CUSUM_mse = np.argmax(list(map(abs, cusumList[1:])))
		print("Level === ", level)
		print("Old Component size === ", strongCompList[CUSUM_mse])
		print("New Component size === ", strongCompList[CUSUM_mse+1])
		print("Day === ", dayList[CUSUM_mse+1], "\n")
		### Perform change-point analysis recursively
		changePointAnalysis(strongCompList[:CUSUM_mse+1], dayList[:CUSUM_mse+1], level+1)
		changePointAnalysis(strongCompList[CUSUM_mse+1:], dayList[CUSUM_mse+1:], level+1)	
	return
	


# Importing dataset
orig_data = pd.read_csv("532projectdataset.txt", delimiter=" ", header=None)

# Converting into numpy array
data = np.array(orig_data)
data1 = deepcopy(data)

# Removing the rows in the dataset that correspond to weekends
data[:,0] = np.asarray(data[:,0], dtype='datetime64[s]')
data1[:,0] = np.asarray(data1[:,0], dtype='datetime64[s]')
vfunc = np.vectorize(lambda t: t.weekday())
data1[:,0] = vfunc(data1[:,0])
data = np.delete(data, np.where((data1[:,0]==5) | (data1[:,0]==6)), axis=0)
# Converting the date-time to days
vfunc1 = np.vectorize(lambda t: t.date())
data[:,0] = vfunc1(data[:,0])
G_unique = np.unique(data[:,0]).tolist()


### Gets all the graph snapshots in day order and stores them in a list
graphSnapList = list(map(graphSnapShots, G_unique, [data]*len(G_unique)))

print("\n****************** Change-Point Analysis for Largest Strongly Connected Components *******************************")
strongCompList = list(map(getStrongComponents, graphSnapList))
dayList = np.linspace(1,len(strongCompList),len(strongCompList))
changePointAnalysis(strongCompList, dayList, 1)

