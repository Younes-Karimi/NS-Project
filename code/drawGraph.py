#################################################################################################################
### Author: R. Padmavathi Iyer [NetID: RI928946]
### This file draws the CUSUM chart of the original data with significant changes in the background.
### The code is almost same to the extra_credit.py file except for some minor modifications.
#################################################################################################################


##################### Imports #####################
import numpy as np
import pandas as pd
import networkx as nx
import time
import datetime
from copy import deepcopy
import matplotlib.pyplot as plt
from decimal import Decimal



def graphSnapShots(dateArray, data):
	df = pd.DataFrame(data,columns=['c1','c2','c3'])
	dg = df.loc[df['c1']== dateArray]
	var = np.array(dg)
	edge_data = var[:,[1,2]]
	df = pd.DataFrame(edge_data,columns=['c1','c2'])
	dg = pd.DataFrame({'count' : df.groupby(['c1', 'c2']).size()}).reset_index()
	edge_weight_data = np.array(dg)
	edge_weight_data = np.delete(edge_weight_data, np.where(edge_weight_data[:,0]==edge_weight_data[:,1]), axis=0)
	G = nx.DiGraph()
	G.add_weighted_edges_from(list(map(tuple, edge_weight_data)))
	return G
	
	
def getStrongComponents(graph):
	g = sorted(nx.strongly_connected_components(graph),key=len,reverse=True)[0]
	return len(g)

	
def getCUSUM(comp):
	mean_val = np.mean(comp)
	S = [0]*(len(comp)+1)
	for i in list(range(1,(len(comp)+1))):
		S[i] = S[i-1] + (comp[i-1]-mean_val)
		
	return S

	

def changePointAnalysis(strongCompList, dayList, level):
	strongCompCopy = deepcopy(strongCompList)
	cusumList = getCUSUM(strongCompList)
	Sdiff0 = max(cusumList) - min(cusumList)
	Sdiff = []
	for i in range(1000):
		np.random.shuffle(strongCompCopy)
		boot_cusumList = getCUSUM(strongCompCopy)
		Sdiff.append(max(boot_cusumList) - min(boot_cusumList))
	Sdiff = np.array(Sdiff)	
	confidence = round(Decimal(len(np.where(Sdiff<Sdiff0)[0])/len(Sdiff)),3)
	print("\nConfidence === ", confidence)
	if (confidence >= 0.9):
		CUSUM_mse = np.argmax(list(map(abs, cusumList[1:])))
		print("Level === ", level)
		print("Old Component size === ", strongCompList[CUSUM_mse])
		print("New Component size === ", strongCompList[CUSUM_mse+1])
		print("Day === ", dayList[CUSUM_mse+1], "\n")
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

vfunc1 = np.vectorize(lambda t: t.date())
data[:,0] = vfunc1(data[:,0])
G_unique = np.unique(data[:,0]).tolist()


graphSnapList = list(map(graphSnapShots, G_unique, [data]*len(G_unique)))


strongCompList = list(map(getStrongComponents, graphSnapList))
dayList = np.linspace(1,len(strongCompList),len(strongCompList))
cusumList = getCUSUM(strongCompList)
### Added zero at the beginning just for the shading at the beginning of the plot before the first day is encountered.
### Similarly, added len(strongCompList) to mark the last day.
#shades = sorted([0,443,279,134,98,364,337,821,725,647,574,521,508,518,549,627,603,678,770,739,749,743,795,813,857,840,827,920,len(strongCompList)])
### The "shades" list contains the list of days when significant changes took place
shades = sorted([0,443,279,134,364,821,725,857,len(strongCompList)])
plt.xlim(0,len(strongCompList))
### Actual Evolution of strong components plot
#plt.plot(dayList, strongCompList, c='g')
### CUSUM chart of the evolution plot
plt.plot(dayList, cusumList[1:], c='k')

### Changing the background color of the plot according to the significant changes
ind=0
while ind<(len(shades)-1):
	plt.axvspan(shades[ind],shades[ind+1],facecolor='b',alpha=0.3)
	if (ind+1)<(len(shades)-1):
		plt.axvspan(shades[ind+1],shades[ind+2],facecolor='0.2',alpha=0.2)
	ind = ind + 2

plt.xlabel('Days')
plt.ylabel('CUSUM (Strongly connected components)')
plt.title('CUSUM chart with background colors depicting significant changes')
plt.show()
