import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv('network.csv')

#top 20
data = df.groupby(['node'], sort=True).count().reset_index().sort_values('users', ascending=False).head(20)['node'].values

plt.figure(3,figsize=(12,12))
G = nx.from_pandas_edgelist(df, 'node', 'users')
nx.draw_spring(G, with_labels=False, node_size=20, font_size=9, width=0.1)

labels={}
for node in G.nodes():
	if node in data:
		labels[node]=node

nx.draw_networkx_labels(G, pos=nx.spring_layout(G), labels=labels)
plt.show()

