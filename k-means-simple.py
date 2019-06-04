import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import table
from mpl_toolkits.mplot3d import Axes3D
import pymysql
import sqlite3
import pandas as pd
from sklearn.cluster import KMeans

f = open('sql.txt','r')
sql_script = f.read()
data = pd.read_sql(sql_script, database_conn)
columns = ['', '', '']
data = data.loc[:, columns]
X = data.values

#fit clusters
kmeans = KMeans(n_clusters=4, init='random', random_state=0, verbose=1).fit(X)
labels = kmeans.labels_

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2],
           c=labels.astype(np.float), edgecolor='k')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.view_init(30, 20)
plt.show()


