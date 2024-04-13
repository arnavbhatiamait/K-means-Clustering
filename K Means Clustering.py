# %% [markdown]
# K Means Clustering

# %% [markdown]
# Importing The Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# Importing The Data Set

# %%
df=pd.read_csv("Mall_Customers.csv")
df.head()

# %%
x=df.iloc[:,[3,4]].values
# ! here last variable is not a dependent variable but a feature
# * no dependent variable so no y in clustering
x

# %% [markdown]
# Using Elbow Method to find the Optimal Clustering Numbers

# %%
from sklearn.cluster import KMeans
wcss=[]
# ! wcss is cluster sum of squares in the data set
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init="k-means++",random_state=42)
    # ! 42 is luck and can be changed also
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.savefig("Elbow Method Graph.png")
plt.show()

# %%
# ! we will use  5 clusters as it is the optimal no of clusters

# %% [markdown]
# Training the KMeans Clustering Model on the data set

# %%
kmeans=KMeans(n_clusters=5,init="k-means++",random_state=42)
y_kmeans=kmeans.fit_predict(x)
#! fit predict not only fits but also predicts the output 
print(y_kmeans)

# %% [markdown]
# Visualizing the Clusters

# %%
# here x has 2 variables income at 0 and spending score at 1 and the format of selection is row, column 
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label="Cluster 1")
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='blue',label="Cluster 2")
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='green',label="Cluster 3")
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c='orange',label="Cluster 4")
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,c='cyan',label="Cluster 5")
# print(kmeans.cluster_centers_)
#! 0 column has x coordinates and y coordinates at 1 
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label="Centroid")
plt.title("Cluster of the no of customers")
plt.xlabel("Anuual Income (in rs)")
plt.ylabel("Spending Score (0-100)")
plt.legend(loc="upper right")
plt.savefig('Clusters of no of customers.png')
plt.show()


