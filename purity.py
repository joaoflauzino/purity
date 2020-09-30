import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import seaborn as sns
from collections import Counter

# Function to visualize groups
def visualize(df, Model, n_cluster):
    s = sns.scatterplot(data=df, x="petal.length", y="petal.width", hue=Model)
    fig = s.get_figure()
    fig.savefig('image/kmeans_clusters_{}.png'.format(n_cluster))
    fig.clf()

# Function to apply kmeans
def apply_model(x, k):
    Model = KMeans(k)
    return Model.fit_predict(x)

# Function to calculate purity
def purity_metric(target, prediction):
   
    majority_sum = 0  
    for cl in set(prediction):
        labels_cl = Counter(l for l, c in zip(target, prediction) if c == cl)
        majority_sum += max(labels_cl.values())

    return majority_sum / len(prediction)


if __name__ == "__main__":
    
    df = pd.read_csv('dataset/iris.csv', sep = ",")
    x = df[['sepal.length','sepal.width','petal.length','petal.width']]
    
    # Running k-means for each cluster number
    for n_cluster in [2,3,4]:
        # Apply model
        Model = apply_model(x, n_cluster)
        # Saving results in png file
        visualize(df, Model, n_cluster)
        # Calculating purity
        purity_value = purity_metric(df['variety'], Model)
   
        print('The purity value with {} clusters is: '.format(str(n_cluster)), purity_value)







