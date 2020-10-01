import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import seaborn as sns
from collections import Counter
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-inputFile',
        default='iris.csv',
        type=str,
        help='name of file that will be worked')
    parser.add_argument(
        '-sep',
        default=',',
        help='column identifier')
    parser.add_argument(
        '-dec',
        default='.',
        help='decimal identifier')

    parser.add_argument(
        '-target',
        default='variety',
        type=str,
        help='target')

    return parser.parse_args()

# Function to visualize groups
def visualize(df, Model, n_cluster):
    s = sns.scatterplot(data=x, x=x.columns[0], y=x.columns[1], hue=Model)
    fig = s.get_figure()
    fig.savefig('image/kmeans_clusters_{}.png'.format(n_cluster))
    fig.clf()

# Function to apply kmeans
def apply_model(x, k):
    Model = KMeans(k)
    return Model.fit_predict(x)

# Function to calculate purity
def purity_metric(target, prediction):
   
    sum_highest_value = 0  
    for cl in set(prediction):
        cluster_labels = Counter(
            k for k, v in zip(target, prediction) 
            if v == cl
        )
        sum_highest_value += max(cluster_labels.values())

    return sum_highest_value / len(prediction)


if __name__ == "__main__":

    args = get_args()

    # Reading dataset
    df = pd.read_csv('dataset/' + args.inputFile, sep=args.sep, decimal=args.dec)
    
    # Selecting just numeric values
    x = df.select_dtypes(exclude=['object'])

    # Running k-means for each cluster number
    for n_cluster in [2,3,4]:
        # Apply model
        Model = apply_model(x, n_cluster)
        # Saving results in png file
        visualize(x, Model, n_cluster)
        # Calculating purity
        purity_value = purity_metric(df[args.target], Model)
   
        print('The purity value with {} clusters is: '.format(str(n_cluster)), purity_value)







