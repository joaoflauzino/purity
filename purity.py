import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
import seaborn as sns

# Function to visualize groups
def visualize(df, n_cluster):
    s = sns.scatterplot(data=df, x="petal.length", y="petal.width", hue=df['pred'].tolist())
    fig = s.get_figure()
    fig.savefig('image/kmeans_clusters_{}.png'.format(n_cluster))
    fig.clf()

# Function to apply kmeans
def apply_model(x, k):
    Model = KMeans(k)
    Model.fit(x)
    return Model

# Function to calculate purity
def purity(df):
    confusion_matrix = metrics.cluster.contingency_matrix(df['variety'], df['pred'])
    # Taking the sum of the ratings and dividing by the total
    return np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix) 

if __name__ == "__main__":
    
    df = pd.read_csv('dataset/iris.csv', sep = ",")
    x = df[['sepal.length','sepal.width','petal.length','petal.width']]
    
    # Running k-means for each cluster number
    for n_cluster in [2,3,4]:
        # Apply model
        Model = apply_model(x, n_cluster)

        import pdb; pdb.set_trace()

        # Saving result model in a column
        df['pred'] = Model.labels_

        # Saving results in png file
        visualize(df, n_cluster)
    
        # Calculating purity
        purity_value = purity(df)

        print('The purity value with {} clusters is: '.format(str(n_cluster)), purity_value)







