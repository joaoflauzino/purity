import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.cluster import KMeans
from sklearn import datasets


def visualize_model_results(x, labels):
    plt.scatter(x.Petal_Length, x.Petal_width,c=colormap[labels],s=40)
    plt.title('Classification K-means ')
    plt.show()

if __name__ == "__main__":

    colormap=np.array(['Red','green','blue'])

    iris = datasets.load_iris()
    x = pd.DataFrame(iris.data)
    x.columns = ['Sepal_Length','Sepal_width','Petal_Length','Petal_width']
    y = pd.DataFrame(iris.target)
    y.columns = ['Targets']

    Model = KMeans(n_clusters=3)
    # Apply model
    Model.fit(x)
    
    #Visualize results
    visualize_model_results(x, Model.labels_)

    #print(Model.labels_)







