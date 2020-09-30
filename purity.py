import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from sklearn import datasets
import numpy as np
from sklearn import metrics
import seaborn as sns


def visualize(df):
    selectors = ['variety', 'pred']
    for element in range(len(selectors)):
        s = sns.scatterplot(data=df, x="petal.length", y="petal.width", hue=df[selectors[element]].tolist())
        fig = s.get_figure()
        fig.savefig('image/' + selectors[element] + '.png')
        fig.clf()

def apply_model(x):
    Model = KMeans(n_clusters=3)
    Model.fit(x)
    return Model

def select(x):
    if x == 'Setosa':
        return '0'
    if x == 'Versicolor':
        return '1'
    else:
        return '2'

def purity(df):
    df['pred'].apply(lambda x: select(x))
    confusion_matrix = metrics.cluster.contingency_matrix(df['variety'], df['pred'])
    return np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix) 

if __name__ == "__main__":
    
    df = pd.read_csv('dataset/iris.csv', sep = ",")
    x = df[['sepal.length','sepal.width','petal.length','petal.width']]
    y = df[['variety']]

    # Apply model
    Model = apply_model(x)

    # Save result in a column
    df['pred'] = Model.labels_

    # Save results in png file
    visualize(df)
    
    # Calculate purity
    purity_value = purity(df)

    print('The purity value is: ', purity_value)







