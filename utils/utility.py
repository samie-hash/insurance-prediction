# utility.py

from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import chi2, zscore
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, cols):
        """Select columns in a pandas dataframe object"""
        self.cols = cols

    def fit(self, X, y=None):
        return self 

    def transform(self, X, y=None):
        return pd.DataFrame(X.loc[:, self.cols].values, columns=self.cols)
        
class DataFrameImputer(BaseEstimator, TransformerMixin):

    def __init__(self):
        """Impute missing values

        Columns of dtype object are imputed with the most frequent value in the column

        Columns of other types are imputed with the mean of column
        """
        pass

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts(). index[0]
                               if X[c].dtype == np.dtype('O') else np.mean(X[c])
                               for c in X],
                              index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


class MyLabelEncoder(BaseEstimator, TransformerMixin):
    """Label encode categorical variables"""
        
    def __init__(self, cat_attribs):
        self.cat_attribs = cat_attribs
        self.enc = LabelEncoder()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for attr in self.cat_attribs:
            X[attr] = self.enc.fit_transform(X[attr])
        return X


def chi_square_test(X_attribs, y_attribs, data, alpha=.01):
    """
    Perform chi squared statistics

    Parameters
    __________

    X_attribs:  array-like of shape (n_samples, )
        The column names (independent variables)
    y_attribs:  str
        The column name of the dependent variable

    Returns
    _______

    None
    """
    print('H0: There is no relationship between the two categorical variables')
    print('H1: There is a relationship between the two categorical variables')
    print(f'Dependent variable: {y_attribs}')
    print('_'*80, '\n')
    chi_list = []
    for c in X_attribs:
        dataset_table = pd.crosstab(data[c], data[y_attribs])
        val = stats.chi2_contingency(dataset_table)
        chi_list.append(val)

    for t, c in zip(X_attribs, chi_list):
        fmt_str = 'Reject' if c[1] <= alpha else 'Fail to reject'
        print(f'{t:50}{c[1]:.5f}\t\t{fmt_str}')


def remove_symbols(x):
    x = re.sub(r',', '', x)
    x = re.sub(r'\s+', '', x)
    match = re.search('\d+(\.\d+)?', x)
    if match:
        return x[match.start(): match.end()]
    else:
        return np.NaN


def remove_outliers(data):
    num_attribs = [c for c in data.columns if data[c].dtype != np.dtype('O')]
    z_scores = zscore(data[num_attribs])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    new_data = data[filtered_entries]
    return new_data


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def calc_elbow_curve(X=None, K=(2, 10), show=True):
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(K[0], K[1])

    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)

        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                            'euclidean'), axis=1)) / X.shape[0])
        inertias.append(kmeanModel.inertia_)

        mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                       'euclidean'), axis=1)) / X.shape[0]
        mapping2[k] = kmeanModel.inertia_

    if show:
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method using Distortion')
        plt.show()
    else:
        return mapping1


def calc_silhouette_score(X=None, K=(2, 10), show=True):
    sil = []
    K = range(K[0], K[1])

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in K:
        model = KMeans(n_clusters=k).fit(X)
        labels = model.labels_
        sil.append(silhouette_score(X, labels, metric='euclidean'))

    if show:
        plt.plot(K, sil, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method using Distortion')
        plt.show()
    else:
        return sil


def plot_corr(corr):
    # plot correlation heatmap
 
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()
    return f

if __name__ == '__main__':
    data = pd.read_csv('Data/train.csv')
    cat_attribs = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage']
    imp = DataFrameSelector(cat_attribs)
    data = imp.fit_transform(data)
    print(data.head())