from keras.models import Sequential
from keras.layers import Flatten
from tensorflow.keras import backend as K
from sklearn.cluster import KMeans
import pandas as pd
import tensorflow as tf


def calc_classification_accuracy(bin, errors):
    tmp1 = errors[errors['true_card'] > bin[0]]
    tmp1 = tmp1[tmp1['true_card'] <= bin[1]]

    # Find which bin the estimated cardinality belongs to
    tmp2 = errors[errors['est_card'] > bin[0]]
    tmp2 = tmp2[tmp2['est_card'] <= bin[1]]

    # For each of the indices in tmp1, find if the same index exists in tmp2
    # If it does, then the estimated cardinality is in the same bin as the true cardinality
    # If it doesn't, then the estimated cardinality is in a different bin as the true cardinality
    # Calculate the percentage of estimated cardinalities that are in the same bin as the true cardinality

    try:
        acc = len(tmp1[tmp1.index.isin(tmp2.index)]) / len(tmp1)
    except:
        acc = -1
    return acc

with tf.device("GPU:0"):
    def create_flatten_mlp(shape):
        """
        Creates Flattening and possible adds an MLP Layer for X, A, E separately
        :param shape: [input_shape] of the input that needs to be passed by the layers
        :return: model
        """

        total = 1
        for s in shape:
            total *= s
        model = Sequential()
        model.add(Flatten(input_shape=shape))
        
        # return our model
        return model

    def normalize_MINMAX(y):
        """ Normalization with MIN and MAX value """
        if 'MIN' not in globals():
            global MIN, MAX
            MIN = min(y)
            MAX = max(y)
        y = (y - MIN) / (MAX - MIN)
        return y

    def denormalize_MINMAX(y):
        """ Denormalization with MIN and MAX value """
        y = (y * (MAX - MIN)) + MIN
        return y

    def inverse_transform_minmax(y):
        """ Returns the inverse transform of the number """
        y = denormalize_MINMAX(y)
        y = K.exp(y)
        return y

    def q_loss(y_true, y_pred):
        """ Calculation of q_loss with the original values """
        y_true = inverse_transform_minmax(y_true)
        y_pred = inverse_transform_minmax(y_pred)
        return K.maximum(y_true, y_pred) / K.minimum(y_true, y_pred)

def apply_clustering(training_set, testing_set, no_clusters, query_type, joins):
    ## Apply Clustering and Store  
    # Read training set
    if query_type == 'star':
        train_df = pd.read_csv(training_set, sep=':', header=None)
        testing_df = pd.read_csv(testing_set, sep=':', header=None)
    else:
        train_df = pd.read_csv(training_set, sep=',', header=None)
        testing_df = pd.read_csv(testing_set, sep=',', header=None)

        # Keep column 1 to a new dataframe
    train_df_cards = train_df[1]
    testing_df_cards = testing_df[1]

    if query_type == 'star':
        # Separate column 0 on comma into two new columns
        train_df = train_df[0].str.split(',', expand=True)
        testing_df = testing_df[0].str.split(',', expand=True)
    else:
        # Separate column 0 on comma into three new columns
        train_df = train_df[0].str.split('-', expand=True)
        testing_df = testing_df[0].str.split('-', expand=True)

    if query_type == 'star':
        # Separate all columns on '-' and add them to the dataframe
        for i in range(len(train_df.columns)):
            train_df = train_df.join(train_df[i].str.split('-', expand=True).add_prefix('col' + str(i)))
        for i in range(len(testing_df.columns)):
            testing_df = testing_df.join(testing_df[i].str.split('-', expand=True).add_prefix('col' + str(i)))

        # Drop some columns
        train_df = train_df.drop(columns=range(joins))
        testing_df = testing_df.drop(columns=range(joins))

        # Rename columns sequentially
        train_df.columns = range(train_df.shape[1])
        testing_df.columns = range(testing_df.shape[1])


        # Convert everything to float except if it is *
        train_df = train_df.applymap(lambda x: float(x) if x != '*' else x)
        testing_df = testing_df.applymap(lambda x: float(x) if x != '*' else x)

        ids = list()
        for index in range(joins):
                if index == 0:
                    ids.append(2)
                else:
                    ids.append(2 + (3 * (index)))
        kmeans = KMeans(n_clusters=no_clusters, max_iter=10000, algorithm='elkan', n_init=50, init='k-means++').fit(train_df[ids])
    else:
        ids = list()
        for index in range(joins):
                if index == 0:
                    ids.append(2)
                else:
                    ids.append(3 + (3 * (index)))
        kmeans = KMeans(n_clusters=no_clusters, max_iter=10000, algorithm='elkan', n_init=50, init='k-means++').fit(train_df[ids])

    # Add cluster labels to the dataframe
    train_df['cluster'] = kmeans.labels_

    # Convert columns to float except if it is *
    if query_type == 'star':
        train_df = train_df.applymap(lambda x: float(x) if x != '*' else x)
        testing_df = testing_df.applymap(lambda x: float(x) if x != '*' else x)
        # Assign cluster labels to the testing set
        testing_df['cluster'] = kmeans.predict(testing_df[ids])
    else:
        train_df = train_df.applymap(lambda x: float(x) if x != '*' else x)
        testing_df = testing_df.applymap(lambda x: float(x) if x != '*' else x)
        # Assign cluster labels to the testing set
        testing_df['cluster'] = kmeans.predict(testing_df[ids])

    if query_type == 'star':
        # Keep cluster column to a separate dataframe
        clusters_train = train_df['cluster']
        train_df = train_df.drop(columns=['cluster'])
        clusters_testing = testing_df['cluster']
        testing_df = testing_df.drop(columns=['cluster'])

        # Per three columns, combine to one 
        train_df_new = pd.DataFrame()
        testing_df_new = pd.DataFrame()
        for i in range(joins):
            train_df[i + (2*i)] = train_df[i + (2*i)].astype(str)
            train_df[i+1 + (2*i)] = train_df[i+1 + (2*i)].astype(str)
            train_df[i+2 + (2*i)] = train_df[i+2 + (2*i)].astype(str)
            testing_df[i + (2*i)] = testing_df[i + (2*i)].astype(str)
            testing_df[i+1 + (2*i)] = testing_df[i+1 + (2*i)].astype(str)
            testing_df[i+2 + (2*i)] = testing_df[i+2 + (2*i)].astype(str)
            if i == 0:
                train_df_new = train_df[i + (2*i)] + '-' + train_df[i+1 + (2*i)] + '-' + train_df[i+2 + (2*i)] + ','
                testing_df_new = testing_df[i + (2*i)] + '-' + testing_df[i+1 + (2*i)] + '-' + testing_df[i+2 + (2*i)] + ','
            else:
                train_df_new = train_df_new + train_df[i + (2*i)] + '-' + train_df[i+1 + (2*i)] + '-' + train_df[i+2 + (2*i)] + ','
                testing_df_new = testing_df_new + testing_df[i + (2*i)] + '-' + testing_df[i+1 + (2*i)] + '-' + testing_df[i+2 + (2*i)] + ','
        train_df_new = train_df_new.to_frame()
        testing_df_new = testing_df_new.to_frame()

        # Drop the last comma
        train_df_new[0] = train_df_new[0].str[:-1]
        testing_df_new[0] = testing_df_new[0].str[:-1]
        train_df = train_df_new
        testing_df = testing_df_new

        # Add cluster column
        train_df['cluster'] = clusters_train
        testing_df['cluster'] = clusters_testing

        # Add cards
        train_df = train_df.join(train_df_cards)
        testing_df = testing_df.join(testing_df_cards)
    else:
        clusters_train = train_df[train_df.columns[-1]]
        train_df = train_df.drop(columns=[train_df.columns[-1]])

        clusters_testing = testing_df[testing_df.columns[-1]]
        testing_df = testing_df.drop(columns=[testing_df.columns[-1]])

        # Merge all columns with separator '-'
        train_df = train_df[train_df.columns].apply(lambda x: '-'.join(x.dropna().astype(str)), axis=1)
        testing_df = testing_df[testing_df.columns].apply(lambda x: '-'.join(x.dropna().astype(str)), axis=1)

        train_df = train_df.to_frame()
        testing_df = testing_df.to_frame()

        # Add clusters
        train_df['cluster'] = clusters_train
        testing_df['cluster'] = clusters_testing

        # Add cards
        train_df = train_df.join(train_df_cards)
        testing_df = testing_df.join(testing_df_cards)

    # Store the data per cluster in a separate file
    for i in range(no_clusters):
        new_df = train_df[train_df['cluster'] == i]
        new_df = new_df.drop(columns=['cluster'])
        if query_type == 'star':
            new_df.to_csv('cluster_' + str(i) + '.txt', sep=':', header=None, index=False)
        else:
            new_df.to_csv('cluster_' + str(i) + '.txt', sep=',', header=None, index=False)

    # Store the data per cluster in a separate file
    for i in range(no_clusters):
        new_df = testing_df[testing_df['cluster'] == i]
        new_df = new_df.drop(columns=['cluster'])
        if query_type == 'star':
            new_df.to_csv('cluster_' + str(i) + '_test.txt', sep=':', header=None, index=False)
        else:
            new_df.to_csv('cluster_' + str(i) + '_test.txt', sep=',', header=None, index=False)