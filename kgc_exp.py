import complex_reader_kgc_se
import pandas as pd
import numpy as np
import os
import sys
from keras.layers import concatenate, multiply, Dense
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import functions as func
import time

import warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

################# EXPERIMENT CONFIGURATION ##################
datasets_path = 'data/'
query_type = sys.argv[1]
query_join = int(sys.argv[2])
dataset_name = sys.argv[3]
no_clusters = int(sys.argv[4])
epochs = int(sys.argv[5])
batch_size = int(sys.argv[6])

print ("The query type is: " + query_type)
print ("The query join is: " + str(query_join))
print ("The dataset name is: " + dataset_name)
print ("The number of clusters is: " + str(no_clusters))
print ("The number of epochs is: " + str(epochs))
print ("The batch size is: " + str(batch_size))

if dataset_name == 'yelp':
    d = 9213125
    b = 4
    n = query_join + 1
    e = query_join

if dataset_name == 'swdf':
    d = 106436
    b = 65
    n = query_join + 1
    e = query_join

if dataset_name == 'wiki':
    d = 739139277
    b = 8605
    n = query_join + 1
    e = query_join

if dataset_name == 'lubm':
    d = 545166
    b = 13
    n = query_join + 1
    e = query_join

if query_type == 'chain':
    separator = ','
if query_type == 'star':
    separator = ':'

training_set = (datasets_path + query_type  + '/' + query_type + '_' + str(query_join) + '_' + dataset_name + '_sim_80_20.csv')
testing_set =(datasets_path + query_type + '/eval_' + query_type + '_' + str(query_join) +'_' + dataset_name + '_sim_80_20.csv')
print('The training set is: ' + training_set)
print('The testing set is: ' + testing_set)

################# APPLY CLUSTERING ##################
func.apply_clustering(training_set, testing_set, no_clusters, query_type, query_join)
#####################################################


################# DEFINE THE MODEL ##################
# 2 matrices, binary encoding
binary_d = int(np.ceil(np.log2(d))) + 1
binary_b = int(np.ceil(np.log2(b))) + 1

# create the MLP models
NNs = []
for i in range(query_join):
    NNs.append(func.create_flatten_mlp((n, binary_d)))
    NNs.append(func.create_flatten_mlp((e, binary_b)))
    NNs.append(func.create_flatten_mlp((e, n, n)))

combinedInputs = []
for i in range(query_join):
    combinedInputs.append(concatenate([NNs[3*i].output, NNs[3*i+1].output, NNs[3*i+2].output]))

ss = []
for i in range(query_join):
    ss.append(func.create_flatten_mlp([1]))
#######################################################


####### NETWORK CONFIGURATION #######
# Create neural networks with the defined inputs
Zs = []
for i in range(query_join):
    Zs.append(Dense(256, activation="relu")(combinedInputs[i]))
    Zs[i] = Dense(256, activation="relu")(Zs[i])

# Multiply
for i in range(query_join):
    Zs[i] = multiply([Zs[i], ss[i].output])

# Combine to a single output layer
for i in range(query_join):
    if i == 0:
        z = Zs[i]
    else:
        z = concatenate([z, Zs[i]])

# Apply a regression output layer
z = Dense(256, activation="relu")(z)
z = Dense(256, activation="relu")(z)
z = Dense(1, activation="sigmoid")(z)

# Create the model
inputs = []
for i in range(query_join):
    inputs.append(NNs[3*i].input)
    inputs.append(NNs[3*i+1].input)
    inputs.append(NNs[3*i+2].input)

for i in range(query_join):
    inputs.append(ss[i].input)
#####################################


####### TRAINING #######
# Repeat for each temporary file
total_training_time = 0
for j in range(no_clusters):
    training_set = 'cluster_' + str(j) + '.txt'
    model_name = 'kgc_exp_' + str(j) + '.h5'

    X_train, A_train, E_train, s_train, y_train, encoding_time = complex_reader_kgc_se.read_queries(training_set, d, b, n, e, query_join, query_type)
    y_train = np.log(y_train)
    y_train = func.normalize_MINMAX(y_train)

    model = Model(inputs=inputs, outputs=z)

    Xs = []
    As = []
    Es = []
    ss = []

    for i in range(query_join):
        curr_X = []
        curr_A = []
        curr_E = []
        curr_s = []
        for k in range(len(X_train)):
            curr_X.append(X_train[k][i])
            curr_A.append(A_train[k][i])
            curr_E.append(E_train[k][i])
            curr_s.append(s_train[k][i])
        Xs.append(curr_X)
        As.append(curr_A)
        Es.append(curr_E)
        ss.append(curr_s)

    # Convert to float numpy arrays
    for i in range(query_join):
        Xs[i] = np.array(Xs[i], dtype=np.float32)
        As[i] = np.array(As[i], dtype=np.float32)
        Es[i] = np.array(Es[i], dtype=np.float32)
        ss[i] = np.array(ss[i], dtype=np.float32)
    
    # Train the model
    print("[INFO] training model " + str(j) + "...")
    optimizer = Adam(learning_rate = 1e-4, decay = 1e-4 / 100)
    #model.compile(optimizer, loss='mse')
    model.compile(optimizer, loss=func.q_loss)

    all_data = []
    for i in range(query_join):
        all_data.append(Xs[i])
        all_data.append(Es[i])
        all_data.append(As[i])

    for i in range(query_join):
        all_data.append(ss[i])

    # Count execution time
    start_time = time.time()
    trained_model = model.fit(all_data, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    end_time = time.time()
    total_training_time += (end_time - start_time) * 1000

    # Store the model
    save_path = 'models/'+dataset_name+"/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.save(save_path + model_name)

print('Finished!')
##########################


################# EVALUATE THE MODEL ##################
all_errors = pd.DataFrame()
total_evaluation_time = 0

# For eaech evaluation file read the queries and predict the cardinality
for j in range(no_clusters):
    testing_set = 'cluster_' + str(j) + '_test.txt'
    model_name = 'kgc_exp_' + str(j) + '.h5'

    errors = pd.read_csv(testing_set, sep=separator, header=None)
    errors = errors.rename(columns={0: "est", 1: "true_card"})

    # Load the model
    save_path = 'models/'+dataset_name+"/"
    nn_model = load_model(save_path + model_name, custom_objects={"q_loss": func.q_loss})
    
    # Read the queries
    X_test, A_test, E_test, s_test, y_test, encoding_time = complex_reader_kgc_se.read_queries(testing_set, d, b, n, e, query_join, query_type)

    # Predict on test set
    Xs_test = []
    As_test = []
    Es_test = []
    ss_test = []

    for i in range(query_join):
        curr_X = []
        curr_A = []
        curr_E = []
        curr_s = []
        for k in range(len(X_test)):
            curr_X.append(X_test[k][i])
            curr_A.append(A_test[k][i])
            curr_E.append(E_test[k][i])
            curr_s.append(s_test[k][i])
        Xs_test.append(curr_X)
        As_test.append(curr_A)
        Es_test.append(curr_E)
        ss_test.append(curr_s)

    # Convert to float numpy arrays
    for i in range(query_join):
        Xs_test[i] = np.array(Xs_test[i], dtype=np.float32)
        As_test[i] = np.array(As_test[i], dtype=np.float32)
        Es_test[i] = np.array(Es_test[i], dtype=np.float32)
        ss_test[i] = np.array(ss_test[i], dtype=np.float32)

    all_data_test = []
    for i in range(query_join):
        all_data_test.append(Xs_test[i])
        all_data_test.append(Es_test[i])
        all_data_test.append(As_test[i])

    for i in range(query_join):
        all_data_test.append(ss_test[i])

    start_time = time.time()
    y_pred = nn_model.predict(all_data_test)
    end_time = time.time()
    total_evaluation_time += (end_time - start_time) * 1000
    y_pred = func.denormalize_MINMAX(y_pred)
    y_pred = np.exp(y_pred)

    errors['est_card'] = list(y_pred.flatten())
    errors['true_card'] = list(y_test)
    
    errors['q1'] = errors['est_card'] / errors['true_card']
    errors['q2'] = errors['true_card'] / errors['est_card']
    errors['q_error'] =  errors[['q1', 'q2']].max(axis=1)

    # Concatenate all errors to the all_errors dataframe
    all_errors = pd.concat([all_errors, errors], ignore_index=True)
#######################################################


################# WRITE THE RESULTS ###################
output_file = open('results/'+dataset_name+'/kgc_exp_'+query_type+'_'+str(query_join)+'_'+str(no_clusters)+'_'+str(epochs)+'_'+str(batch_size)+'.txt', 'w')
output_file.write('Training time: ' + str(total_training_time) + '\n')
output_file.write('Evaluation time: ' + str(total_evaluation_time) + '\n\n')

all_errors['true_card_bin'] = pd.qcut(all_errors['true_card'], 5, labels=False, duplicates='drop')

results_incr = pd.DataFrame()
results_equi = pd.DataFrame()
err = all_errors['true_card'][all_errors['true_card'] > 0]
indices_1 = all_errors['true_card'][all_errors['true_card'] > 0][err <= 10].index
results_incr = results_incr.append({'Bin': '1 to 10  (' + str(len(indices_1)) + ' elements)', 'avg q-error': all_errors['q_error'][indices_1].mean()}, ignore_index=True)
err = all_errors['true_card'][all_errors['true_card'] > 10]
indices_10 = all_errors['true_card'][all_errors['true_card'] > 10][err <= 100].index
results_incr = results_incr.append({'Bin': '10 to 100  (' + str(len(indices_10)) + ' elements)', 'avg q-error': all_errors['q_error'][indices_10].mean()}, ignore_index=True)
err = all_errors['true_card'][all_errors['true_card'] > 100]
indices_100 = all_errors['true_card'][all_errors['true_card'] > 100][err <= 1000].index
results_incr = results_incr.append({'Bin': '100 to 1000  (' + str(len(indices_100)) + ' elements)', 'avg q-error': all_errors['q_error'][indices_100].mean()}, ignore_index=True)
err = all_errors['true_card'][all_errors['true_card'] > 1000]
indices_1000 = all_errors['true_card'][all_errors['true_card'] > 1000][err <= 10000].index
results_incr = results_incr.append({'Bin': '1000 to 10000  (' + str(len(indices_1000)) + ' elements)', 'avg q-error': all_errors['q_error'][indices_1000].mean()}, ignore_index=True)
indices_10000 = all_errors['true_card'][all_errors['true_card'] > 10000][err <= 100000].index
results_incr = results_incr.append({'Bin': '10000 to 100000  (' + str(len(indices_10000)) + ' elements)', 'avg q-error': all_errors['q_error'][indices_10000].mean()}, ignore_index=True)
results_incr = results_incr.append({'Bin': 'All errors  (' + str(len(all_errors)) + ' elements)', 'avg q-error': all_errors['q_error'].mean()}, ignore_index=True)
results_incr = results_incr.set_index('Bin')

bin1 = all_errors[all_errors['true_card_bin'] == 0]
bin2 = all_errors[all_errors['true_card_bin'] == 1]
bin3 = all_errors[all_errors['true_card_bin'] == 2]
bin4 = all_errors[all_errors['true_card_bin'] == 3]
bin5 = all_errors[all_errors['true_card_bin'] == 4]

results_equi = results_equi.append({'Bin': str(bin1['true_card'].min()) + ' to ' + str(bin1['true_card'].max()) + '  (' + str(len(bin1)) + ' elements)', 'avg q-error': bin1['q_error'].mean()}, ignore_index=True)
results_equi = results_equi.append({'Bin': str(bin2['true_card'].min()) + ' to ' + str(bin2['true_card'].max()) + '  (' + str(len(bin2)) + ' elements)', 'avg q-error': bin2['q_error'].mean()}, ignore_index=True)
results_equi = results_equi.append({'Bin': str(bin3['true_card'].min()) + ' to ' + str(bin3['true_card'].max()) + '  (' + str(len(bin3)) + ' elements)', 'avg q-error': bin3['q_error'].mean()}, ignore_index=True)
results_equi = results_equi.append({'Bin': str(bin4['true_card'].min()) + ' to ' + str(bin4['true_card'].max()) + '  (' + str(len(bin4)) + ' elements)', 'avg q-error': bin4['q_error'].mean()}, ignore_index=True)
results_equi = results_equi.append({'Bin': str(bin5['true_card'].min()) + ' to ' + str(bin5['true_card'].max()) + '  (' + str(len(bin5)) + ' elements)', 'avg q-error': bin5['q_error'].mean()}, ignore_index=True)
results_equi = results_equi.append({'Bin': 'All errors  (' + str(len(all_errors)) + ' elements)', 'avg q-error': all_errors['q_error'].mean()}, ignore_index=True)
results_equi = results_equi.set_index('Bin')

err = all_errors['true_card'][all_errors['true_card'] > 0]
indices_1 = all_errors[all_errors['true_card'] > 0][err <= 10]
results_incr.loc['1 to 10  (' + str(len(indices_1)) + ' elements)', 'median q-error'] = indices_1['q_error'].median()
err = all_errors['true_card'][all_errors['true_card'] > 10]
indices_10 = all_errors[all_errors['true_card'] > 10][err <= 100]
results_incr.loc['10 to 100  (' + str(len(indices_10)) + ' elements)', 'median q-error'] = indices_10['q_error'].median()
err = all_errors['true_card'][all_errors['true_card'] > 100]
indices_100 = all_errors[all_errors['true_card'] > 100][err <= 1000]
results_incr.loc['100 to 1000  (' + str(len(indices_100)) + ' elements)', 'median q-error'] = indices_100['q_error'].median()
err = all_errors['true_card'][all_errors['true_card'] > 1000]
indices_1000 = all_errors[all_errors['true_card'] > 1000][err <= 10000]
results_incr.loc['1000 to 10000  (' + str(len(indices_1000)) + ' elements)', 'median q-error'] = indices_1000['q_error'].median()
indices_10000 = all_errors[all_errors['true_card'] > 10000][err <= 100000]
results_incr.loc['10000 to 100000  (' + str(len(indices_10000)) + ' elements)', 'median q-error'] = indices_10000['q_error'].median()
results_incr.loc['All errors  (' + str(len(all_errors)) + ' elements)', 'median q-error'] = all_errors['q_error'].median()

results_equi.loc[str(bin1['true_card'].min()) + ' to ' + str(bin1['true_card'].max()) + '  (' + str(len(bin1)) + ' elements)', 'median q-error'] = bin1['q_error'].median()
results_equi.loc[str(bin2['true_card'].min()) + ' to ' + str(bin2['true_card'].max()) + '  (' + str(len(bin2)) + ' elements)', 'median q-error'] = bin2['q_error'].median()
results_equi.loc[str(bin3['true_card'].min()) + ' to ' + str(bin3['true_card'].max()) + '  (' + str(len(bin3)) + ' elements)', 'median q-error'] = bin3['q_error'].median()
results_equi.loc[str(bin4['true_card'].min()) + ' to ' + str(bin4['true_card'].max()) + '  (' + str(len(bin4)) + ' elements)', 'median q-error'] = bin4['q_error'].median()
results_equi.loc[str(bin5['true_card'].min()) + ' to ' + str(bin5['true_card'].max()) + '  (' + str(len(bin5)) + ' elements)', 'median q-error'] = bin5['q_error'].median()
results_equi.loc['All errors  (' + str(len(all_errors)) + ' elements)', 'median q-error'] = all_errors['q_error'].median()

bins1 = [(0,10), (10, 100), (100, 1000), (1000, 10000), (10000, 100000)]
results_incr.loc['1 to 10  (' + str(len(indices_1)) + ' elements)', 'classification accuracy'] = func.calc_classification_accuracy(bins1[0], indices_1)
results_incr.loc['10 to 100  (' + str(len(indices_10)) + ' elements)', 'classification accuracy'] = func.calc_classification_accuracy(bins1[1], indices_10)
results_incr.loc['100 to 1000  (' + str(len(indices_100)) + ' elements)', 'classification accuracy'] = func.calc_classification_accuracy(bins1[2], indices_100)
results_incr.loc['1000 to 10000  (' + str(len(indices_1000)) + ' elements)', 'classification accuracy'] = func.calc_classification_accuracy(bins1[3], indices_1000)
results_incr.loc['10000 to 100000  (' + str(len(indices_10000)) + ' elements)', 'classification accuracy'] = func.calc_classification_accuracy(bins1[4], indices_10000)

bins2 = [(bin1['true_card'].min(), bin1['true_card'].max()), (bin2['true_card'].min(), bin2['true_card'].max()), (bin3['true_card'].min(), bin3['true_card'].max()), (bin4['true_card'].min(), bin4['true_card'].max()), (bin5['true_card'].min(), bin5['true_card'].max())]
results_equi.loc[str(bin1['true_card'].min()) + ' to ' + str(bin1['true_card'].max()) + '  (' + str(len(bin1)) + ' elements)', 'classification accuracy'] = func.calc_classification_accuracy(bins2[0], bin1)
results_equi.loc[str(bin2['true_card'].min()) + ' to ' + str(bin2['true_card'].max()) + '  (' + str(len(bin2)) + ' elements)', 'classification accuracy'] = func.calc_classification_accuracy(bins2[1], bin2)
results_equi.loc[str(bin3['true_card'].min()) + ' to ' + str(bin3['true_card'].max()) + '  (' + str(len(bin3)) + ' elements)', 'classification accuracy'] = func.calc_classification_accuracy(bins2[2], bin3)
results_equi.loc[str(bin4['true_card'].min()) + ' to ' + str(bin4['true_card'].max()) + '  (' + str(len(bin4)) + ' elements)', 'classification accuracy'] = func.calc_classification_accuracy(bins2[3], bin4)
results_equi.loc[str(bin5['true_card'].min()) + ' to ' + str(bin5['true_card'].max()) + '  (' + str(len(bin5)) + ' elements)', 'classification accuracy'] = func.calc_classification_accuracy(bins2[4], bin5)

## 95th Percentile Errors
results_incr.loc['1 to 10  (' + str(len(indices_1)) + ' elements)', '95th percentile q-error'] = indices_1.quantile(0.95)[4]
results_incr.loc['10 to 100  (' + str(len(indices_10)) + ' elements)', '95th percentile q-error'] = indices_10.quantile(0.95)[4]
results_incr.loc['100 to 1000  (' + str(len(indices_100)) + ' elements)', '95th percentile q-error'] = indices_100.quantile(0.95)[4]
results_incr.loc['1000 to 10000  (' + str(len(indices_1000)) + ' elements)', '95th percentile q-error'] = indices_1000.quantile(0.95)[4]
results_incr.loc['10000 to 100000  (' + str(len(indices_10000)) + ' elements)', '95th percentile q-error'] = indices_10000.quantile(0.95)[4]
results_incr.loc['All errors  (' + str(len(all_errors)) + ' elements)', '95th percentile q-error'] = all_errors.quantile(0.95)[4]

results_equi.loc[str(bin1['true_card'].min()) + ' to ' + str(bin1['true_card'].max()) + '  (' + str(len(bin1)) + ' elements)', '95th percentile q-error'] = bin1.quantile(0.95)[4]
results_equi.loc[str(bin2['true_card'].min()) + ' to ' + str(bin2['true_card'].max()) + '  (' + str(len(bin2)) + ' elements)', '95th percentile q-error'] = bin2.quantile(0.95)[4]
results_equi.loc[str(bin3['true_card'].min()) + ' to ' + str(bin3['true_card'].max()) + '  (' + str(len(bin3)) + ' elements)', '95th percentile q-error'] = bin3.quantile(0.95)[4]
results_equi.loc[str(bin4['true_card'].min()) + ' to ' + str(bin4['true_card'].max()) + '  (' + str(len(bin4)) + ' elements)', '95th percentile q-error'] = bin4.quantile(0.95)[4]
results_equi.loc[str(bin5['true_card'].min()) + ' to ' + str(bin5['true_card'].max()) + '  (' + str(len(bin5)) + ' elements)', '95th percentile q-error'] = bin5.quantile(0.95)[4]
results_equi.loc['All errors  (' + str(len(all_errors)) + ' elements)', '95th percentile q-error'] = all_errors.quantile(0.95)[4]

indices_1_95 = indices_1.copy()
indices_10_95 = indices_10.copy()
indices_100_95 = indices_100.copy()
indices_1000_95 = indices_1000.copy()
indices_10000_95 = indices_10000.copy()
bin1_95 = bin1.copy()
bin2_95 = bin2.copy()
bin3_95 = bin3.copy()
bin4_95 = bin4.copy()
bin5_95 = bin5.copy()

indices_1_95 = indices_1_95.sort_values(by=['q_error'], ascending=False)
bin1_95 = bin1_95.sort_values(by=['q_error'], ascending=False)
for i in range(0, int(0.05*len(indices_1_95))):
    indices_1_95 = indices_1_95.iloc[1:]
    bin1_95 = bin1_95.iloc[1:]

indices_10_95 = indices_10_95.sort_values(by=['q_error'], ascending=False)
bin2_95 = bin2_95.sort_values(by=['q_error'], ascending=False)
for i in range(0, int(0.05*len(indices_10_95))):
    indices_10_95 = indices_10_95.iloc[1:]
    bin2_95 = bin2_95.iloc[1:]

indices_100_95 = indices_100_95.sort_values(by=['q_error'], ascending=False)
bin3_95 = bin3_95.sort_values(by=['q_error'], ascending=False)
for i in range(0, int(0.05*len(indices_100_95))):
    indices_100_95 = indices_100_95.iloc[1:]
    bin3_95 = bin3_95.iloc[1:]

indices_1000_95 = indices_1000_95.sort_values(by=['q_error'], ascending=False)
bin4_95 = bin4_95.sort_values(by=['q_error'], ascending=False)
for i in range(0, int(0.05*len(indices_1000_95))):
    indices_1000_95 = indices_1000_95.iloc[1:]
    bin4_95 = bin4_95.iloc[1:]

indices_10000_95 = indices_10000_95.sort_values(by=['q_error'], ascending=False)
bin5_95 = bin5_95.sort_values(by=['q_error'], ascending=False)
for i in range(0, int(0.05*len(indices_10000_95))):
    indices_10000_95 = indices_10000_95.iloc[1:]
    bin5_95 = bin5_95.iloc[1:]

results_incr.loc['1 to 10  (' + str(len(indices_1)) + ' elements)', '95th percentile accuracy'] = func.calc_classification_accuracy(bins1[0], indices_1_95)
results_incr.loc['10 to 100  (' + str(len(indices_10)) + ' elements)', '95th percentile accuracy'] = func.calc_classification_accuracy(bins1[1], indices_10_95)
results_incr.loc['100 to 1000  (' + str(len(indices_100)) + ' elements)', '95th percentile accuracy'] = func.calc_classification_accuracy(bins1[2], indices_100_95)
results_incr.loc['1000 to 10000  (' + str(len(indices_1000)) + ' elements)', '95th percentile accuracy'] = func.calc_classification_accuracy(bins1[3], indices_1000_95)
results_incr.loc['10000 to 100000  (' + str(len(indices_10000)) + ' elements)', '95th percentile accuracy'] = func.calc_classification_accuracy(bins1[4], indices_10000_95)

results_equi.loc[str(bin1['true_card'].min()) + ' to ' + str(bin1['true_card'].max()) + '  (' + str(len(bin1)) + ' elements)', '95th percentile accuracy'] = func.calc_classification_accuracy(bins2[0], bin1_95)
results_equi.loc[str(bin2['true_card'].min()) + ' to ' + str(bin2['true_card'].max()) + '  (' + str(len(bin2)) + ' elements)', '95th percentile accuracy'] = func.calc_classification_accuracy(bins2[1], bin2_95)
results_equi.loc[str(bin3['true_card'].min()) + ' to ' + str(bin3['true_card'].max()) + '  (' + str(len(bin3)) + ' elements)', '95th percentile accuracy'] = func.calc_classification_accuracy(bins2[2], bin3_95)
results_equi.loc[str(bin4['true_card'].min()) + ' to ' + str(bin4['true_card'].max()) + '  (' + str(len(bin4)) + ' elements)', '95th percentile accuracy'] = func.calc_classification_accuracy(bins2[3], bin4_95)
results_equi.loc[str(bin5['true_card'].min()) + ' to ' + str(bin5['true_card'].max()) + '  (' + str(len(bin5)) + ' elements)', '95th percentile accuracy'] = func.calc_classification_accuracy(bins2[4], bin5_95)

## 75th Percentile Errors
results_incr.loc['1 to 10  (' + str(len(indices_1)) + ' elements)', '75th percentile q-error'] = indices_1.quantile(0.75)[4]
results_incr.loc['10 to 100  (' + str(len(indices_10)) + ' elements)', '75th percentile q-error'] = indices_10.quantile(0.75)[4]
results_incr.loc['100 to 1000  (' + str(len(indices_100)) + ' elements)', '75th percentile q-error'] = indices_100.quantile(0.75)[4]
results_incr.loc['1000 to 10000  (' + str(len(indices_1000)) + ' elements)', '75th percentile q-error'] = indices_1000.quantile(0.75)[4]
results_incr.loc['10000 to 100000  (' + str(len(indices_10000)) + ' elements)', '75th percentile q-error'] = indices_10000.quantile(0.75)[4]
results_incr.loc['All errors  (' + str(len(all_errors)) + ' elements)', '75th percentile q-error'] = all_errors.quantile(0.75)[4]

results_equi.loc[str(bin1['true_card'].min()) + ' to ' + str(bin1['true_card'].max()) + '  (' + str(len(bin1)) + ' elements)', '75th percentile q-error'] = bin1.quantile(0.75)[4]
results_equi.loc[str(bin2['true_card'].min()) + ' to ' + str(bin2['true_card'].max()) + '  (' + str(len(bin2)) + ' elements)', '75th percentile q-error'] = bin2.quantile(0.75)[4]
results_equi.loc[str(bin3['true_card'].min()) + ' to ' + str(bin3['true_card'].max()) + '  (' + str(len(bin3)) + ' elements)', '75th percentile q-error'] = bin3.quantile(0.75)[4]
results_equi.loc[str(bin4['true_card'].min()) + ' to ' + str(bin4['true_card'].max()) + '  (' + str(len(bin4)) + ' elements)', '75th percentile q-error'] = bin4.quantile(0.75)[4]
results_equi.loc[str(bin5['true_card'].min()) + ' to ' + str(bin5['true_card'].max()) + '  (' + str(len(bin5)) + ' elements)', '75th percentile q-error'] = bin5.quantile(0.75)[4]
results_equi.loc['All errors  (' + str(len(all_errors)) + ' elements)', '75th percentile q-error'] = all_errors.quantile(0.75)[4]

indices_1_75 = indices_1.copy()
indices_10_75 = indices_10.copy()
indices_100_75 = indices_100.copy()
indices_1000_75 = indices_1000.copy()
indices_10000_75 = indices_10000.copy()
bin1_75 = bin1.copy()
bin2_75 = bin2.copy()
bin3_75 = bin3.copy()
bin4_75 = bin4.copy()
bin5_75 = bin5.copy()

indices_1_75 = indices_1_75.sort_values(by=['q_error'], ascending=False)
bin1_75 = bin1_75.sort_values(by=['q_error'], ascending=False)
for i in range(0, int(0.25*len(indices_1_75))):
    indices_1_75 = indices_1_75.iloc[1:]
    bin1_75 = bin1_75.iloc[1:]

indices_10_75 = indices_10_75.sort_values(by=['q_error'], ascending=False)
bin2_75 = bin2_75.sort_values(by=['q_error'], ascending=False)
for i in range(0, int(0.25*len(indices_10_75))):
    indices_10_75 = indices_10_75.iloc[1:]
    bin2_75 = bin2_75.iloc[1:]

indices_100_75 = indices_100_75.sort_values(by=['q_error'], ascending=False)
bin3_75 = bin3_75.sort_values(by=['q_error'], ascending=False)
for i in range(0, int(0.25*len(indices_100_75))):
    indices_100_75 = indices_100_75.iloc[1:]
    bin3_75 = bin3_75.iloc[1:]

indices_1000_75 = indices_1000_75.sort_values(by=['q_error'], ascending=False)
bin4_75 = bin4_75.sort_values(by=['q_error'], ascending=False)
for i in range(0, int(0.25*len(indices_1000_75))):
    indices_1000_75 = indices_1000_75.iloc[1:]
    bin4_75 = bin4_75.iloc[1:]

indices_10000_75 = indices_10000_75.sort_values(by=['q_error'], ascending=False)
bin5_75 = bin5_75.sort_values(by=['q_error'], ascending=False)
for i in range(0, int(0.25*len(indices_10000_75))):
    indices_10000_75 = indices_10000_75.iloc[1:]
    bin5_75 = bin5_75.iloc[1:]

results_incr.loc['1 to 10  (' + str(len(indices_1)) + ' elements)', '75th percentile accuracy'] = func.calc_classification_accuracy(bins1[0], indices_1_75)
results_incr.loc['10 to 100  (' + str(len(indices_10)) + ' elements)', '75th percentile accuracy'] = func.calc_classification_accuracy(bins1[1], indices_10_75)
results_incr.loc['100 to 1000  (' + str(len(indices_100)) + ' elements)', '75th percentile accuracy'] = func.calc_classification_accuracy(bins1[2], indices_100_75)
results_incr.loc['1000 to 10000  (' + str(len(indices_1000)) + ' elements)', '75th percentile accuracy'] = func.calc_classification_accuracy(bins1[3], indices_1000_75)
results_incr.loc['10000 to 100000  (' + str(len(indices_10000)) + ' elements)', '75th percentile accuracy'] = func.calc_classification_accuracy(bins1[4], indices_10000_75)

results_equi.loc[str(bin1['true_card'].min()) + ' to ' + str(bin1['true_card'].max()) + '  (' + str(len(bin1)) + ' elements)', '75th percentile accuracy'] = func.calc_classification_accuracy(bins2[0], bin1_75)
results_equi.loc[str(bin2['true_card'].min()) + ' to ' + str(bin2['true_card'].max()) + '  (' + str(len(bin2)) + ' elements)', '75th percentile accuracy'] = func.calc_classification_accuracy(bins2[1], bin2_75)
results_equi.loc[str(bin3['true_card'].min()) + ' to ' + str(bin3['true_card'].max()) + '  (' + str(len(bin3)) + ' elements)', '75th percentile accuracy'] = func.calc_classification_accuracy(bins2[2], bin3_75)
results_equi.loc[str(bin4['true_card'].min()) + ' to ' + str(bin4['true_card'].max()) + '  (' + str(len(bin4)) + ' elements)', '75th percentile accuracy'] = func.calc_classification_accuracy(bins2[3], bin4_75)
results_equi.loc[str(bin5['true_card'].min()) + ' to ' + str(bin5['true_card'].max()) + '  (' + str(len(bin5)) + ' elements)', '75th percentile accuracy'] = func.calc_classification_accuracy(bins2[4], bin5_75)

# Delete all temporary files
for i in range(no_clusters):
    os.remove('cluster_' + str(i) + '.txt')
    os.remove('cluster_' + str(i) + '_test.txt')

# Write the results to the output file
output_file.write('Incremental query errors:\n')
output_file.write(str(results_incr) + '\n\n')

output_file.write('Equi-width query errors:\n')
output_file.write(str(results_equi) + '\n\n')
output_file.close()

# Serialize the results to a binary file
results_incr.to_pickle('results/'+dataset_name+'/kgc_exp_'+query_type+'_'+str(query_join)+'_'+str(no_clusters)+'_'+str(epochs)+'_'+str(batch_size)+'_incr.pkl')
results_equi.to_pickle('results/'+dataset_name+'/kgc_exp_'+query_type+'_'+str(query_join)+'_'+str(no_clusters)+'_'+str(epochs)+'_'+str(batch_size)+'_equi.pkl')
all_errors.to_pickle('results/'+dataset_name+'/kgc_exp_'+query_type+'_'+str(query_join)+'_'+str(no_clusters)+'_'+str(epochs)+'_'+str(batch_size)+'_all_errors.pkl')
#######################################################