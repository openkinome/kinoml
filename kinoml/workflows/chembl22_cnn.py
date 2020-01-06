'''
This script uses the pre-processed data file df_reduced.pkl that is generated from the ChEMBL22 data set.

Type of target used: single-target (i.e. ligand-based model)
Type of featurizer for the ligand: One Hot Encoding of SMILES
Type of model: a Convolutional Neural Network (CNN) using Pytorch

Please run in terminal the following command:
$ export PYTHONPATH=$(dirname $(pwd))

And check the PYTHONPATH with: 
$ echo $PYTHONPATH

'''
# Imports
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim

from ml.models_pytorch import CNN
from features.ligand import OneHotSMILESFeaturizer


if __name__ == '__main__':

    # Load pre-processed data
    p = Path(__file__).resolve()
    data = pd.read_pickle(f'{p.parents[1]}/data/df_reduced.pkl')

    # Set threshold for activity
    c = 6.3
    print(f'Threshold at {c}')

    seed_cv = 42 # For reproductibility
    K = 5 # Fold Cross Validation
    k =  35 # Nb of kinase to remove
    nb_epoch = 25 # Nb of epochs for training

    # Statistics per kinase
    acc_mean_per_kinase = []
    acc_std_per_kinase = []
    zero_count_per_kinase = []
    conf_matrix_mean_per_kinase = []
    conf_matrix_std_per_kinase = []
    
    # Build model which will be used for all kinases
    model = CNN()
    print(model)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    criterion = nn.BCELoss() # Binary Cross Entropy Loss

    # Loop over all kinases in the data set and obtain statistics
    for i,kinase in enumerate(data.columns[:(-2-k)]):

        print('====================================================')
        print(f'{K} Fold Cross Validation for kinase #{i+1} {kinase}')

        df = data[[kinase, 'smi_can']]

        # remove NA
        df_na = df.dropna()
        df_na.reset_index(drop=True, inplace=True)

        print(f'Kinase {kinase} has {len(df_na)} measurements')

        # Compute longest smiles
        len_smi = [len(smi) for smi in data['smi_can']]
        max_smi = max(len_smi)

        print(f' Dictionary : {len(OneHotSMILESFeaturizer.DICTIONARY)} and longest smiles : {max_smi} ')

        # Define input: Transform smiles into OH encoding and pad
        x = df_na['smi_can'].apply(lambda x: OneHotSMILESFeaturizer(x, max_smi).featurize())
        x = torch.tensor(x).type(torch.FloatTensor)

        # Convert data to binary values at threshold c

        # If values is >= c, set it to 1 (activity)
        df_thres = df_na.where(df_na[kinase] >= c, 0) # Careful : pd.where replaces values where the condition is False.

        # If values is < c, set it to 0 (no activity)
        df_thres = df_thres.where(df_thres[kinase] < c, 1)

        # Define output :
        y = df_thres[kinase]
        y = torch.from_numpy(np.asarray(y)).type(torch.FloatTensor)

        if x.shape[0] != y.shape[0]:
            print(f'Shape of input and target differ')
        else:
            print(f'Shape of input: {x.shape} and output: {y.shape} .')


        # Metric per fold (K values)
        acc_per_fold = []
        conf_matrix_per_fold = []

        # K fold Cross Validation
        kfold = KFold(n_splits=K, random_state=seed_cv)
        for j, (train_index, test_index) in enumerate(kfold.split(x, y)):

            print(f'Fold {j}')

            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            print('---Train')

            for epoch in range(nb_epoch):
                
                # Clear the previous gradients
                optimizer.zero_grad()

                # Feedforward pass
                outputs_train = torch.squeeze(model(x_train)) # Use torch.squeeze to remove axis with 1 dimension
                if outputs_train.shape != y_train.shape:
                    print('The shape of the true value of the target differs..'
                          '..from the shape of the value predicted by the model.')
                if torch.min(outputs_train)<0 or torch.max(outputs_train)>1:
                    print('The predicted values are out of range.')
                    
                if all(outputs_train[0] == elem for elem in outputs_train):
                    print(f'The model predicts all values at {outputs_train[0]} .')

                loss = criterion(outputs_train, y_train)
                loss.backward()
                optimizer.step()

            print('---Predict')
            # Test trained model on the test set
            outputs_test = torch.squeeze(model(x_test))

            # Transform target vectors into numpy array for scores
            y_pred = outputs_test.detach().numpy()
            y_test = y_test.detach().numpy()

            # Transform probability of belonging to one class into 0 or 1 value at threshold 0.5
            y_pred =(y_pred>0.5) 

            if(all(y_pred[0] == elem for elem in y_pred)):
                print(f'The model predicts all values at {y_pred[0]}')

            print(f'# of elements on the test set: {len(y_test)}, # zero counts on the test set: {(y_test==0).sum()} ')

            # Metrics on each fold
            acc_per_fold.append(accuracy_score(y_test, y_pred))
            conf_matrix_per_fold.append(confusion_matrix(y_test, y_pred).ravel())

        # Mean on all K folds
        acc_mean = np.mean(acc_per_fold)
        conf_matrix_mean = np.mean(np.array(conf_matrix_per_fold), axis=0)

        # Std on all K folds
        acc_std = np.std(acc_per_fold)
        conf_matrix_std = np.std(np.array(conf_matrix_per_fold), axis=0)

        print(f' Metrics of CV for kinase {kinase} ')
        print(f' Accuracy : Mean : {round(acc_mean, 2)} and std : {round(acc_std, 2)} ')
        print(f' TN : Mean : {round(conf_matrix_mean[0], 2)} and std : {round(conf_matrix_std[0], 2)} ')
        print(f' FP : Mean : {round(conf_matrix_mean[1], 2)} and std : {round(conf_matrix_std[1], 2)} ')
        print(f' FN : Mean : {round(conf_matrix_mean[2], 2)} and std : {round(conf_matrix_std[2], 2)} ')
        print(f' TP : Mean : {round(conf_matrix_mean[3], 2)} and std : {round(conf_matrix_std[3], 2)} ')

        # Zero Counts per kinase
        zero_count = (y==0).sum()
        print(f' Total nb: {len(y)} and nb of zero counts : {zero_count}')
        zero_count_per_kinase.append(zero_count)

        acc_mean_per_kinase.append(acc_mean)
        acc_std_per_kinase.append(acc_std)

        conf_matrix_mean_per_kinase.append(conf_matrix_mean)
        conf_matrix_std_per_kinase.append(conf_matrix_std)


    # Save results to a pickle table in folder cv_results
    results_cv = pd.DataFrame(data=[acc_mean_per_kinase, acc_std_per_kinase,
                                    conf_matrix_mean_per_kinase, conf_matrix_std_per_kinase,
                                    zero_count_per_kinase])
    results_cv.columns = [data.columns[:(-2-k)]]
    results_cv.index = ['Acc Mean', 'Acc Std', 'Conf Mean', 'Conf Std', 'Zero Count']
    results_cv = results_cv.to_pickle('CNN_class_results_cv.pkl')
