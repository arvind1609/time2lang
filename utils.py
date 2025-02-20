import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime

def save_model(model, base_dir, name):
    
    today_date = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(base_dir, today_date)
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(save_dir, f"{name}.pt")
    
    # Save the model's state dictionary
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

def dictionary_to_arrays(my_dict):
    all_features = []
    all_labels = []
    
    for k in my_dict.keys():
        features = torch.tensor(my_dict[k]['features'].clone().detach(), dtype=torch.float32).numpy()
        labels = my_dict[k]['labels']
        
        all_features.append(features)
        all_labels.append(labels)

    all_features = np.vstack(all_features)
    all_labels = np.hstack(all_labels)
    return all_features, all_labels

def dictionary_to_arrays_numpy(my_dict):
    all_features = []
    all_labels = []
    
    for k in my_dict.keys():
        features = my_dict[k]['features']
        labels = my_dict[k]['labels']
        
        all_features.append(features)
        all_labels.append(labels)

    all_features = np.vstack(all_features)
    all_labels = np.hstack(all_labels)
    return all_features, all_labels

def zscore_sample_wise(data):
    
    mean_vals = np.mean(data, axis=1, keepdims=True)
    std_vals = np.std(data, axis=1, keepdims=True)
    normalized_data = (data - mean_vals) / std_vals

    return normalized_data


def zscore(data):
    
    mean_vals = np.mean(data)
    std_vals = np.std(data)
    normalized_data = (data - mean_vals) / std_vals

    return normalized_data

def classification(X_train, y_train, X_test, y_test, backbone='lr'):
    
    if backbone == 'lr':
        model = LogisticRegression(max_iter=1000)
        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.1, 1, 10, 100],  
            'solver': ['saga'],  
            'l1_ratio': [0.5, 0.7, 0.9],  
        }

    if backbone == 'rf':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20, 50],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [1, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
        }
        
    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, scoring='accuracy', verbose=10, n_jobs=-1)

    # Fit the model with the training data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Predict on the test set
    y_pred = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]

    return best_model, best_params, y_pred, y_pred_prob

def apply_sss(X_train, y_train, X_test, y_test, backbone='lr'):
    sss = StratifiedShuffleSplit(n_splits=5, random_state=8)
    auc_avg = []
    aupr_avg = []
    i = 0 
    
    for train_index, _ in sss.split(X_train, y_train):
        print(f"------ Running split {i+1} -------")
        X_train_spl, y_train_spl = X_train[train_index], y_train[train_index]
        best_model, best_params, y_pred, y_pred_proba = classification(X_train=X_train_spl,
                                                                  y_train=y_train_spl,
                                                                  X_test=X_test,
                                                                  y_test=y_test,
                                                                  backbone=backbone)
        auc_avg.append(roc_auc_score(y_test, y_pred_proba))
        aupr_avg.append(average_precision_score(y_test, y_pred_proba))
        i+=1

    return auc_avg, aupr_avg

