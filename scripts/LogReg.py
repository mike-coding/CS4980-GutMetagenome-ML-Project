from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import clone
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


#model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=42)
#model = LogisticRegression(penalty='l1',solver='liblinear',tol=0.01,C=100,fit_intercept=False,l1_ratio=0.5,max_iter=100, random_state=42)
#model = LogisticRegression(random_state=42, C=1.225,fit_intercept=False,l1_ratio=0.0,max_iter=12,penalty='l1',solver='liblinear',tol=0.0011)
#model = RandomForestClassifier(bootstrap=True, criterion='gini',max_depth=None,max_features=None,min_samples_leaf=2,min_samples_split=2,n_estimators=1000,n_jobs=-1,random_state=42,verbose=0)

# data loading, preprocessing
def load_dataset(setName="", perform_split=True, test_size=0.3, random_state=42):
    df = pd.read_csv(setName)
    target_column = 'PwD'
    X = df.drop(target_column, axis=1)
    Y = df[target_column]
    if perform_split:
        setDict = split_data(X,Y,test_size,random_state)
        return setDict
    else:
        return X,Y

def load_presplit_data_old():
    target_column='PwD'
    df_train = pd.read_csv('train_master.csv')
    df_test = pd.read_csv('test_master.csv')
    X_train = df_train.drop(target_column,axis=1)
    Y_train = df_train[target_column]
    X_test = df_test.drop(target_column,axis=1)
    Y_test = df_test[target_column]
    return X_train, X_test, Y_train, Y_test

def load_presplit_data(data=""):
    target_column='PwD'
    setDict = {'train':{}, 'test':{}}
    for setName in setDict.keys():
        filepath = f'{setName}_master.csv'
        if len(data)>0:
            filepath = f'{data}_{setName}_master.csv'
        df = pd.read_csv(filepath)
        setDict[setName]['X'] = df.drop(target_column,axis=1)
        setDict[setName]['Y'] = df[target_column]
    return setDict

def split_data(X, Y, test_size=0.3, random_state=42):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state, stratify=Y)
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    setDict = {'train':{'X':X_train, 'Y':Y_train},'test':{'X':X_test,'Y':Y_test}}
    return setDict


# model training
def train_log_reg_elastic(X_train, Y_train):
    model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=42)
    model.fit(X_train, Y_train)
    return model

def train_log_reg_classic(X_train, Y_train):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, Y_train)
    return model

def train_bagged_log_reg(X_train, Y_train):
    model = BaggingClassifier(
        estimator = LogisticRegression(random_state=42, C=1.225,fit_intercept=False,l1_ratio=0.0,max_iter=12,penalty='l1',solver='liblinear',tol=0.0011),
        n_estimators=105,
        random_state=42,
        max_samples=0.9,
        max_features=0.85,
        bootstrap=False,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X_train, Y_train)
    return model

def train_random_forest_classic(X_train, Y_train, create_feature_importance_report=False):
    #n_estimators=600
    model = RandomForestClassifier(bootstrap=True, criterion='gini',max_depth=None,max_features=None,min_samples_leaf=2,min_samples_split=2,n_estimators=1000,n_jobs=-1,random_state=42,verbose=0)
    model.fit(X_train, Y_train)
    if create_feature_importance_report:
        report_feature_importances(model, X_train.columns, True)
    return model


def train_log_reg_grid_search(X_train,Y_train,gridLevel):
    random_state=42
    fit_intercept=False
    l1_ratio=0.0
    penalty="l1"
    solver="liblinear"
    tol=0.001
    C=1
    max_iter=100
    if gridLevel==2:
        C=1.25
        max_iter=25
    if gridLevel==3:
        C=1.225
        max_iter=12
        tol=0.00011
    model = LogisticRegression(random_state=random_state, C=C,fit_intercept=fit_intercept,l1_ratio=l1_ratio,max_iter=max_iter,penalty=penalty,solver=solver,tol=tol)
    model.fit(X_train,Y_train)
    return model

def get_log_reg_grid_search_model(gridLevel):
    random_state=42
    fit_intercept=False
    l1_ratio=0.0
    penalty="l1"
    solver="liblinear"
    tol=0.001
    C=1
    max_iter=100
    if gridLevel==2:
        C=1.25
        max_iter=25
    if gridLevel==3:
        C=1.225
        max_iter=12
        tol=0.00011
    model = LogisticRegression(random_state=random_state, C=C,fit_intercept=fit_intercept,l1_ratio=l1_ratio,max_iter=max_iter,penalty=penalty,solver=solver,tol=tol)
    return model

# model eval
def report_feature_importances(rf_model, feature_names, writeToCSV=False):
    importances = rf_model.feature_importances_
    feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    if writeToCSV:
        feature_importances.to_csv('feature_importances.csv')
        
def make_predictions(model, X_test):
    Y_pred = model.predict(X_test)
    return Y_pred

def evaluate_model(Y_test, Y_pred, Y_prob=None):
    accuracy = accuracy_score(Y_test, Y_pred)
    cm = confusion_matrix(Y_test,Y_pred)
    cr = classification_report(Y_test, Y_pred, output_dict=False)
    roc_auc_val = None
    if Y_prob is not None and len(Y_prob) == len(Y_test):
        fpr, tpr, _ = roc_curve(Y_test, Y_prob)
        roc_auc_val = auc(fpr, tpr)

    return {
        'accuracy': accuracy,
        'cm': cm,
        'cr': cr,
        'roc_auc': roc_auc_val,
        'fpr': fpr,
        'tpr': tpr,
        'classes': sorted(Y_test.unique())
    }

def evaluate_model_OLD_style():
    setDict = load_presplit_data('study2_yolo')
    #model = train_log_reg_grid_search(setDict['train']['X'],setDict['train']['Y'],3)
    model = train_random_forest_classic(setDict['train']['X'],setDict['train']['Y'])
    Y_pred = make_predictions(model, setDict['test']['X'])
    Y_prob = model.predict_proba(setDict['test']['X'])[:, 1]
    results = evaluate_model(setDict['test']['Y'], Y_pred, Y_prob)
    report_results(results)

def cross_validate_model(model, X, Y, n_splits=10, random_state=42):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_preds = []
    all_probs = []
    all_truths = []

    for train_idx, test_idx in cv.split(X, Y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

        # Clone the model to ensure a fresh model for each fold
        fold_model = clone(model)
        fold_model.fit(X_train, Y_train)

        # Get predictions
        preds = fold_model.predict(X_test)
        all_preds.extend(preds)
        all_truths.extend(Y_test)

        # If the model supports predict_proba, gather probabilities for ROC
        if hasattr(fold_model, "predict_proba"):
            probs = fold_model.predict_proba(X_test)[:, 1]
            all_probs.extend(probs)

    # Convert all_truths and all_preds to proper arrays if needed
    all_truths = pd.Series(all_truths)
    all_preds = pd.Series(all_preds)

    # Compute metrics from combined predictions
    accuracy = accuracy_score(all_truths, all_preds)
    cm = confusion_matrix(all_truths, all_preds)
    cr = classification_report(all_truths, all_preds, output_dict=False)

    # ROC-AUC
    fpr, tpr, roc_auc_val = None, None, None
    if len(all_probs) == len(all_truths) and len(all_probs) > 0:
        fpr, tpr, _ = roc_curve(all_truths, all_probs)
        roc_auc_val = auc(fpr, tpr)

    results = {
        'accuracy': accuracy,
        'cm': cm,
        'cr': cr,
        'roc_auc': roc_auc_val,
        'fpr': fpr,
        'tpr': tpr,
        'classes': sorted(Y.unique())
    }
    return results

def verify_data_splits(setDict):
    target_column='PwD'
    for set in setDict.keys():
        if target_column in setDict[set]['X'].columns:
            print(f'{target_column} column present in set: {set}_X')
            return
        else:
            print(f'{target_column} column was not found in set: {set}_X')

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

def plot_classification_report(cr, title='Classification Report'):
    cr_df = pd.DataFrame(cr).transpose()
    
    # Optionally, remove the 'accuracy', 'macro avg', and 'weighted avg' rows if not needed
    rows_to_drop = ['accuracy', 'macro avg', 'weighted avg']
    cr_df = cr_df.drop(rows_to_drop, errors='ignore')
    
    # Round the values for better display
    cr_df = cr_df.round(2)
    
    # Reset index to have 'Class' as a column
    cr_df.reset_index(inplace=True)
    cr_df.rename(columns={'index': 'Class'}, inplace=True)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 2 + 0.5 * len(cr_df)))  # Adjust height based on number of classes
    
    # Hide axes
    ax.axis('off')
    ax.axis('tight')
    
    # Create the table
    table = ax.table(cellText=cr_df.values, colLabels=cr_df.columns, loc='center', cellLoc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Add title
    plt.title(title, fontsize=16, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0,1],[0,1],'r--')  # Baseline
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def report_results(results):
    print(f'\nRun Complete.')
    print(f'\nAccuracy: {results["accuracy"]}')
    print(f'\nClassification Report:\n\n{results["cr"]}')
    if 'roc_auc' in results.keys(): 
        print(f'\nROC AUC result: {results["roc_auc"]}')
        plot_roc_curve(results['fpr'], results['tpr'], results['roc_auc'])
    plot_confusion_matrix(cm=results['cm'], classes=results['classes'], title="Classic Logistic Regression")

def perform_cross_validation_test():
    X,Y = load_dataset('study2_yolo_train_master.csv',perform_split=False)

    #model = get_log_reg_grid_search_model(3)
    model = RandomForestClassifier(bootstrap=True, criterion='gini',max_depth=None,max_features=None,min_samples_leaf=2,min_samples_split=2,n_estimators=1000,n_jobs=-1,random_state=42,verbose=0)
    '''

        model = model = BaggingClassifier(
        estimator = LogisticRegression(random_state=42, C=1.225,fit_intercept=False,l1_ratio=0.0,max_iter=12,penalty='l1',solver='liblinear',tol=0.0011),
        n_estimators=105,
        random_state=42,
        max_samples=0.9,
        max_features=0.85,
        bootstrap=False,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X,Y)
    X_test, Y_test = load_dataset('study2_yolo_test_master.csv', perform_split=False)
    Y_pred = make_predictions(model, X_test)
    Y_prob = model.predict_proba(X_test)[:, 1]
    results = evaluate_model(Y_test, Y_pred, Y_prob)
    report_results(results)
    model = BaggingClassifier(
        estimator = LogisticRegression(random_state=42, C=1.225,fit_intercept=False,l1_ratio=0.0,max_iter=12,penalty='l1',solver='liblinear',tol=0.0011),
        n_estimators=105,
        random_state=42,
        max_samples=0.9,
        max_features=0.85,
        bootstrap=False,
        n_jobs=-1,
        verbose=0,
    )
    '''
    results = cross_validate_model(model, X, Y)
    report_results(results)

def perform_yolo_test():
    X,Y = load_dataset('study2_yolo_master.csv',perform_split=False)
    model = LogisticRegression(random_state=42, C=1.225,fit_intercept=False,l1_ratio=0.0,max_iter=12,penalty='l1',solver='liblinear',tol=0.0011)

def find_ideal_logReg_params():
    X,Y = load_dataset('study2_master.csv',perform_split=False)
    model = LogisticRegression()
    param_grid = {
    'penalty': ['none', 'l1', 'l2', 'elasticnet'],
    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    'fit_intercept': [True, False],
    'solver': ['lbfgs', 'liblinear', 'sag', 'saga', 'newton-cg'],
    'max_iter': [100, 200, 500, 1000],
    'l1_ratio': [0.0, 0.5, 1.0],
    'tol': [1e-4, 1e-3, 1e-2],
    'random_state':[42]
    }
    param_grid2 = {
        'penalty':['l1'],
        'C':[1.225,1.25,1.275],
        'fit_intercept':[False],
        'l1_ratio':[0.0],
        'max_iter':[9,10,11,12,13,14],
        'solver':['liblinear'],
        'tol':[0.00095, 0.0011, 0.000115],
        'random_state':[42]
    }
    perform_grid_search(X,Y, model, param_grid2)

def find_ideal_RF_params():
    X,Y = load_dataset('study2_master.csv',perform_split=False)
    model = RandomForestClassifier()
    param_grid = {
        'criterion': ['gini'],
        'n_estimators': [150,175,200,225,250],
        'max_depth': [None],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [2],
        'max_features': [None],
        'bootstrap': [True],
        'n_jobs':[-1],
        'random_state':[42],
        'verbose':[0]
    }
    perform_grid_search(X,Y, model, param_grid)

def find_ideal_SVC_params():
    X,Y = load_dataset('study2_master.csv',perform_split=False)
    model = SVC()
    param_grid = {}
    perform_grid_search(X,Y, model, param_grid)

def find_ideal_SGDClassifier_params():
    X,Y = load_dataset('study2_master.csv',perform_split=False)
    model= SGDClassifier()
    param_grid = {
    'loss': ['squared_hinge'],
    'penalty': ['elasticnet'],
    'alpha': [0.0001],
    'learning_rate': ['optimal'],
    'eta0': [0.0],
    'tol': [0.0001],
    'early_stopping': [False],
    'class_weight': [None],
    'n_jobs':[-1],
    'random_state':[42],
    'max_iter':[10,20,25,30,35,40,50]
    }
    perform_grid_search(X,Y, model, param_grid)

def find_ideal_bagged_logReg_params():
    X,Y = load_dataset('study2_master.csv',perform_split=False)
    model = BaggingClassifier()
    param_grid= {
    'estimator':[LogisticRegression(random_state=42, C=1.225,fit_intercept=False,l1_ratio=0.0,max_iter=12,penalty='l1',solver='liblinear',tol=0.0011)],
    'n_estimators':[105,110,115],
    'max_samples':[0.85,0.9,0.95],
    'max_features':[0.85,0.875,0.9],
    'bootstrap':[False],
    'n_jobs':[-1],
    'random_state':[42],
    'verbose':[0]
    }
    perform_grid_search(X,Y,model,param_grid)

def perform_grid_search(X, Y, model, parameter_grid):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
    estimator=model,
    param_grid=parameter_grid,
    scoring='roc_auc',  
    cv=cv,
    n_jobs=-1, 
    error_score=np.nan, 
    verbose=1
    )
    grid_search.fit(X, Y)
    print("Best Parameters found by Grid Search:", grid_search.best_params_)
    print("Best ROC-AUC found by Grid Search:", grid_search.best_score_)

#find_ideal_params()
perform_cross_validation_test()
#evaluate_model_OLD_style()
#find_ideal_RF_params()
#find_ideal_SGDClassifier_params()
#find_ideal_bagged_logReg_params()