import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union, List

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.base import clone
from csv_preprocessor import DataPreprocessor
from sklearn.neighbors import KNeighborsClassifier

class MLInterface:
    """
    - Load datasets via dataPreprocessor.get_experiment()
    - Select and configure models (Logistic Regression, Random Forest, SVM, KNN)
    - Train selected model
    - Evaluate model performance (single hold-out, K-fold cross-validation)
    - Generate evaluation reports, plots
    - Perform grid search for hyperparameter tuning
    - Apply bagging/boosting to selected model
    """
    def __init__(self, target_column: str = 'PwD'):
        self.target_column = target_column
        self.data: Dict[str, Dict[str, pd.DataFrame]] = {}  # e.g. {'train': {'X':..., 'Y':...}, 'test':{'X':...,'Y':...}}
        self.model = None
        self.dataPreprocessor=DataPreprocessor()
        self.path=self.dataPreprocessor.get_data_path()
        self.results_path = os.path.join(self.path,'..', 'results')
        self.random_state=42

    # ============================
    # Data Loading
    # ============================
    def load_experiment_set(self, study: int = 2, experiment_type: str = 'classic'):
        """
        Load all available sets (full, test, train) for the given experiment type and study.
        Parameters:
        - study (int): The study number (1 or 2). Defaults to 2.
        - experiment_type (str): The experiment type (e.g., 'classic', 'genus', 'classic_demo', genus_demo', 'yolo', 'demo'). Defaults to 'classic'.
        """
        self.data = {}
        dataframes=self.dataPreprocessor.get_experiment(experiment_type,study)
        for _set, df in dataframes.items():
            if self.target_column not in df.columns:
                print(f"Warning: {self.target_column} not in {experiment_type}/{_set} data.")
                continue
            X = df.drop(self.target_column, axis=1)
            Y = df[self.target_column]
            self.data[_set] = {'X': X, 'Y': Y}
        print(f"Data loaded for study{study}/{experiment_type}. Available sets: {list(self.data.keys())}")
    
    # ============================
    # Model Selection
    # ============================
    def select_model(self, model_type: str = 'lg', **model_params):
        """
        Select the model type: 'lg' for LogisticRegression, 'rf' for RandomForest, 'svm' for SVC.
        Additional model_params can be passed directly to the constructor.
        """
        if model_type == 'lg':
            self.model = LogisticRegression(random_state=self.random_state, **model_params)
        elif model_type == 'rf':
            self.model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1, **model_params)
        elif model_type == 'svm':
            self.model = SVC(probability=True, random_state=self.random_state, **model_params)
        elif model_type =='knn':
            self.model = KNeighborsClassifier(n_jobs=-1)
        else:
            raise ValueError("model_type must be one of ['lg', 'rf', 'svm']")

        print(f"Model selected: {self.model}")

    # ============================
    # Training
    # ============================
    def train_model(self, dataset: str = 'train'):
        """
        Train the currently selected model on the specified dataset part ('train' or 'full').
        """
        if self.model is None:
            raise ValueError("No model selected. Call select_model() first.")
        if dataset not in self.data:
            raise ValueError(f"No {dataset} data loaded.")

        X = self.data[dataset]['X']
        Y = self.data[dataset]['Y']
        self.model.fit(X, Y)
        print(f"Model trained on {dataset} set with {X.shape[0]} samples and {X.shape[1]} features.")

    # ============================
    # Evaluation
    # ============================
    def evaluate_model(
        self, 
        method: str = 'holdOut', 
        dataset: str = 'test', 
        n_splits: int = 10,
        gen_cr: bool = True,
        gen_roc: bool = True,
        gen_cm: bool = True,
        suppress_report: bool = False
    ):
        """
        Evaluate the model using the specified method:
        - 'holdOut': Uses self.data[dataset] for evaluation.
        - 'crossValidation': Perform cross-validation on full (X,Y) from self.data['full'].

        Parameters:
        - method: 'holdOut' or 'crossValidation'
        - dataset: dataset key to use if holdOut
        - n_splits: for cross-validation
        - gen_cr: whether the function prints a classification report
        - gen_roc: whether the function plots a ROC Cruve
        - gen_cm: whether the function plots a confusion matrix
        """
        if self.model is None:
            raise ValueError("No model selected or trained.")

        if method == 'holdOut':
            if dataset not in self.data:
                raise ValueError(f"No {dataset} data loaded for hold-out evaluation.")
            X_test = self.data[dataset]['X']
            Y_test = self.data[dataset]['Y']
            Y_pred = self.model.predict(X_test)
            Y_prob = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, "predict_proba") else None
            
            results = self._compute_metrics(Y_test, Y_pred, Y_prob)
            self._report_results(results, gen_cr, gen_roc, gen_cm)

        elif method == 'cv':
            if 'full' not in self.data:
                print("[ERROR]: No full data set. Cannot do cross-validation.")
                return self._generate_blank_results()
            X = self.data['full']['X']
            Y = self.data['full']['Y']
            results = self._cross_validate_model(X, Y, n_splits=n_splits)
            if not suppress_report:
                self._report_results(results, gen_cr, gen_roc, gen_cm)
            return results
        else:
            raise ValueError("method must be 'holdOut' or 'cv'")

    def _compute_metrics(self, Y_test, Y_pred, Y_prob=None):
        accuracy = accuracy_score(Y_test, Y_pred)
        cm = confusion_matrix(Y_test, Y_pred)
        cr = classification_report(Y_test, Y_pred, output_dict=False)
        
        fpr, tpr, roc_auc_val = None, None, None
        if Y_prob is not None:
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

    def _generate_blank_results(self):
        return {
            'accuracy': '',
            'cm': '',
            'cr': '',
            'roc_auc': '',
            'fpr': '',
            'tpr': '',
            'classes': ''
        }

    def _report_results(self, results, show_cr, show_roc, show_cm):
        print("\nEvaluation Complete:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        if show_cr:
            print("\nClassification Report:\n", results['cr'])
        if results['roc_auc'] is not None:
            print(f"\nROC AUC: {results['roc_auc']:.4f}")
        if show_roc and results['fpr'] is not None:
            self.plot_roc_curve(results['fpr'], results['tpr'], results['roc_auc'])
        if show_cm:
            self.plot_confusion_matrix(results['cm'], results['classes'], title='Confusion Matrix')

    def _cross_validate_model(self, X, Y, n_splits=5):
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        all_preds = []
        all_probs = []
        all_truths = []

        for train_idx, test_idx in cv.split(X, Y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

            fold_model = clone(self.model)
            fold_model.fit(X_train, Y_train)

            preds = fold_model.predict(X_test)
            all_preds.extend(preds)
            all_truths.extend(Y_test)

            if hasattr(fold_model, "predict_proba"):
                probs = fold_model.predict_proba(X_test)[:, 1]
                all_probs.extend(probs)

        all_truths = pd.Series(all_truths)
        all_preds = pd.Series(all_preds)

        return self._compute_metrics(all_truths, all_preds, all_probs if len(all_probs)==len(all_truths) else None)

    def perform_study_level_experiment(self, study, model):
        if type(study)!=int:
            print(f'[ERROR]: Invalid study: {study}\nExpected integer: 1, or 2')
            return
        self.select_model(model)
        name = f'{type(self.model).__name__}_experiment_on_study{study}'
        resultLog=f'{name}\n========================================================\n========================================================\n\n'
        print(f'Running experiment: {name}')
        for experiment_type in self.dataPreprocessor.valid_experiments[int(study)]:
            self.load_experiment_set(study,experiment_type)
            results = self.evaluate_model('cv',suppress_report=True)
            if len(str(results['accuracy']))<1:
                continue
            resultLog = ''.join([resultLog, f"Experiment: {experiment_type}\n--------------------------------------------------------\n", f'Accuracy: {str(results["accuracy"])}\n', f'ROC_AUC: {str(results["roc_auc"])}\n\n', f'{results["cr"]}\n', f'Confusion matrix:\n{str(results["cm"])}\n\n\n\n'])
        output_file = os.path.join(self.results_path, f"{name}.txt")
        with open(output_file, 'w') as f:
            f.write(resultLog)
        print(resultLog[:-3])
        print(f"Results saved to {output_file}")

    # ============================
    # Plotting Utilities
    # ============================
    def plot_confusion_matrix(self, cm, classes, title='Confusion Matrix'):
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, fpr, tpr, roc_auc):
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

    # ============================
    # Grid Search & Hyperparameters
    # ============================
    def grid_search_params(self, param_grid: dict, scoring='roc_auc', n_splits=5):
        """
        Perform grid search on the currently selected model.
        """
        if self.model is None:
            raise ValueError("No model selected. select_model() first.")
        if 'full' not in self.data:
            raise ValueError("Full data not loaded. Cannot do grid search.")

        X = self.data['full']['X']
        Y = self.data['full']['Y']
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            error_score=np.nan,
            verbose=1
        )
        grid_search.fit(X, Y)
        print("Best Parameters found by Grid Search:", grid_search.best_params_)
        print("Best Score found by Grid Search:", grid_search.best_score_)

        # Update the model with best estimator
        self.model = grid_search.best_estimator_

    # ============================
    # Bagging/Boosting
    # ============================
    def bag_or_boost_current_model(
        self, 
        method: str = 'bagging', 
        n_estimators: int = 50, 
        **kwargs
    ):
        """
        Wraps the current model in a BaggingClassifier or AdaBoostClassifier.
        
        Parameters:
        - method: 'bagging' or 'boosting'
        - n_estimators: number of estimators
        - kwargs: additional params to the ensemble
        """
        if self.model is None:
            raise ValueError("No model selected. select_model() first.")
        
        base_model = clone(self.model)
        if method == 'bagging':
            ensemble_model = BaggingClassifier(estimator=base_model, n_estimators=n_estimators, random_state=42, n_jobs=-1, **kwargs)
        elif method == 'boosting':
            ensemble_model = AdaBoostClassifier(estimator=base_model, n_estimators=n_estimators, random_state=42, **kwargs)
        else:
            raise ValueError("method must be 'bagging' or 'boosting'")

        # Retrain ensemble on train data if available
        if 'train' in self.data:
            X_train = self.data['train']['X']
            Y_train = self.data['train']['Y']
            ensemble_model.fit(X_train, Y_train)
            self.model = ensemble_model
            print(f"Current model wrapped with {method}.")
        else:
            self.model = ensemble_model
            print(f"Current model wrapped with {method}, but no training data found. Train again using train_model().")

if __name__ == "__main__":
    interface=MLInterface()
    interface.perform_study_level_experiment(1,'svm')