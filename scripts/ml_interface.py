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

class MLInterface:
    """
    A flexible machine learning interface class that:
    - Loads data (split, pre-split, or full)
    - Selects and configures models (Logistic Regression, Random Forest, SVM)
    - Trains the chosen model
    - Evaluates model performance (hold-out, cross-validation)
    - Generates evaluation reports and plots
    - Performs grid search for hyperparameter tuning
    - Applies bagging/boosting to the current model
    """

    def __init__(self, target_column: str = 'PwD'):
        self.target_column = target_column
        self.data: Dict[str, Dict[str, pd.DataFrame]] = {}  # e.g. {'train': {'X':..., 'Y':...}, 'test':{'X':...,'Y':...}}
        self.full_data: Optional[pd.DataFrame] = None
        self.model = None
        self.classes_: List[Any] = []

    # ============================
    # Data Loading
    # ============================
    def load_data(
        self, 
        filepath: str, 
        mode: str = 'split', 
        test_size: float = 0.3, 
        random_state: int = 42
    ):
        """
        Loads the dataset. Depending on the mode:
        - 'split': Split into train/test using train_test_split
        - 'full': Load full dataset (X,Y) without splitting
        - 'presplit': Load pre-split files (e.g. train_master.csv & test_master.csv)
        
        Parameters:
        - filepath: path to the csv if mode='full' or 'split'
        - mode: One of ['split', 'full', 'presplit']
        - test_size: only used if mode='split'
        """
        if mode == 'full':
            df = pd.read_csv(filepath)
            X = df.drop(self.target_column, axis=1)
            Y = df[self.target_column]
            self.full_data = df
            self.data = {'full': {'X': X, 'Y': Y}}
        
        elif mode == 'split':
            df = pd.read_csv(filepath)
            X = df.drop(self.target_column, axis=1)
            Y = df[self.target_column]
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=test_size, random_state=random_state, stratify=Y
            )
            self.data = {
                'train': {'X': X_train, 'Y': Y_train},
                'test': {'X': X_test, 'Y': Y_test}
            }

        elif mode == 'presplit':
            # Expecting files named 'train_master.csv' and 'test_master.csv' or a pattern thereof
            # If custom naming is needed, adjust accordingly.
            train_path = 'train_master.csv' if filepath == '' else f'{filepath}_train_master.csv'
            test_path = 'test_master.csv' if filepath == '' else f'{filepath}_test_master.csv'

            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            self.data = {
                'train': {'X': df_train.drop(self.target_column, axis=1), 'Y': df_train[self.target_column]},
                'test': {'X': df_test.drop(self.target_column, axis=1), 'Y': df_test[self.target_column]}
            }
        else:
            raise ValueError("Mode must be one of ['split', 'full', 'presplit']")

        print(f"Data loaded with mode={mode}. Keys in self.data: {list(self.data.keys())}")

    # ============================
    # Model Selection
    # ============================
    def select_model(self, model_type: str = 'lg', **model_params):
        """
        Select the model type: 'lg' for LogisticRegression, 'rf' for RandomForest, 'svm' for SVC.
        Additional model_params can be passed directly to the constructor.
        """
        if model_type == 'lg':
            self.model = LogisticRegression(random_state=42, **model_params)
        elif model_type == 'rf':
            self.model = RandomForestClassifier(random_state=42, n_jobs=-1, **model_params)
        elif model_type == 'svm':
            self.model = SVC(probability=True, random_state=42, **model_params)
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
        self.classes_ = sorted(Y.unique())
        print(f"Model trained on {dataset} set with {X.shape[0]} samples and {X.shape[1]} features.")

    # ============================
    # Evaluation
    # ============================
    def evaluate_model(
        self, 
        method: str = 'holdOut', 
        dataset: str = 'test', 
        n_splits: int = 5,
        produce_classification_report: bool = True,
        produce_roc_curve_plot: bool = True,
        produce_confusion_matrix_plot: bool = True
    ):
        """
        Evaluate the model using the specified method:
        - 'holdOut': Uses self.data[dataset] for evaluation.
        - 'crossValidation': Perform cross-validation on full (X,Y) from self.data['full'].

        Parameters:
        - method: 'holdOut' or 'crossValidation'
        - dataset: dataset key to use if holdOut
        - n_splits: for cross-validation
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
            self._report_results(results, produce_classification_report, produce_roc_curve_plot, produce_confusion_matrix_plot)

        elif method == 'crossValidation':
            if 'full' not in self.data:
                raise ValueError("Full data not loaded. Cannot do cross-validation.")
            X = self.data['full']['X']
            Y = self.data['full']['Y']
            results = self._cross_validate_model(X, Y, n_splits=n_splits)
            self._report_results(results, produce_classification_report, produce_roc_curve_plot, produce_confusion_matrix_plot)
        else:
            raise ValueError("method must be 'holdOut' or 'crossValidation'")

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
            ensemble_model = BaggingClassifier(base_estimator=base_model, n_estimators=n_estimators, random_state=42, n_jobs=-1, **kwargs)
        elif method == 'boosting':
            ensemble_model = AdaBoostClassifier(base_estimator=base_model, n_estimators=n_estimators, random_state=42, **kwargs)
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
    pass