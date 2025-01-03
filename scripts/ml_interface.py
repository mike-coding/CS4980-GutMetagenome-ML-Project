import os
import pandas as pd
from typing import Optional, Dict, Any, Union, List

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay, permutation_importance
from sklearn.decomposition import PCA

from csv_preprocessor import DataPreprocessor
from model_hyperparameters import HyperparameterConfig

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
        self.current_data_set = None
        self.current_experiment=None
        self.model = None
        self.dataPreprocessor=DataPreprocessor()
        self.data_path=self.dataPreprocessor.get_data_path()
        self.results_path = os.path.join(self.data_path,'..', 'results')
        self.random_state=42
        self.result_log={'title':'','body':''}

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
        self.current_data_set = f"Study{study}_{experiment_type}"
        print(f"Data loaded for study{study}/{experiment_type}. Available sets: {list(self.data.keys())}")
    
    # ============================
    # Model Selection, Properties
    # ============================
    def select_model(self, model_type: str = 'lg', **model_params):
        """
        Select the model type: 'lg' for LogisticRegression, 'rf' for RandomForest, 'svm' for SVC.
        Additional model_params can be passed directly to the constructor.
        """
        model_params.pop('random_state', None)
        model_params.pop('n_jobs', None)
        model_params.pop('kernel', None)
        model_params.pop('probability', None)
        if model_type == 'lg' or model_type == 'LogisticRegression':
            self.model = LogisticRegression(random_state=self.random_state, **model_params)
        elif model_type == 'rf' or model_type == 'RandomForestClassifier':
            self.model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1, **model_params)
        elif model_type == 'SVC_rbf':
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', SVC(probability=True, random_state=self.random_state, **model_params))
            ])
        elif model_type == 'SVC_poly':
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', SVC(probability=True, random_state=self.random_state, kernel='poly', **model_params))
            ])
        elif model_type == 'LinearSVC':
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', CalibratedClassifierCV(estimator=LinearSVC(random_state=self.random_state, **model_params)))
            ])
        elif model_type == 'knn' or model_type == 'kneighborsclassifier':
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', KNeighborsClassifier(n_jobs=-1))
            ])
        else:
            raise ValueError("model_type must be one of ['lg', 'rf', 'svm']")

        print(f"Model selected: {self.model}")

    def get_model_name(self):
        name= type(self.model).__name__
        if name=='Pipeline':
            name = self.get_pipeline_model_name()
        return name
    
    def get_pipeline_model_name(self):
        clf = self.model.named_steps['clf']
        base = clf.estimator if hasattr(clf, 'estimator') else clf
        name = type(base).__name__
        if name == 'SVC':
            name += f'_{base.kernel}'
        return name

    # ============================
    # Training
    # ============================
    def train_model(self, dataset: str = 'train', use_reduced_x=False):
        """
        Train the currently selected model on the specified dataset part ('train' or 'full').
        """
        if self.model is None:
            raise ValueError("No model selected. Call select_model() first.")
        if dataset not in self.data:
            raise ValueError(f"No {dataset} data loaded.")

        X = self.data[dataset][f'X{"_reduced" if use_reduced_x else ""}']
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
            if not suppress_report:
                self._report_results(results, gen_cr, gen_roc, gen_cm)
            return results

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
            print("[ERROR]: Method must be 'holdOut' or 'cv'")
            return self._generate_blank_results()

    def _compute_metrics(self, Y_test, Y_pred, Y_prob=None):
        accuracy = accuracy_score(Y_test, Y_pred)
        cm = confusion_matrix(Y_test, Y_pred)
        cr = classification_report(Y_test, Y_pred, output_dict=False)
        
        fpr, tpr, roc_auc_val = None, None, None
        if Y_prob is not None: #perform check for 
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
        self.start_new_result_log(name)
        print(f'Running experiment: {name}')
        for experiment_type in self.dataPreprocessor.valid_experiments[int(study)]:
            self.load_experiment_set(study,experiment_type)
            results = self.evaluate_model('cv',suppress_report=True)
            if len(str(results['accuracy']))<1:
                continue
            self.write_to_result_log(results, experiment_type)
        self.dump_result_log()

    def run_experiment_1(self):
        self.current_experiment=1
        reset_model = lambda m: self.select_model('lg',penalty='elasticnet',solver='saga',l1_ratio=0.5) if m=='lg' else self.select_model('rf')
        for model in ['lg', 'rf']:
            reset_model(model)
            model_name=self.get_model_name()
            ## part a
            self.start_new_result_log(f'{model_name} Comparative Analysis', 'experiment_1')
            reset_model(model)
            self.load_experiment_set(2,'classic')
            self.train_model()
            results = self.evaluate_model(suppress_report=True)
            self.write_to_result_log(results, 'Original Dataset Split & Original Parameters')
            self.direct_plot_results(results, f'{model_name} Original Data Split & Parameters')
            #10-fold CV test
            reset_model(model)
            results = self.evaluate_model(method='cv',suppress_report=True)
            self.write_to_result_log(results, '10-Fold Cross Validation & Original Parameters')
            self.direct_plot_results(results, f'{model_name} 10-Fold CV & Original Parameters')
            # grid search test
            if model=='lg': self.select_model('lg') 
            else: self.select_model('rf')
            self.load_grid_search_parameters()
            results = self.evaluate_model(method='cv',suppress_report=True)
            self.write_to_result_log(results, '10-Fold Cross Validation & Grid Search Parameters')
            self.direct_plot_results(results, f'{model_name} 10-Fold CV & Grid Search Parameters')
            if model=='lg':
                #try bagging
                self.bag_or_boost_current_model(n_estimators=105,max_samples=0.9,max_features=0.85,bootstrap=False,verbose=0)
                results = self.evaluate_model(method='cv',suppress_report=True)
                self.write_to_result_log(results, '10-Fold Cross Validation & Bagged Grid Search Best Parameters')
                self.direct_plot_results(results, f'Bagged {model_name} 10-Fold CV & Grid Search Parameters')
            self.dump_result_log()

    def run_experiment_2(self):
        self.current_experiment=2
        for model in [('SVC_rbf','RBF'), ('SVC_poly', 'Poly'), ('LinearSVC', 'Linear')]:
            model_name = model[0]
            kernel = model[1]
            self.start_new_result_log(f'Support Vector Machine ({kernel}) Performance On Metagenomic Signature Data', 'experiment_2')
            self.select_model(model_name)

            ## default params, default split
            self.load_experiment_set(2,'classic')
            self.train_model()
            results = self.evaluate_model(suppress_report=True)
            self.write_to_result_log(results, 'Original Dataset Split & Default Parameters')
            self.direct_plot_results(results, f'{model_name} Original Data Split & Default Parameters')

            #10-fold CV test
            self.select_model(model_name)
            results = self.evaluate_model(method='cv',suppress_report=True)
            self.write_to_result_log(results, '10-Fold Cross Validation & Default Parameters')
            self.direct_plot_results(results, f'{model_name} 10-Fold CV & Default Parameters')

            # grid search test
            self.select_model(model_name)
            self.load_grid_search_parameters()
            results = self.evaluate_model(method='cv',suppress_report=True)
            self.write_to_result_log(results, '10-Fold Cross Validation & Grid Search Parameters')
            self.direct_plot_results(results, f'{model_name} 10-Fold CV & Grid Search Parameters')

            #try bagging
            self.bag_or_boost_current_model(n_estimators=105,max_samples=0.9,max_features=0.85,bootstrap=False,verbose=0)
            results = self.evaluate_model(method='cv',suppress_report=True)
            self.write_to_result_log(results, '10-Fold Cross Validation & Bagged Grid Search Best Parameters')
            self.direct_plot_results(results, f'Bagged {model_name} 10-Fold CV & Grid Search Parameters')

            self.dump_result_log()

    def run_experiment_3(self):
        self.current_experiment=3
        self.start_new_result_log(f'YOLO Comparison- Training Our Models on Synthetic Data', 'experiment_3')
        self.load_experiment_set(2,'yolo')
        for model in ['lg', 'rf', 'SVC_rbf', 'SVC_poly', 'LinearSVC']:
            self.select_model(model)
            model_name = self.get_model_name()
            self.load_grid_search_parameters()
            self.train_model()
            results = self.evaluate_model(suppress_report=True)
            result_title = f'{model_name} Trained on Yolo Synthetic Dataset'
            self.write_to_result_log(results, result_title)
            self.direct_plot_results(results, result_title)
        self.dump_result_log()

    def run_experiment_4(self):
        self.current_experiment=4
        self.load_experiment_set()
        feature_names = self.data['full']['X'].columns
        # logistic regression
        self.start_new_result_log(f'Feature Importances- Logistic Regression', 'experiment_4')
        self.select_model('lg')
        self.train_model('full')
        # get coefficients
        coefficients = self.model.coef_[0]
        intercepts = self.model.intercept_
        coefficient_series = pd.Series(coefficients, index=feature_names).sort_values(ascending=False).to_string()
        intercept_series = pd.Series(intercepts, index=['Intercept']).sort_values(ascending=False).to_string()
        self.write_string_to_result_log(coefficient_series)
        self.write_string_to_result_log(intercept_series)
        self.dump_result_log()

        # random forest
        self.start_new_result_log(f'Feature Importances- Random Forest', 'experiment_4')
        self.select_model('rf')
        self.train_model('full')
        feature_importances = self.model.feature_importances_
        feature_importances_series = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False)
        output_file = os.path.join(self.results_path, f"rf_importances_test.txt")
        string_importances = feature_importances_series.to_string()
        self.write_string_to_result_log(string_importances)
        self.dump_result_log()

        #SVM(RBF)
        self.start_new_result_log(f'Feature Importances- SVM_RBF', 'experiment_4')
        self.select_model('SVC_rbf')
        self.train_model('full')
        importances = permutation_importance(self.model, self.data['full']['X'],self.data['full']['Y'], n_repeats=30, random_state=self.random_state)
        importances_series = pd.Series(importances.importances_mean, index=feature_names).sort_values(ascending=False).to_string()
        self.write_string_to_result_log(importances_series)
        self.dump_result_log()

        #SVM(Linear)
        self.start_new_result_log(f'Feature Importances- SVM_Linear', 'experiment_4')
        self.select_model('LinearSVC')
        self.train_model('full')
        importances = permutation_importance(self.model, self.data['full']['X'],self.data['full']['Y'], n_repeats=30, random_state=self.random_state)
        importances_series = pd.Series(importances.importances_mean, index=feature_names).sort_values(ascending=False).to_string()
        self.write_string_to_result_log(importances_series)
        self.dump_result_log()

    def run_all_experiments(self):
        self.run_experiment_1()
        self.run_experiment_2()
        self.run_experiment_3()
        self.run_experiment_4()

    # ============================
    # Plotting Utilities, Reporting
    # ============================
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

    def plot_confusion_matrix(self, cm, classes, title='Confusion Matrix', direct_write=False):
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
        plt.title(title, wrap=True, pad=10)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        if not direct_write:
            plt.show()
        else:
            target_path = self.results_path
            if self.current_experiment:
                sub_dir = f'experiment_{self.current_experiment}'
                target_path = os.path.join(target_path, sub_dir)
            write_name=self.filter_title_for_write(title)
            figures_dir = os.path.join(target_path, 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            save_path = os.path.join(figures_dir, f"{write_name}.png")
            plt.savefig(save_path)
            plt.close()

    def plot_roc_curve(self, fpr, tpr, roc_auc, title='ROC Curve', direct_write=False):
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.025])
        plt.xlabel('False Positive Rate',labelpad=10)
        plt.ylabel('True Positive Rate',labelpad=5)
        plt.subplots_adjust(top=0.85)
        plt.title(title,wrap=True, pad=25)
        plt.legend(loc="lower right")
        if not direct_write:
            plt.show()
        else:
            target_path = self.results_path
            if self.current_experiment:
                sub_dir = f'experiment_{self.current_experiment}'
                target_path = os.path.join(target_path, sub_dir)
            write_name=self.filter_title_for_write(title)
            figures_dir = os.path.join(target_path, 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            save_path = os.path.join(figures_dir, f"{write_name}.png")
            plt.savefig(save_path)
            plt.close()

    def direct_plot_results(self, results, title):
        cm = results.get('cm')
        fpr = results.get('fpr')
        tpr = results.get('tpr')
        roc_auc = results.get('roc_auc')
    
        classes = self.dataPreprocessor.classes 
        if fpr is not None and tpr is not None and roc_auc is not None:
            self.plot_roc_curve(
                fpr=fpr,
                tpr=tpr,
                roc_auc=roc_auc,
                title=f'{title} ROC Curve',
                direct_write=True
            )
        if cm is not None and classes is not None:
            self.plot_confusion_matrix(
                cm=cm,
                classes=classes,
                title=f'{title} CM',
                direct_write=True
            )

    def filter_title_for_write(self, title):
        write_name=title.replace('Cross Validation', 'CV')
        write_name=write_name.replace('Dataset', '')
        write_name=write_name.replace('&', '')
        write_name=write_name.replace('Original', 'OG')
        write_name=write_name.replace('Grid Search Best', 'gridSearch')
        write_name=write_name.replace('Parameters','params')
        write_name=write_name.strip()
        write_name=write_name.replace(' ','_')
        return write_name

    def start_new_result_log(self,title, sub_dir=''):
        self.result_log={'title':'','body':'', 'sub_dir':sub_dir}
        self.result_log['title']=title

    def write_to_result_log(self, results, subTitle):
        self.result_log['body'] = ''.join([self.result_log['body'], f"Experiment: {subTitle}\n--------------------------------------------------------\n", f'Accuracy: {str(results["accuracy"])}\n', f'ROC_AUC: {str(results["roc_auc"])}\n\n', f'{results["cr"]}\n', f'Confusion matrix:\n{str(results["cm"])}\n\n\n\n'])

    def write_string_to_result_log(self, string, subTitle=''):
        self.result_log['body'] = ''.join([subTitle, self.result_log['body'], string])

    def dump_result_log(self):
        name = self.result_log['title']
        resultLog = self.result_log['body']
        resultLog= f'{name}\n========================================================\n========================================================\n\n'+resultLog
        output_file = os.path.join(self.results_path, f"{name}.txt")
        if self.current_experiment:
            sub_dir = f'experiment_{self.current_experiment}'
            sub_dir_path = os.path.join(self.results_path, sub_dir)
            os.makedirs(sub_dir_path, exist_ok=True)
            output_file = os.path.join(sub_dir_path, f"{name}.txt")
        with open(output_file, 'w') as f:
            f.write(resultLog)
        print(resultLog[:-3])
        print(f"Results saved to {output_file}")
        self.result_log={'title':'','body':'', 'sub_dir':''}

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

    def plot_SVM_decision_boundaries(self):
        self.load_experiment_set()
        for kernel_name in ['RBF', 'poly']:
            self.select_model(f'SVC_{kernel_name.lower()}')
            self.load_grid_search_parameters()
            pca = PCA(n_components=2, random_state=self.random_state)
            self.data['train']['X_reduced'] = pca.fit_transform(self.data['train']['X'].values)
            self.train_model(use_reduced_x=True)

            fig, ax = plt.subplots(figsize=(6, 4))

            #window buffers are hardcoded for now- bad
            x_min, x_max = self.data['train']['X_reduced'][:, 0].min() - 1, self.data['train']['X_reduced'][:, 0].max() + 1
            y_min, y_max = self.data['train']['X_reduced'][:, 1].min() - 1, self.data['train']['X_reduced'][:, 1].max() + 1
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            # color mesh
            DecisionBoundaryDisplay.from_estimator(
                self.model,
                self.data['train']['X_reduced'],
                response_method="predict",
                plot_method="pcolormesh",
                alpha=0.3,
                cmap='cool',
                ax=ax
            )
            # boundary lines
            DecisionBoundaryDisplay.from_estimator(
                self.model,
                self.data['train']['X_reduced'],
                response_method="decision_function",
                plot_method="contour",
                levels=[-1, 0, 1],
                colors=["k", "k", "k"],
                linestyles=["--", "-", "--"],
                ax=ax
            )

            support_vectors_scaled = self.model.named_steps['clf'].support_vectors_
            support_vectors = self.model.named_steps['scaler'].inverse_transform(support_vectors_scaled)
            # support vectors
            ax.scatter(
                support_vectors[:, 0],
                support_vectors[:, 1],
                s=100,
                facecolors="none",
                edgecolors="k",
            )
            # all points
            scatter = ax.scatter(
                self.data['train']['X_reduced'][:, 0],
                self.data['train']['X_reduced'][:, 1],
                c=self.data['train']['Y'].values,
                edgecolors='k',
                cmap='cool',
                s=30,
            )

            handles, labels = scatter.legend_elements()
            ax.legend(
                handles=[handles[0], handles[1], plt.Line2D([], [], marker='o', linestyle='', 
                                                            markeredgecolor='k', markerfacecolor='none', 
                                                            markersize=10, label='Support Vectors')],
                labels=['HC', 'PwD'],
                loc="upper right"
            )
            ax.set_title(f'SVM({kernel_name}) Support Vectors (PCA Reduced)', wrap=True)
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            plt.tight_layout()
            plot_path = os.path.join(self.results_path, 'SVM_decision_boundary_figures')
            os.makedirs(plot_path, exist_ok=True)
            final_path = os.path.join(plot_path, f"SVM ({kernel_name})_support_vectors_pca.png")
            plt.savefig(final_path)
            plt.close()
            print(f"PCA plot of support vectors saved to {plot_path}")

    def report_support_vectors(self, dataSet):
        import numpy as np
        vector_doc=''
        support_indices=self.model.named_steps['clf'].support_
        support_vector_labels=self.data[dataSet]['Y'].iloc[support_indices]
        support_vectors_scaled = self.model.named_steps['clf'].support_vectors_
        support_vectors = self.model.named_steps['scaler'].inverse_transform(support_vectors_scaled)
        vector_magnitudes = np.linalg.norm(support_vectors, axis=1)
        labeled_vectors = list(zip(support_vector_labels, vector_magnitudes))
        labeled_vectors_sorted = sorted(labeled_vectors, key=lambda x: x[1], reverse=True)

        output_file = os.path.join(self.results_path, f"SVC_rbf_SupportVectors.txt")

        with open(output_file, 'w') as f:
            for label, magnitude in labeled_vectors_sorted:
                f.write(f"{label}: {magnitude:.4f}\n")

    # ============================
    # Grid Search & Hyperparameters
    # ============================
    def load_grid_search_parameters(self):
        model_name = self.get_model_name()
        if HyperparameterConfig.check_for_model_hyperparameters(self.model):
            best_params = HyperparameterConfig.get_model_hyperparameters(self.model)
            self.select_model(model_name, **best_params)
        else:
            #print(f'No best params found for model: {model_name}')
            #awaitUser=input('Start new grid search?')
            #if 'y' in awaitUser.tolower():
            self.grid_search_params(HyperparameterConfig.parameter_grids[model_name])
    
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
            error_score=float('nan'),
            verbose=1
        )
        grid_search.fit(X, Y)
        print("Best Parameters found by Grid Search:", grid_search.best_params_)
        print("Best Score found by Grid Search:", grid_search.best_score_)

        # Update the model with best estimator
        self.model = grid_search.best_estimator_
        HyperparameterConfig.store_model_hyperparameters(self.model)

    def get_model_hyperparameter_grid(self):
        return HyperparameterConfig.parameter_grids[type(self.model).__name__]
    
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
    print('CS:4980 GUT METAGENOME ML PROJECT')
    print('Preparing to run all experiments (1-4).')
    print('')
    print('[WARNING]:')
    print('This program may first need to perform pre-processing on the datasets.')
    print('The full experimental pipeline may take up to 15 minutes to run.')
    print('')
    user_choice = input('Proceed? [Y/N]\n')
    if 'y' in user_choice.lower():
        print('Running all experiments, please wait.')
        interface.run_all_experiments()
        interface.plot_SVM_decision_boundaries()
    else:
        print('Exiting.')

