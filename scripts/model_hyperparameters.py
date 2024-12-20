import os
import json

class HyperparameterConfig:
        parameter_grids = {
            "LogisticRegression": {
                "penalty": ["none", "l1", "l2", "elasticnet"],
                "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                "fit_intercept": [True, False],
                "solver": ["lbfgs", "liblinear", "sag", "saga", "newton-cg"],
                "max_iter": [100, 200, 500, 1000],
                "l1_ratio": [0.0, 0.5, 1.0],
                "tol": [1e-4, 1e-3, 1e-2],
                "random_state": [42]
            },
            "RandomForestClassifier": {
                "criterion": ["gini"],
                "n_estimators": [150, 175, 200, 225, 250],
                "max_depth": [None],
                "min_samples_split": [2, 3],
                "min_samples_leaf": [2],
                "max_features": [None],
                "bootstrap": [True],
                "n_jobs": [-1],
                "random_state": [42],
                "verbose": [0]
            },
            #entries below prefixed with 'clf__' because they are wrapped in pipeline
            "SVC_poly": {
                "clf__C": [0.75, 1, 1.25, 1.5],
                'clf__degree':[3],
                "clf__gamma": ["scale"],
                "clf__shrinking": [True, False],
                "clf__probability": [True],
                "clf__tol": [0.00005, 0.0001, 0.00025],
                "clf__max_iter": [25, 40, 50, 60, 75],
                "clf__class_weight": [None, "balanced"],
                "clf__decision_function_shape": ["ovo", "ovr"],
                "clf__random_state": [42]
            },
            "SVC_rbf": {
                "clf__C": [75,100,125,150],
                "clf__gamma": ["scale", "auto", 0.0001, 0.00005, 0.00015],
                "clf__shrinking": [True, False],
                "clf__probability": [True],
                "clf__tol": [0.00005, 0.0001, 0.00025],
                "clf__max_iter": [75, 100, 125, 150],
                "clf__class_weight": [None, "balanced"],
                "clf__decision_function_shape": ["ovo", "ovr"],
                "clf__random_state": [42]
            },
            #prefixed with 'clf__estimator__' because pipeline classifier is also wrapped
            "LinearSVC": {
                "clf__estimator__penalty": ["l1", "l2"],
                "clf__estimator__loss": ["hinge", "squared_hinge"],
                "clf__estimator__C": [0.005,0.0075, 0.01, 0.025, 0.05],
                "clf__estimator__dual": ['auto'],
                "clf__estimator__fit_intercept": [True, False],
                "clf__estimator__intercept_scaling": [25, 50, 75, 100],
                "clf__estimator__class_weight": [None, "balanced"],
                "clf__estimator__tol": [1e-6, 1e-5, 1e-4],
                "clf__estimator__max_iter": [85, 100, 115],
                "clf__estimator__random_state": [42]
            }}
            
        @staticmethod
        def store_model_hyperparameters(model):
            data_path = HyperparameterConfig.get_data_path()
            model_name = type(model).__name__
            if model_name=='Pipeline':
                model_name = HyperparameterConfig.get_pipeline_model_name(model)
                clf = model.named_steps['clf']
                model = clf.estimator if hasattr(clf, 'estimator') else clf
            parameters = model.get_params()
            best_parameters_file = os.path.join(data_path, 'best_parameters.json')
            if os.path.exists(best_parameters_file):
                try:
                    best_parameters = json.load(open(best_parameters_file, 'r'))
                except:
                    best_parameters = {}
            else:
                best_parameters = {}
            best_parameters[model_name] = parameters
            try:
                json.dump(best_parameters, open(best_parameters_file, 'w'), indent=4)
            except Exception as e:
                raise IOError(f"Failed to write best parameters to {best_parameters_file}.") from e

        @staticmethod
        def check_for_model_hyperparameters(model):
            data_path = HyperparameterConfig.get_data_path()
            model_name = type(model).__name__
            if model_name=='Pipeline':
                model_name = HyperparameterConfig.get_pipeline_model_name(model)
            best_parameters_file = os.path.join(data_path, 'best_parameters.json')
            if not os.path.exists(best_parameters_file):
                return False
            try:
                best_parameters = json.load(open(best_parameters_file, 'r'))
            except Exception as e:
                raise IOError(f"Failed to read parameters from {best_parameters_file}.") from e
            if model_name not in best_parameters:
                return False
            return True

        @staticmethod
        def get_model_hyperparameters(model):
            data_path = HyperparameterConfig.get_data_path()
            model_name = type(model).__name__
            if model_name=='Pipeline':
                model_name = HyperparameterConfig.get_pipeline_model_name(model)
            best_parameters_file = os.path.join(data_path, 'best_parameters.json')
            if not os.path.exists(best_parameters_file):
                raise FileNotFoundError(f"Trying to get {best_parameters_file} but file does not exist.")
            try:
                best_parameters = json.load(open(best_parameters_file, 'r'))
            except Exception as e:
                raise IOError(f"Failed to read best parameters from {best_parameters_file}.") from e
            if model_name not in best_parameters:
                raise KeyError(f"Hyperparameters for model '{model_name}' not found in {best_parameters_file}.")
            model_parameters = best_parameters[model_name]
            return model_parameters

        @staticmethod
        def get_data_path():
            script_dir = os.path.dirname(os.path.realpath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, '..'))
            data_processed_abs_path = os.path.join(project_root, 'data', 'processed')
            return data_processed_abs_path + os.sep
        
        @staticmethod
        def get_pipeline_model_name(model):
            clf = model.named_steps['clf']
            base = clf.estimator if hasattr(clf, 'estimator') else clf
            name = type(base).__name__
            if name == 'SVC':
                name += f'_{base.kernel}'
            return name