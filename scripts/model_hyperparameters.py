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
            }}

        @staticmethod
        def store_model_hyperparameters(model):
            data_path = HyperparameterConfig.get_data_path()
            parameters = model.get_params()
            model_name = type(model).__name__
            best_parameters_file = os.path.join(data_path, 'best_parameters.json')
            if os.path.exists(best_parameters_file):
                try:
                    best_parameters = json.load(open(best_parameters_file, 'r'))
                except:
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