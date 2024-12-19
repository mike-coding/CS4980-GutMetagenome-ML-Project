class HyperparameterConfig:
        config = {
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
            }
            }