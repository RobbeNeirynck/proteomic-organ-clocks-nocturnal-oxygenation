import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


DATA_FILE = "data/synthetic_example_proteomics.csv"
ORGAN_MAP_FILE = "data/gtex_organ_protein_map.csv"
OUTPUT_FILE = "lightgbm_test_predictions.csv"
REPETITIONS = 10
OPTUNA_TRIALS = 60
ORGANS = [
    "Conventional", "Heart", "Lung", "Kidney", "Artery", "Brain",
    "Adipose", "Liver", "Muscle", "Pancreas", "Immune", "Intestine",
]


def outer_folds(data, repetition):
    age_quantile = pd.qcut(data["Age"], q=5, labels=False)
    strata = age_quantile.astype(str) + "_" + data["Sex_F"].astype(str)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=repetition)
    return [
        data.iloc[test_index]["visit_id"].to_numpy()
        for _, test_index in cv.split(data["visit_id"], strata)
    ]


def tune_lightgbm(X, y, trials):
    def objective(trial):
        parameters = {
            "objective": "regression",
            "metric": "l1",
            "verbose": -1,
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 600),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 10, log=True),
            "feature_fraction": 0.75,
            "n_jobs": 1,
            "random_state": 42,
        }
        results = lgb.cv(
            parameters,
            lgb.Dataset(X, label=y),
            num_boost_round=5000,
            nfold=5,
            stratified=False,
            shuffle=True,
            seed=42,
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        return np.min(results["valid l1-mean"])

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=trials, n_jobs=1)
    return study.best_params


def main():
    data = pd.read_csv(DATA_FILE)
    organ_map = pd.read_csv(ORGAN_MAP_FILE)
    predictions = []

    for repetition in range(1, REPETITIONS + 1):
        folds = outer_folds(data, repetition)

        for organ in ORGANS:
            proteins = organ_map.loc[organ_map["Organ"] == organ, "SeqID"].str.replace("-", "_").tolist()
            features = ["Sex_F"] + proteins

            for fold_number, test_ids in enumerate(folds, start=1):
                test_data = data[data["visit_id"].isin(test_ids)]
                train_data = data[~data["visit_id"].isin(test_ids)]

                X_train = train_data[features]
                y_train = train_data["Age"]
                X_test = test_data[features]

                X_fit, X_valid, y_fit, y_valid = train_test_split(
                    X_train, y_train, test_size=0.1, random_state=42
                )
                best_parameters = tune_lightgbm(X_fit, y_fit, OPTUNA_TRIALS)

                model = lgb.LGBMRegressor(
                    objective="regression",
                    n_jobs=1,
                    random_state=42,
                    metric="l1",
                    verbose=-1,
                    n_estimators=5000,
                    **best_parameters,
                )
                model.fit(
                    X_fit,
                    y_fit,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric="l1",
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )

                result = test_data[["visit_id", "Age", "Sex_F", "Round"]].copy()
                result["Predicted_Age"] = model.predict(X_test)
                result["Model_Type"] = "LightGBM"
                result["Repetition_Number"] = repetition
                result["Fold_Number"] = fold_number
                result["Organ_Type"] = organ
                predictions.append(result)

    columns = [
        "Model_Type", "Repetition_Number", "Fold_Number", "Organ_Type",
        "visit_id", "Age", "Predicted_Age", "Sex_F", "Round",
    ]
    pd.concat(predictions, ignore_index=True)[columns].to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
