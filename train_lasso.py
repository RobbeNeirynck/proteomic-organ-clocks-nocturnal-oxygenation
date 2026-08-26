import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedKFold, cross_val_score


DATA_FILE = "data/synthetic_example_proteomics.csv"
ORGAN_MAP_FILE = "data/gtex_organ_protein_map.csv"
OUTPUT_FILE = "lasso_test_predictions.csv"
REPETITIONS = 10
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


def main():
    data = pd.read_csv(DATA_FILE)
    organ_map = pd.read_csv(ORGAN_MAP_FILE)
    alphas = np.logspace(-5, 1, 100)
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

                mean_scores = []
                for alpha in alphas:
                    model = Lasso(alpha=alpha, max_iter=1000, random_state=42)
                    scores = cross_val_score(
                        model,
                        X_train,
                        y_train,
                        cv=5,
                        scoring="neg_mean_absolute_error",
                        n_jobs=1,
                    )
                    mean_scores.append(np.mean(scores))

                best_alpha = alphas[np.argmax(mean_scores)]
                model = Lasso(alpha=best_alpha, max_iter=1000, random_state=42)
                model.fit(X_train, y_train)

                result = test_data[["visit_id", "Age", "Sex_F", "Round"]].copy()
                result["Predicted_Age"] = model.predict(X_test)
                result["Model_Type"] = "LASSO"
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
