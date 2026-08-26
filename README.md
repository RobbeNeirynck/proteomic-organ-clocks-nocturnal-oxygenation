# Proteomic organ clocks for nocturnal oxygenation

Code accompanying the manuscript *“Nocturnal oxygenation and the circulating proteome: metabolic reprogramming and accelerated organ ageing in the general population”* by Neirynck et al.

This repository contains the algorithms used to construct the organ-age clocks in the manuscript:

- `train_lightgbm.py`: trains the LightGBM clocks.
- `train_lasso.py`: trains the LASSO clocks.
- `calculate_age_gaps.R`: calculates LOESS-adjusted age gaps from the LightGBM test predictions.

The `data` folder contains the organ–protein map from the GTEx analysis together with a synthetic test proteomic dataset (`synthetic_example_proteomics.csv`).

## Run

```bash
pip install lightgbm numpy optuna pandas scikit-learn
python train_lightgbm.py
python train_lasso.py
```

The age-gap calculation requires the R package `dplyr`:

```bash
Rscript calculate_age_gaps.R
```
