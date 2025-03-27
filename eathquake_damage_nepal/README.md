# Earthquake Damage Prediction Project
__Predicting building damage levels (1-3) after earthquakes using geographic, structural, and material features.__

This project predicts the damage grade (1: low, 2: medium, 3: high) of buildings after earthquakes using machine learning. The goal is to aid disaster response by prioritizing inspections of heavily damaged structures.

### Key Achievements
- Achieved `F1 Micro` = __0.7403__ (__30.9%__ improvement over baseline).
- Optimized `LightGBM` with hyperparameter tuning.
- Full pipeline: preprocessing → training → evaluation → deployment.

## Project Overview
This project aims to predict building damage levels after earthquakes using machine learning. The damage grades are:
- 1: Low damage
- 2: Medium damage  
- 3: Almost complete destruction

## Technical Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-389636?style=for-the-badge&logo=lightgbm&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

- **Python** (Primary language)
- **LightGBM** (Main ML model)
- **Pandas/Numpy** (Data processing)
- **Scikit-learn** (Metrics and utilities)
- **Jupyter Notebooks** (Analysis and prototyping)

## Project Structure
project-root/
│
├── data/
│ ├── test_values.csv
│ └── train_labels.csv
│ └── train_values.csv
│
├── earth_quake_damage_nepal.ipynb


## Data Preprocessing

### Handling High Cordinality (geo level ids)

```python
# Impute high cordiniality geo_level data
train_mean = train_values["damage_grade"].mean()

# For geo_1
geo1_mean = train_values.groupby("geo_level_1_id")["damage_grade"].mean().to_dict()
train_values["geo_level_1_id"] = train_values["geo_level_1_id"].map(geo1_mean).fillna(train_mean)
test_values["geo_level_1_id"] = test_values["geo_level_1_id"].map(geo1_mean).fillna(train_mean)

# For geo_2
geo2_mean = train_values.groupby("geo_level_2_id")["damage_grade"].mean().to_dict()
train_values["geo_level_2_id"] = train_values["geo_level_2_id"].map(geo2_mean).fillna(train_mean)
test_values["geo_level_2_id"] = test_values["geo_level_2_id"].map(geo2_mean).fillna(train_mean)

# For geo_3
geo3_mean = train_values.groupby("geo_level_3_id")["damage_grade"].mean().to_dict()
train_values["geo_level_3_id"] = train_values["geo_level_3_id"].map(geo3_mean).fillna(train_mean)
test_values["geo_level_3_id"] = test_values["geo_level_3_id"].map(geo3_mean).fillna(train_mean)
```

### Feature Engineering
```python
# Put age gap
age_gap = train_values["age"].quantile(0.99)
train_values["age"] = train_values["age"].clip(upper=age_gap)
# on test data too
test_values["age"] = test_values["age"].clip(upper=age_gap)
```

### Handling Categorical Features
```python
#  convert categorical columns to 'category' dtype
categorical_cols = [
    "land_surface_condition", "foundation_type", "roof_type",
    "ground_floor_type", "other_floor_type", "position", "legal_ownership_status",
    "plan_configuration"
]

for col in categorical_cols:
    train_values[col] = train_values[col].astype("category")

# Do the same with test data
for col in categorical_cols:
    test_values[col] = test_values[col].astype("category")
```

## Model Training

### LightGBM Configuration

```python
params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "booting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_jobs": -1,
    "verbose": -1
}
```

### Training Process
```python
# Train
model = lgb.train(
    params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=100
)
```

### Evaluation Metrics
----------- --------
| Metric   |  Score |
------------ -------
| F1 Micro | 0.7403 |
----------- --------
| Baseline | 0.5657 |
------------ -------

### Error Analysis
Severe errors (true=3 → pred=1): `235`

## Future Improvement
- Experiment with ordinal regression models
- Add feature importance analysis
- Create Flask API for predictions

<div align="center" style="margin-top: 10px;">

![Optimized with LightGBM](https://img.shields.io/badge/Optimized_with-LightGBM-389636?style=for-the-badge&logo=lightgbm&logoColor=white)
![Built with Jupyter](https://img.shields.io/badge/Built_with-Jupyter_Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

## 📝 Lessons Learned
✅ Ordinality Matters: Converting labels to [0,1,2] for LightGBM was critical.
✅ Geo Features Dominate: Geographic region (geo_level_3_id) was the top predictor.
✅ Fine-Tuning: Using `tuna` to fintune the LightGBM model can significally improve model performance.

_Last updated: 3/27/2025_
