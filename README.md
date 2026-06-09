# Traffic Demand Prediction using Stacked Ensemble Learning

## Overview

This project focuses on predicting traffic demand using machine learning techniques and advanced feature engineering. The solution was designed using only the allowed input features:

* `geohash`
* `day`
* `timestamp`
* `RoadType`
* `LargeVehicles`
* `Weather`

The project combines:

* feature engineering
* target encoding
* ensemble learning
* stacked regression models

to achieve strong prediction performance on structured traffic data.

---

# Project Objective

The objective of this project is to accurately predict traffic demand by extracting meaningful spatial, temporal, and environmental patterns from structured tabular data.

The solution emphasizes:

* robust preprocessing
* advanced feature engineering
* ensemble learning
* generalization on unseen data
* reduction of overfitting

---

# Dataset Features Used

The following features were used for model training:

| Feature       | Description                      |
| ------------- | -------------------------------- |
| geohash       | Encoded geographical region      |
| day           | Day identifier                   |
| timestamp     | Time information                 |
| RoadType      | Type of road                     |
| LargeVehicles | Presence/count of large vehicles |
| Weather       | Weather conditions               |

---

# Data Preprocessing

The preprocessing pipeline includes:

* Standardizing column names
* Handling missing values
* Parsing timestamp fields
* Converting categorical columns into consistent formats
* Protecting the feature matrix from:

  * NaN values
  * infinite values
  * invalid numerical entries

---

# Feature Engineering

Feature engineering was one of the most important components of the solution.

## Time-Based Features

Extracted:

* `hour`
* `minute`

Generated cyclical features:

* `hour_sin`
* `hour_cos`
* `day_sin`
* `day_cos`

Generated binary indicators:

* `is_weekend`
* `is_weekday`
* `is_rush_hour`
* `is_night`

These features help capture recurring traffic patterns across:

* daily cycles
* weekly cycles
* rush hours
* nighttime traffic

---

## Frequency Encoding

Frequency encoding was applied on:

* `geohash`
* `RoadType`
* `LargeVehicles`
* `Weather`

This helps models understand the distribution and importance of categorical variables.

---

## Smoothed Target Encoding

Smoothed target encoding was applied for:

* `geohash`
* `RoadType`
* `LargeVehicles`
* `Weather`

This technique helps the model learn relationships between categorical values and traffic demand while reducing overfitting.

---

## Interaction Features

Created interaction features:

* `geo_road_te`
* `geo_weather_te`
* `road_weather_te`
* `road_large_te`
* `time_weather`
* `time_road`
* `rush_weather`

These features help models capture complex relationships between:

* location
* weather
* road conditions
* traffic timing

---

# Models Used

The project uses multiple ensemble-based regression models.

## Base Models

* ExtraTreesRegressor
* RandomForestRegressor
* GradientBoostingRegressor
* HistGradientBoostingRegressor
* LightGBM Regressor *(optional)*
* CatBoost Regressor *(optional)*

### Why Ensemble Models?

Tree-based ensemble models:

* capture non-linear relationships
* handle structured data effectively
* automatically learn feature importance
* improve prediction robustness

---

# Stacking and Ensembling

A stacked ensemble pipeline was implemented.

## Process

1. Train each base model using 5-Fold Cross Validation
2. Generate out-of-fold predictions
3. Use predictions as inputs for a meta learner
4. Train a Ridge Regression meta-model
5. Blend final predictions with the best-performing base model

## Meta Learner

* Ridge Regression

### Benefits of Stacking

Stacking helps:

* combine strengths of multiple models
* reduce individual model weaknesses
* improve overall prediction accuracy
* enhance generalization performance

---

# Cross Validation

## Method Used

* 5-Fold Cross Validation

### Purpose

Cross-validation helps:

* reduce overfitting
* improve reliability
* ensure stable model performance

---

# Technologies and Libraries

## Programming Language

* Python

## Libraries Used

* pandas
* numpy
* scikit-learn
* lightgbm *(optional)*
* catboost *(optional)*

---

# Project Structure

```bash
project/
│
├── train.csv
├── test.csv
├── submission.csv
├── solution.py
├── README.md
│
└── dataset/
```

---

# Output

The final predictions are saved in:

```bash
submission.csv
```

## Output Format

| Index | demand |
| ----- | ------ |

---

# How to Run

## Install Dependencies

```bash
pip install pandas numpy scikit-learn lightgbm catboost
```

## Run the Project

```bash
python solution.py
```

---

# Key Highlights

* Advanced feature engineering
* Target encoding
* Interaction features
* Ensemble learning
* Stacked regression pipeline
* Strong generalization capability
* Robust against overfitting

---

# Final Observations

The major performance improvements came from:

* feature engineering
* target encoding
* interaction-based learning
* ensemble models
* stacking multiple regressors

Instead of deep learning, the solution focuses on extracting maximum information from structured tabular data, which is generally more effective for traffic prediction tasks of this type.

The final pipeline was designed to:

* capture temporal traffic patterns
* capture spatial traffic behavior
* generalize effectively
* improve prediction robustness and stability
