# Robot Object Distance Estimation from Vision

Estimate the distance between a robotic manipulator and an object using DINOv2 visual embeddings and classical machine learning regression models.

## Features

- DINOv2 visual embeddings
- Gymnasium Fetch environment
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- MLP Regressor
- Regression evaluation (MAE, RMSE, R², MAPE)

## Pipeline

Environment → Image → DINOv2 → Embeddings → Regression Model → Distance Prediction

## Results

| Model | MAE | RMSE | R² |
|------|------|------|------|
| Random Forest | ... | ... | ... |
| XGBoost | ... | ... | ... |

## Requirements

```bash
pip install -r requirements.txt
