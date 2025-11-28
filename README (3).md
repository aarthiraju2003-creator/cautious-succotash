# Advanced Time Series Forecasting with Seq2Seq Attention

# Project Overview
This project implements an advanced multivariate time series forecasting framework using a Sequence-to-Sequence (Seq2Seq) model with additive attention. The goal is to model complex temporal patterns in the Appliances Energy dataset and generate multi-step forecasts with improved interpretability.

The project also includes a baseline LSTM model for comparison, detailed evaluation metrics, attention visualizations, and an organized output structure suitable for academic submissions, research experiments, or production-style pipelines.

All code is optimized for execution in Google Colab and supports reproducible training, evaluation, and artifact generation.

# Key Features
- End-to-end multivariate time series forecasting workflow.
- Encoder–Decoder LSTM architecture with additive attention.
- Baseline LSTM for comparative performance analysis.
- Fully configurable preprocessing, scaling, windowing, and dataloaders.
- Multi-step forecasting (12-step horizon by default).
- Visualizations of prediction curves, attention heatmaps, and loss curves.
- Training and validation loss tracking for model convergence analysis.
- All outputs exported automatically into a structured folder.
- Ready for academic submission, research, or GitHub deployment.

# Dataset
The project uses the Appliances Energy Consumption dataset.

Dataset includes:
- Timestamp
- Appliances energy use (target variable)
- Multiple temperature and humidity sensor readings
- Weather data collected from a nearby station

The dataset is preprocessed by parsing timestamps, handling missing values, selecting numeric features, and scaling using StandardScaler.

## Project Structure
A suggested repository structure:

```
project/
│
├─ notebooks/
│   └─ time_series_attention.ipynb
│
├─ src/
│   ├─ data_loader.py
│   ├─ preprocessing.py
│   ├─ model_attention.py
│   ├─ model_lstm.py
│   ├─ train.py
│   └─ utils.py
│
├─ project_outputs/
│   ├─ metrics.json
│   ├─ predictions.csv
│   ├─ true_values.csv
│   ├─ loss_curve.png
│   ├─ prediction_sample.png
│   ├─ attention_heatmap.png
│   ├─ seq2seq_attention_model.pth
│   └─ scaler.save
│
└─ README.md
```

# Model Architecture

# Seq2Seq with Additive Attention
- Encoder: Processes the input sequence using LSTM.
- Decoder: Generates each forecast step using LSTMCell.
- Additive Attention: Weighs encoder outputs to compute context vectors that highlight relevant historical patterns.
- Output: Multi-step regression prediction for energy usage.

# Baseline LSTM
- A simpler LSTM model that uses only the final hidden state to predict future values.
- Serves as a performance baseline to evaluate the benefit of attention.

# Workflow
1. Load the dataset and parse timestamps.
2. Clean missing values using forward/backward fill.
3. Scale features and target using StandardScaler.
4. Convert time series data into supervised windows (sequence → target).
5. Train Seq2Seq Attention model.
6. Train baseline LSTM model.
7. Evaluate both models using MAE, RMSE, and MAPE.
8. Visualize prediction curves, loss curves, and attention heatmaps.
9. Save all artifacts automatically to a designated folder.

# Training Details
- Teacher forcing ratio supports faster training for Seq2Seq models.
- Gradient clipping avoids exploding gradients.
- Early stopping prevents overfitting.
- GPU usage is automatic if available in Colab.
- Typical training time: 10–20 epochs depending on hardware.

# Evaluation Metrics
The following metrics are computed for both models:
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error

Metrics are saved as `project_outputs/metrics.json`.

# Visualizations
This project generates the following plots:
- Multi-step forecast comparison (true vs. predicted values)
- Attention heatmap showing which past timesteps the model focuses on
- Training and validation loss curves

All visual outputs are saved as PNG files.

# Output Folder
All important artifacts are saved inside:

```
/content/project_outputs/
```

This includes:
- Model weights
- Scaler
- Predictions CSV
- True values CSV
- Loss curve
- Prediction sample
- Attention heatmap
- Metrics JSON

The entire folder can be zipped for download when running in Google Colab.

# Running on Google Colab
1. Upload the dataset `energydata_complete.csv`.
2. Copy the provided notebook code into a Colab notebook.
3. Update the file path if necessary.
4. Run all cells in order.
5. Colab will generate all outputs and save them into `project_outputs`.

# Requirements
```
torch
numpy
pandas
matplotlib
scikit-learn
optuna
joblib
```

These packages are preinstalled or installable in Colab.

# Future Enhancements
- Implement Transformer-based architecture.
- Add multivariate VAR or N-BEATS baseline.
- Integrate Optuna for extensive hyperparameter tuning.
- Build an interactive dashboard for predictions.
- Wrap the model into a real-time inference API.

# Author
Created by Nandhini as part of an advanced deep learning project focusing on Seq2Seq models, attention mechanisms, time series forecasting, and model evaluation.
