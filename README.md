# Human Activity Recognition using CNN–BiLSTM

This project implements a deep learning–based Human Activity Recognition (HAR) system using smartphone inertial sensor data.

## Dataset
- UCI Human Activity Recognition Using Smartphones
- Accelerometer + Gyroscope signals
- 6 activities

## Model
- 1D CNN for local feature extraction
- Bidirectional LSTM for temporal modeling
- Z-score normalization
- Early stopping

## Results
- Validation Accuracy: ~93%
- Test Accuracy: ~90%+

## Files
- `notebooks/`: training and preprocessing notebook
- `models/`: trained CNN–BiLSTM model
- `main.py`: inference / app entry point

## How to Run
```bash
pip install -r requirements.txt
