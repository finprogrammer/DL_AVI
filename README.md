CNN and Transformer Model with Optuna Optimization
This repository contains code for training a Convolutional Neural Network (CNN) and Vision Transformer model for a classification task using PyTorch. The model is optimized using Optuna, a hyperparameter optimization framework. It also has integration for TensorBoard.
Usage
Modify the configuration files (train_config.yaml and debug_config.yaml) to adjust hyperparameters and other settings according to your requirements.
Files
trainer.py/ Dinov2: Main script for training the CNN model using Optuna optimization.
network.py: Contains the CNN architecture definition.
dataloader.py: Contains the data preprocessing.
extract_gradcam.py: Script for extracting Grad-CAM visualizations from the trained model.
train_config.yaml: Configuration file for training settings.
debug_config.yaml: Configuration file for debugging settings.
misclassification.py: Get misclassified samples.
