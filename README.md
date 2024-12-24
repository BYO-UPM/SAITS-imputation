README for SAITS Model for Imputation of Smooth Pursuit Eye Movement
Overview
This repository implements the SAITS (Smoothing and Imputation for Time Series) model for the imputation of smooth pursuit eye movement data. The project compares the performance of the SAITS model against various other imputation methods, including K-Nearest Neighbors (KNN), Piecewise Cubic Hermite Interpolating Polynomial (PCHIP), and Singular Spectrum Analysis (SSA). The pipeline is structured into several components for clarity and modularity.

Repository Structure
The repository is organized into the following main directories:

pipeline_all/: Contains the main code for fitting the SAITS model to the data, comparing it with other imputation methods, and computing various performance metrics.
data_preprocessing/: Includes all preprocessing scripts necessary for preparing the data for use in the pipeline_all directory.
Autoencoder/: Contains the code for training and testing the autoencoder, with the best weights saved for use in the pipeline_all directory.
