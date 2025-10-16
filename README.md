# [SAITS](https://github.com/WenjieDu/SAITS) Model for Imputation of Smooth Pursuit Eye Movements

**Imputation of Missing Data in Smooth Pursuit Eye Movements Using a Self-Attention-based Deep Learning Approach**
M Bejani, G Perez-de-Arenaza-Pozo, JD Arias-Londoño, JI Godino-Llorente 
arXiv preprint [arXiv:2506.00545](https://arxiv.org/abs/2506.00545) 

## Overview

This repository implements the [SAITS](https://github.com/WenjieDu/SAITS) model for the imputation of smooth pursuit eye movement data. The project compares the performance of the [SAITS](https://github.com/WenjieDu/SAITS) model against various other imputation methods, including **KNN**, **PCHIP**, and **SSA**. 

## Repository Structure

The repository is organized into the following main directories:

- **pipeline_all/**: Contains the main code for fitting the [SAITS](https://github.com/WenjieDu/SAITS) model to the data, comparing it with other imputation methods, and computing various performance metrics.
- **data_preprocessing/**: Includes all preprocessing scripts necessary for preparing the data for use in the pipeline_all directory.
- **Autoencoder/**: Contains the code for training and testing the autoencoder, with the best weights saved for use in the pipeline_all directory.

