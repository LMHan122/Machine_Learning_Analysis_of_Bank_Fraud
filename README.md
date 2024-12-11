# Machine Learning Analysis of Bank Fraud

This project investigates fraudulent bank transactions using machine learning. The goal is to build a pipeline that processes raw data, creates meaningful features, and trains models to predict fraudulent transactions. The pipeline leverages Weights & Biases (WandB) for experiment tracking and data management.

## Steps

1. **Setup Environment**  
   Use the `environment.yml` file to create a Conda environment with the required dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate <env_name>

2. **Run main.py**


## Files 

### Root Folder
- environment.yml
- config.yaml - configuration file for Weights & Biases to make sure everything runs under the right project name.
- main.py - main script that runs all the other scripts
### Data Folder
 - get_data.py - script that downloads data directly from Kaggle and uploads it to WandB.
#### Data Subfolders
**data_understanding**  
- Data_Understanding.ipynb - has a python notebook covering the entire data exploration process.
**data_preparation**
- clean_data.py - script that performs basic cleaning like dropping columns and changing column names.
- cust_loc_map.py - script that makes a python df of unique city and state values and requests the latitude and longitude
values of these locations using an api. Due to the long run time, it is best to have this created before feature creation
- feature_creation.py - script that creats new features for the dataset like adding the customer
latitude and longitude values
### Model Folder
- train_test_split.py - script that splits the final dataset into train and test sections



## Prerequisites

1. **Kaggle API Key**  
   Ensure you have your Kaggle API key set up to download the dataset. Place it in `~/.kaggle/kaggle.json`.

2. **Weights & Biases API Key**  
   Log in to WandB and set up your API key:
   ```bash
   wandb login
