> **🚧 Work in Progress:** I am currently making modifications on this project. This and the other files may be updated as improvements are made to the model and documentation.

# Machine Learning Analysis of Bank Fraud

This project investigates fraudulent bank transactions using machine learning. 
The goal is to build a pipeline that processes raw data, creates meaningful features, 
and trains models to predict fraudulent transactions. The pipeline uses Weights & Biases (WandB) 
for experiment tracking and data management.

Link to [Weights and Biases Project](https://wandb.ai/lhan122-student/credit_card_fraud?nw=nwuserlhan122)

Link to [Tableau Dashboard](https://public.tableau.com/app/profile/leslie.hanson/viz/Model_dashboard_17341265980230/PublishedDashboard?publish=yes)

## Data
This project uses a credit card transactions dataset available on Kaggle and Hugging Face at:

https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset

https://huggingface.co/datasets/pointe77/credit-card-transaction

The dataset includes 24 columns made up of:
•	Transaction Information: Transaction date and time, merchant details, geographical coordinates of both the transaction and of the merchant, population of the city the transaction occurred in, transaction number, and a Unix timestamp of the transaction.
•	Personal Identifiable Information: Credit card number, first and last name, gender, complete address, job, and date of birth.
•	Fraud label: This label allows for a supervised learning model to be used. 

## Steps
1. **Setup Environment**  
   Use the `environment.yml` file to create an environment in the CLI by executing `conda env create -f environment.yml`.
   Once created, activate it by typing `conda activate cc_fraud`.
2. **Kaggle API Key**  
   Ensure you have your Kaggle API key set up to download the dataset. Place it in `~/.kaggle/kaggle.json`.
3. **Weights & Biases API Key**  
   Log in to WandB and set up your API key.
4. **Run main.py** Downloads data and tests an already selected and trained XGBoost model.

## Files 
### Root Folder
- environment.yml - yaml file to set up conda environment
- main.py - main script that runs a test pipeline on preselected and trained XGBoost model

### Data Folder
 - get_data.py - script that downloads data directly from Kaggle and uploads it to WandB.

#### Data Subfolders
**data_understanding**  
- Data_Understanding.ipynb - has a jupyter notebook covering the entire data exploration process.
**data_preparation**
- clean_data.py - script that performs basic cleaning like dropping columns and changing column names.
- cust_loc_map.py - script that makes a python df of unique city and state values and requests the latitude and longitude
values of these locations using an api. Due to the long run time, it is best to have this created before feature creation
- feature_creation.py - script that creates new features for the dataset like adding the customer
latitude and longitude values

### Model Folder
- train_test_split.py - script that splits the final dataset into train and test sections
- train_all_models.py - trains 3 models on train data. Uploads metrics and model to Weights and Basis.
- train_parameter_tuning.py - after the model (XGBoost) has been selected, script set up for a Weights and 
Biases sweep to run in CLI.
- config.yaml - a config file for wandb sweep, used in train_parameter_tuning.py
- shared_utils.py - script with pre-made functions used in other scripts.
- test_xgboost.py - tests the selected pre-trained xgboost model.



