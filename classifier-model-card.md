> **🚧 Work in Progress:** This model card is currently being developed and may be updated as improvements are made to the model and documentation.


# Model Card: XGBClassifier_tested

## Model Overview
**Developed by:** Leslie Hanson  
**Model Date:** 2025-02-13  
**Model Version:** 1.0.2  
**Model Type:** Classification  
**License:** GNU GENERAL PUBLIC LICENSE  

## Intended Use
### Primary Intended Uses
This machine-learning model was developed as part of an academic project to analyze and detect fraudulent credit card transactions. It serves as a proof of concept to demonstrate the application of supervised learning techniques in fraud detection rather than a production-ready solution. The model is intended for educational and exploratory purposes, showcasing feature engineering, model evaluation, and performance analysis in a controlled dataset.

### Out-of-Scope Uses
- **Real-World Fraud Detection:** This model is a proof of concept and is not designed for deployment in a real banking system. It lacks the robustness, security measures, and regulatory compliance needed for production use.
- **Decision-Making on Financial Transactions:** The model should not be used to make actual fraud-related decisions, as it was trained on a limited dataset and may not generalize well to real-world fraud patterns.
- **Legal or Regulatory Compliance:** The model does not meet the standards required for financial fraud detection under industry regulations (e.g., PCI DSS, GDPR, or banking compliance laws).
- **High-Stakes Environments:** The model has not been stress-tested against adversarial attacks or sophisticated fraud techniques used in real financial systems.

## Datasets Used
This project uses the Credit Card Transactions Dataset available on [Kaggle](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset) and [Hugging Face](https://huggingface.co/datasets/pointe77/credit-card-transaction). 
The dataset contains approximately 1.3 million records spanning 18 months and includes 24 columns, covering:   
- **Transaction Information:** Transaction date and time, merchant details, geographical coordinates of both the transaction and of the merchant, population of the city the transaction occurred in, transaction number, and a Unix timestamp of the transaction.
- **Personal Identifiable Information:** Credit card number, first and last name, gender, complete address, job, and date of birth. 
- **Fraud label:** This label allows for a supervised learning model to be used. 

In the `data/get_data.py` script:
- Dataset was programmatically downloaded using the Kaggle API.
- Uploaded to Weights & Biases as an artifact.  
  -  **Artifact:** `credit_card_data`
  - **Version:** `0`

### Data Preparation  

#### <u>Initial Exploration (`data/data_understanding/Data_Understanding.ipynb`)</u>  
The data was first explored in a Jupyter Notebook using Python, primarily utilizing the **Pandas**, **Matplotlib**, and **Geopandas** libraries. The initial analysis included:  
- Verifying overall dataset statistics, such as the number of null values and total transaction records.  
- Examining each column to determine necessary preprocessing steps.  
- Compiling a list of data cleaning and transformation requirements.  

#### <u>Data Cleaning (`clean_data.py`)</u>  
The following modifications were made during the first stage of data preparation:  

- **Column Renaming:**  
  - `'trans_data_trans_time'` → `'trans_dt'` (for brevity).  

- **Column Removal:**  
  - An unnamed index column was dropped for redundancy.  
  - `'first'`, `'last'`, `'gender'` was dropped for bias mitigation.
  - `'unix_time'`, `'street'`, `'zip'`, and `'merch_zipcode'` were dropped as they were redundant; location and datetime processing used latitude, longitude, and transaction datetime instead.  
  - `'city_pop'` was dropped after determining the data was unreliable.  

- **Column Type Optimization:**  
  - `'category'`, `'merchant'`, `'job'`, `'is_fraud'`, `'state'`, and `'city'` were converted to categorical data types to reduce file size and improve processing speed.  

#### Feature Engineering (`feature_creation.py`)  
The following new features were created to enhance model performance:  

- **Time-Based Features:**  
  - Extracted an `hour` column from transaction timestamps.  
  - Created rolling window features to track:  
    - The number of transactions in the last **hour** (`trans_by_last_hr`) and **day** (`trans_by_last_day`) for each account.  
    - The total transaction amount (`amt_by_last_hr`, `amt_by_last_day`) over these time periods.  
  - Created new datetime-related columns: `date`, `year`, `month`, `day`, `week`, and `quarter`.  
  - Dropped `'trans_dt'` after extracting useful datetime components, as the original format was not model-compatible.  




- List and describe the main datasets used for training
- Include versions and dates of the datasets
- Note any data preprocessing or cleaning steps

### Training Data Characteristics
- Describe key characteristics of the training data
- Note any potential biases in the data
- Include relevant demographics or distributions

## Model Architecture
- Model architecture details
- Key hyperparameters
- Number of parameters
- Training infrastructure requirements

## Performance Evaluation
### Metrics
- List primary evaluation metrics used
- Include benchmark results
- Provide context for interpreting the metrics

### Testing Data
- Describe evaluation datasets
- Note any differences from training data
- Include testing methodology

## Limitations and Biases
### Known Limitations
- Technical limitations
- Domain-specific limitations
- Performance boundaries

### Bias and Fairness Assessments
- Results of bias evaluations
- Fairness metrics across different groups
- Identified disparities in performance

## Ethical Considerations
- Potential societal impacts
- Privacy considerations
- Environmental impact
- Recommendations for responsible deployment

## Maintenance
### Updates and Maintenance Plan
- Update frequency
- Monitoring approach
- Feedback collection process

### Version History
- List of previous versions
- Notable changes
- Deprecation schedule (if applicable)

## Additional Information
### Citations
- Related research papers
- Relevant documentation
- Supporting materials

### Contact Information
- Maintenance team contact
- Reporting issues
- Support channels
