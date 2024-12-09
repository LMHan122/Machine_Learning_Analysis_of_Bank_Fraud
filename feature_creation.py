import pandas as pd
import logging
import wandb
import os
import wandb


def feature_creation():
    # starting logging
    # starting logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)-20s %(message)s")
    logger = logging.getLogger()

    # setting up WandB
    logger.info("Starting a WandB run.")
    run = wandb.init(project="credit_card_fraud", save_code=True)

    try:
        # grabbing the dataset from WandB
        logger.info("Pulling cleaned dataset from WandB")
        artifact = run.use_artifact(
            "lhan122-student/credit_card_fraud/cleaned_credit_card_data:latest",
            type="dataset",
        )
        artifact_dir = artifact.download()
        file_path = os.path.join(artifact_dir, "cleaned_credit_card_fraud.parquet")
        df = pd.read_parquet(file_path)
        logger.info(f"Dataset loaded successfully with shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        run.finish()
        raise

    # starting feature engineering

    # creating new transaction type column
    logger.info("Creating new column \'trans_type\'.")
    transaction_map = {
        "shopping_net": 1,
        "misc_net": 1,
        "grocery_net": 1,
        "grocery_pos": 2,
        "shopping_pos": 2,
        "misc_pos": 2,
        "gas_transport": 2,
        "home": 0,
        "kids_pets": 0,
        "entertainment": 0,
        "food_dining": 0,
        "personal_care": 0,
        "health_fitness": 0,
        "travel": 0,
    }

    #mapping values to new column
    df['trans_type'] = df['category'].map(transaction_map).astype('category')
    logger.info("New column \'trans_type\' added.")

    #accessing premapped customer lat and long values
    #this file was created in Feature_Engineering_Test.ipynb
    cust_loc = pd.read_csv(r'data/customer_locations.csv')








