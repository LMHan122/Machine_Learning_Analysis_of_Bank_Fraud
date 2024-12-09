import pandas as pd
import logging
import wandb
import os
import wandb
from time import time, sleep
from haversine import haversine, Unit


def feature_creation():
    # starting logging
    start_time = time()
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
    logger.info("Creating new column 'trans_type'.")
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

    # mapping values to new column
    df["trans_type"] = df["category"].map(transaction_map).astype("category")
    logger.info("New column 'trans_type' added.")

    # accessing pre-mapped customer lat and long values
    # this file was created in Feature_Engineering_Test.ipynb
    logger.info(
        "Loading pre-mapped lat and long coordinates for customer's city and state, and using it to create 'cust_lat' and 'cust_long' columns."
    )
    cust_loc = pd.read_csv(r"data/customer_locations.csv")

    df = df.merge(
        cust_loc[["city", "state", "cust_lat", "cust_long"]],
        on=["city", "state"],
        how="left",
    )

    # dropping city column
    logger.info("Dropping 'city' column.")
    df.drop(["city"], axis=1, inplace=True)

    logger.info("Creating new columns 'trans_distance_km' and 'merch_distance_km'.")
    # creating column for distance between customer's city and the transaction location
    df["trans_distance_km"] = df.swifter.apply(
        lambda row: haversine(
            (row["cust_lat"], row["cust_long"]),
            (row["lat"], row["long"]),
            unit=Unit.KILOMETERS,
        ),
        axis=1,
    )
    # creating column for distance between customer's city and merchant's location
    df["merch_cust_km"] = df.swifter.apply(
        lambda row: haversine(
            (row["merch_lat"], row["merch_long"]),
            (row["lat"], row["long"]),
            unit=Unit.KILOMETERS,
        ),
        axis=1,
    )

    #creating new timedate columns
    #date related columns
    logger.info("Creating new timedate columns.")
    df['date'] = df['trans_date'].dt.date
    df['year'] = df['trans_dt'].dt.year
    df['month'] = df['trans_dt'].dt.month
    df['day'] = df['trans_dt'].dt.day
    df['quarter'] = df['trans_dt'].dt.quarter
    df['week'] = df['trans_dt'].dt.week
    df['dayofweek'] = df['trans_dt'].dt.dayofweek

    #time related columns
    df['time'] = df['trans_dt'].dt.time
    df['hour'] = df['trans_dt'].dt.hour
    logger.info("New datetime columns added.")











    end_time = time()
    total_time = end_time - start_time
    logger.info(f"Total time {total_time}s")


if __name__ == "__main__":
