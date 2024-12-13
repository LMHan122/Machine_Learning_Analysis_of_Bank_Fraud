import pandas as pd
import logging
import wandb
import os
import wandb
from time import time, sleep
from haversine import haversine, Unit
import swifter
from rapidfuzz import fuzz, process


logging.basicConfig(
    level=logging.INFO, format="%(asctime)-20s %(message)s", filemode="a"
)
logger = logging.getLogger()


def feature_creation():
    """
    Creates new features for the dataset.
    :return: None
    """
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

    cust_artifact = run.use_artifact(
        "lhan122-student/credit_card_fraud/cust_loc_data:latest", type="dataset"
    )
    cust_artifact_dir = cust_artifact.download()
    file_path = os.path.join(cust_artifact_dir, "cust_loc.parquet")
    cust_loc = pd.read_parquet(file_path)

    df = df.merge(
        cust_loc[["city", "state", "cust_lat", "cust_long"]],
        on=["city", "state"],
        how="left",
    )

    # dropping city column
    logger.info("Dropping 'city' and 'state' column.")
    df.drop(["city", "state"], axis=1, inplace=True)

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

    # creating new timedate columns
    # date related columns
    logger.info("Creating new time and date columns.")
    df["date"] = df["trans_dt"].dt.date
    df["year"] = df["trans_dt"].dt.year
    df["month"] = df["trans_dt"].dt.month
    df["day"] = df["trans_dt"].dt.day
    df["quarter"] = df["trans_dt"].dt.quarter
    df["week"] = df["trans_dt"].dt.isocalendar().week
    df["day_of_week"] = df["trans_dt"].dt.dayofweek

    # time related columns
    df["hour"] = df["trans_dt"].dt.hour
    logger.info("New datetime columns added.")

    # creating the rolling columns
    logger.info("Creating new rolling columns.")
    # double-checking column is in the right type and sorted
    df["trans_dt"] = pd.to_datetime(df["trans_dt"])
    df = df.sort_values(["cc_num", "trans_dt"])

    logger.info("Creating rolling trans by last hour column.")
    df["trans_by_last_hr"] = (
        df.groupby("cc_num")
        .rolling("1H", on="trans_dt")["trans_dt"]
        .count()
        .reset_index(drop=True)
    )

    logger.info("Creating rolling trans by last day column.")
    df["trans_by_last_day"] = (
        df.groupby("cc_num")
        .rolling("1D", on="trans_dt")["trans_dt"]
        .count()
        .reset_index(drop=True)
    )

    logger.info("Creating rolling trans amount by last hour column.")
    df["amt_by_last_hr"] = (
        df.groupby("cc_num")
        .rolling("1H", on="trans_dt")["amt"]
        .sum()
        .reset_index(drop=True)
    )

    logger.info("Creating rolling trans amount by last day column.")
    df["amt_by_last_day"] = (
        df.groupby("cc_num")
        .rolling("1D", on="trans_dt")["amt"]
        .sum()
        .reset_index(drop=True)
    )
    """
    This column was not used for modeling, leading to this section of code being
    commented out. Might be used in future versions.
    """
    # creating list of jobs and count of each job
    #logger.info("Creating map for job titles.")
    #df["job"] = df["job"].str.lower().str.strip()

    # creating list fo all unique job titles
    #jobs = df["job"].drop_duplicates()

    # creating an empty set for rapidfuzz
    #similar_jobs = set()

    # using rapidfuzz, compairing job titles together and creating a list
    # for values that have a similarity score higher then 95

    # logger.info("Running RapidFuzz.")
    # for i, job in enumerate(jobs):
    #     matches = process.extract(
    #         job, jobs[i + 1 :], scorer=fuzz.token_sort_ratio, processor=None, limit=5
    #     )
    #
    #     # if the similarity score is 95 or higher, job titles are added to this list
    #     for match in matches:
    #         if match[1] >= 95:
    #             similar_jobs.add((job, match[0]))
    #
    # # creating empty dict for map
    # job_map = {}
    #
    # # picking the shortest job title as the job title to use
    # for job1, job2 in similar_jobs:
    #     j1 = job_map.get(job1, job1)
    #     j2 = job_map.get(job2, job2)
    #
    #     # picking the shortest job title
    #     short_job = min(j1, j2)
    #     job_map[job1] = short_job
    #     job_map[job2] = short_job
    #
    # logger.info("Applying Job Titles to job column.")
    # # applying the job titles map
    # df["job"] = df["job"].map(job_map)

    ###creating 'age_at_trans' column
    logger.info("Creating age_at_trans column.")
    df["age_at_trans"] = (df["trans_dt"] - df["dob"]).dt.days / 365.25

    logger.info("Dropping 'trans_dt' and 'dob' column.")
    df.drop(["trans_dt", "dob"], axis=1, inplace=True)

    logger.info("Feature Creation Done")

    logger.info("Uploading final dataset to WandB.")
    final_file = "final_credit_card_fraud.parquet"
    df.to_parquet(final_file, index=False)

    artifact = wandb.Artifact(
        name="final_credit_card_data",
        type="dataset",
        description="Final dataset with all features created.",
    )
    artifact.add_file(final_file)
    run.log_artifact(artifact)

    # finishing wandb run
    run.finish()

    end_time = time()
    total_time = end_time - start_time
    logger.info(f"Total time {total_time}s")

    logger.info("Feature Creation Done")


if __name__ == "__main__":
    feature_creation()
