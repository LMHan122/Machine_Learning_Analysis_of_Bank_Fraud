from sklearn.model_selection import train_test_split
import logging
import pandas as pd
import wandb
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)-20s %(message)s", filemode="a"
)
logger = logging.getLogger()


def split_data():
    """
    Split data into train and test sets by first splitting customer into fraud
    and no fraud groups, splitting the two groups into train and test sets, then
    combining the two groups back together.
    :return:None
    """

    try:
        # grabbing the dataset from WandB
        logger.info("Pulling final dataset from WandB")
        run = wandb.init(project="credit_card_fraud", save_code=True)
        artifact = run.use_artifact(
            "lhan122-student/credit_card_fraud/final_credit_card_data:v0",
            type="dataset",
        )
        artifact_dir = artifact.download()
        file_path = os.path.join(artifact_dir, "final_credit_card_fraud.parquet")
        df = pd.read_parquet(file_path)
        logger.info(f"Dataset loaded successfully with shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        run.finish()
        raise

    logger.info(
        "Grouping customer into separate lists of those with fraud transactions and those without."
    )
    # grouping customer's transactions together and finding out if they have any fraudulent transactions
    cust_list = df[["cc_num", "is_fraud"]].drop_duplicates()
    # if customer has a value count of 2, then they have both fraud and non fraud transactions
    cust_count = cust_list["cc_num"].value_counts()

    # splitting them into two groups
    cust_fraud = cust_count[cust_count == 2].index.tolist()
    cust_no_fraud = cust_count[cust_count == 1].index.tolist()

    logger.info("Splitting the two groups into train and test sets.")
    # now creating test and train groups for both fraud and no fraud
    train_fraud, test_fraud = train_test_split(
        cust_fraud, test_size=0.2, random_state=7
    )
    train_no_fraud, test_no_fraud = train_test_split(
        cust_no_fraud, test_size=0.2, random_state=15
    )

    logger.info("Combining the groups into a train and test set.")
    # combining the two lists together
    train_cust = train_fraud + train_no_fraud
    test_cust = test_fraud + test_no_fraud

    # applying it to the df to get all columns
    train_df = df[df["cc_num"].isin(train_cust)]
    test_df = df[df["cc_num"].isin(test_cust)]

    # saving the df
    logger.info("Saving train and test datasets.")
    train_file = "train_data.parquet"
    test_file = "test_data.parquet"
    train_df.to_parquet(train_file, index=False)
    test_df.to_parquet(test_file, index=False)

    # upload datasets to wandb
    logger.info("Uploading the two sets to Weights and Biases.")
    train_artifact = wandb.Artifact(name="train_data", type="dataset")
    train_artifact.add_file(train_file)
    run.log_artifact(train_artifact)

    test_artifact = wandb.Artifact(name="test_data", type="dataset")
    test_artifact.add_file(test_file)
    run.log_artifact(test_artifact)

    run.finish()
    logger.info("All done.")


if __name__ == "__main__":
    split_data()
