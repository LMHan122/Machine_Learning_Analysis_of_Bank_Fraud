import pandas as pd
import logging
import wandb


def clean_data():
    # starting logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)-20s %(message)s", filemode="w"
    )
    logger = logging.getLogger()

    # setting up wandb
    logger.info("Starting a WandB run.")
    run = wandb.init(project="credit_card_data", save_code=True)

    # grabbing the dataset from wandb
    logger.info("Pulling original dataset from WandB")
    artifact = run.use_artifact(
        "lhan122-student/credit_card_fraud/credit_card_data:v0", type="dataset"
    )
    df = pd.read_parquet(artifact.file())

    # dropping columns
    # dropping unnamed column
    logger.info("Starting Preprocessing")
    logger.info("Dropping columns")
    df.drop(
        df.columns[df.columns.str.contains("unnamed", case=False)], axis=1, inplace=True
    )

    # dropping named columns
    drop_columns = [
        "first",
        "last",
        "gender",
        "street",
        "zip",
        "merch_zipcode",
        "unix_time",
        "city_pop",
    ]
    df.drop(drop_columns, axis=1, inplace=True)

    # changing column name
    logger.info("Changing column names")
    df.rename(columns={"trans_date_trans_time": "trans_dt"}, inplace=True)

    # dropping any duplicates
    logger.info("Dropping duplicate records (if any)")
    df.drop_duplicates(inplace=True)

    # changing data types
    logger.info("Changing column data types")
    df["trans_dt"] = pd.to_datetime(df["trans_dt"])
    df["dob"] = pd.to_datetime(df["dob"])

    cat_columns = ["category", "merchant", "job", "is_fraud", "state", "city"]
    df[cat_columns] = df[cat_columns].astype("category")

    # converting dataset back to csv file and uploading it to WandB
    logger.info("Uploading cleaned data to Weights & Biases")
    df.to_csv("cleaned_credit_card_fraud.csv", index=False)
    artifact = wandb.Artifact(name="cleaned_credit_card_fraud.csv", type="dataset")
    artifact.add_file("cleaned_credit_card_fraud.csv")
    run.log_artifact(artifact)
    logger.info("Cleand data upload to Weights & Biases is complete.")

    # finishing WandB run
    run.finish()
    logger.info("Data cleaning complete.")