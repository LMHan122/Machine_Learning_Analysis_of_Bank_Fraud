import kagglehub
import shutil
import logging
import os
import wandb

# setting up logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)-20s %(message)s", filemode="w"
)
logger = logging.getLogger()


def data_download():
    # connecting to wandb
    run = wandb.init(project="credit_card_fraud", job_type="data_download")

    # creating log message for start of download
    logger.info("Downloading data from Kaggle")

    # downloading data
    path = kagglehub.dataset_download("priyamchoksi/credit-card-transactions-dataset")

    # creating destination
    destination_dir = r"C:\Users\eelil\OneDrive\Desktop\Capstone\Machine_Learning_Analysis_of_Bank_Fraud\data"
    os.makedirs(destination_dir, exist_ok=True)

    # Move the files
    logger.info("Moving data to %s locally" % destination_dir)
    shutil.move(path, destination_dir)

    # Locating the CSV file within the downloaded folder
    downloaded_folder = os.path.join(
        destination_dir, "1"
    )  # KaggleHub puts files in a folder labeled '1'
    csv_files = [f for f in os.listdir(downloaded_folder) if f.endswith(".csv")]

    # log an error if data is not there
    if not csv_files:
        logger.error("No CSV files found in the downloaded data.")
        run.finish()
        return
    else:
        csv_file_path = os.path.join(downloaded_folder, csv_files[0])

        # Uploading the CSV file to Weights & Biases
        logger.info("Uploading data to Weights & Biases")
        artifact = wandb.Artifact(name="credit_card_data", type="dataset")
        artifact.add_file(csv_file_path)
        run.log_artifact(artifact)
        logger.info("Data upload to Weights & Biases complete.")

    # Finishing the run
    run.finish()


if __name__ == "__main__":
    data_download()
