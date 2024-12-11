from data.data_preparation.cust_loc_map import get_lat_long, cust_loc_map
from data.get_data import data_download
import os
import logging

# setting up logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)-20s %(message)s", filemode="a"
)
logger = logging.getLogger()


if __name__ == "__main__":
    # getting data from Kaggle if file hasn't been downloaded already
    logger.info("Checking if dataset exists")
    orig_data_path = "data/1/credit_card_transactions.csv"
    if not os.path.exists(orig_data_path):
        logger.info("Downloading original dataset")
        data_download()
    else:
        logger.info("Original dataset exists")

    # if city and state location map values don't already exist
    # new file will be created getting these values
    logger.info("Checking if latitude and longitude map exists")
    cust_map_file_path = "data/data_preparation/cust_loc.parquet"

    if not os.path.exists(cust_map_file_path):
        logger.info("Getting City and State latitude and longitude values.")
        cust_loc_map()
    else:
        logger.info("City and State latitude and longitude map exists")
