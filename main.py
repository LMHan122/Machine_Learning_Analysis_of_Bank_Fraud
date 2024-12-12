import os
import logging
import wandb
from data.get_data import *
from data.data_preparation.clean_data import *
from data.data_preparation.cust_loc_map import *
from data.data_preparation.feature_creation import *
from model.shared_utils import *
from model.train_test_split import *
from model.test_xgboost import *

# setting up logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)-20s %(message)s", filemode="a"
)
logger = logging.getLogger()


def xgboost_pipeline():
    """
    This function downloads data, prepares it, and then tests
    a pre-trained XGBoost model.
    :return: Tested model is uploaded to wandb.
    """
    # getting data from Kaggle
    data_download()

    # data exploration occurs in data/data_understanding/Data_Understanding.ipynb

    # cleaning data
    clean_data()

    # creating customer location mapping if not already done
    # cust_loc_map()     since this step has already been completed

    # creating new features
    feature_creation()

    # creating the train and test datasets
    split_data()

    # use model/train_all_models.py to test 3 models
    # not included in pipeline since it requires manual review

    # use model/train_parameter_tuning.py to run a sweep
    # for hyperparameter tuning on the chosen model
    # not include in pipeline since it requires manual review

    # tests xgboost model
    test_xgboost()


if __name__ == "__main__":
    logger.info("Testing XGBoost from pipeline")
    xgboost_pipeline()
