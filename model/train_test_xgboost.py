from xgboost import XGBClassifier
import os
from shared_utils import *
import logging
import pickle

#creating model with parameters from tuning
xgb_model = XGBClassifier(
    booster='gbtree',
    learning_rate=0.3,
    max_depth=7,
    tree_method='hist',
    device='cuda'
    random_state=10
    eval_metric='auc',
)

def preprocess_test_df(run):
    '''
    Retrieving the test dataframe from WandB and preprocessing the test dataframe.
    :return:
    '''
    test_df = run.use_artifact('lhan122-student/credit_card_fraud/test_data:v1', type='dataset')
    test_dir = test_df.download()
    test_path = os.path.join(test_dir, 'test_data.parquet')
    df = pd.read_parquet(test_path)
    logger.info(f"Dataset loaded successfully with shape: {df.shape}")

    logger.info("Preprocessing Test data")
    # dropping columns that won't be used
    logger.info("Dropping columns")
    drop_col = ["cc_num", "job", "trans_num", "date"]
    df.drop(drop_col, axis=1, inplace=True)

    # dropping rows with null values
    logger.info("Dropping Nulls")
    df.dropna(subset=["cust_lat", "cust_long", "trans_distance_km"], inplace=True)

    # encoding category columns
    logger.info("Encoding categorical variables")
    cat_col = ["merchant", "category"]
    df = pd.get_dummies(df, columns=cat_col, drop_first=True)

    X_test = df.drop('is_fraud')
    y_test = df['is_fraud']

    return X_test, y_test


if __name__ == '__main__':
    #setting up logger
    logging.basicConfig(format="%(asctime)-20s %(message)s", level=logging.INFO, filemode='a')
    logger = logging.getLogger()

    #pulling train dataset and starting a wandb run
    df, run = fetch_train_df()

    #preprocessing data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    #training and evaluating model
    xgb_model = xgb_model
    train_evaluate_model(xgb_model, X_train, X_test, y_train, y_test, run)



    #preprocessing the test data

    X_test, y_test = preprocess_test_df(run)













