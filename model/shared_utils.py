from pandas.conftest import axis_1
import wandb
import pandas as pd
import logging
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from time import time


def fetch_train_df():
    '''
    Pulling train data from wandb and starting logger.
    :return: train dataframe and open wandb run
    '''
    logging.basicConfig(level=logging.INFO, format="%(asctime)-20s %(message)s", filemode="a")
    logger = logging.getLogger()

    try:
        # grabbing the dataset from WandB
        logger.info("Pulling train dataset from WandB")
        run = wandb.init(project="credit_card_fraud", job_type='training_model', save_code=True)
        artifact = run.use_artifact('lhan122-student/credit_card_fraud/train_data:latest', type='dataset')
        artifact_dir = artifact.download()
        file_path = os.path.join(artifact_dir, "train_data.parquet")
        df = pd.read_parquet(file_path)
        logger.info(f"Dataset loaded successfully with shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        run.finish()
        raise
    return df, run


def preprocess_data(df):
    '''
    Preprocessing data for modeling
    :param df: dataframe pulled from wandb
    :return: train/test split
    '''
    logging.basicConfig(level=logging.INFO, format="%(asctime)-20s %(message)s", filemode="a")
    logger = logging.getLogger()
    logger.info('Preprocessing data')

    #dropping columns that won't be used
    logger.info('Dropping columns')
    drop_col = ['cc_num', 'job', 'trans_num', 'date']
    df.drop(drop_col, axis=1, inplace=True)

    #dropping rows with null values
    logger.info('Dropping Nulls')
    df.dropna(subset=['cust_lat', 'cust_long', 'trans_distance_km'], inplace=True)

    #encoding category columns
    logger.info('Encoding categorical variables')
    cat_col = ['merchant', 'category']
    df = pd.get_dummies(df, columns=cat_col, drop_first=True)

    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

def train_evaluate_model(model, X_train, X_test, y_train, y_test, run):
    '''
    Train and evaluate model.
    :param model: Model to be trained
    :param X_train: from preprocessing data
    :param X_test: from preprocessing data
    :param y_train: from preprocessing data
    :param y_test: from preprocessing data
    :param run: wandb run
    :return: precision, recall, f1_score, roc_auc_score
    '''
    logging.basicConfig(level=logging.INFO, format="%(asctime)-20s %(message)s", filemode="a")
    logger = logging.getLogger()
    start = time()
    logger.info(f'Training {model}')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    logger.info('Evaluating model')

    precision = precision_score(y_test, y_pred)
    logger.info(f'Precision: {precision}')

    recall = recall_score(y_test, y_pred)
    logger.info(f'Recall: {recall}')

    f1 = f1_score(y_test, y_pred)
    logger.info(f'F1 score: {f1}')

    auc = roc_auc_score(y_test, y_pred)
    logger.info(f'AUC: {auc}')

    end = time()
    total = end - start
    logger.info(f'Total time to train and evaluate {model}: {total}')
    return precision, recall, f1, auc


    #save the model to a pickle file
    model_name = model.__class__.__name__
    logger.info(f'Saving model {model_name} to a pickle file.')
    pickle_file = f'{model_name}.pkl'
    with open(pickle_file, 'wb') as file:
        pickle.dump(model, file)

    # uploading pickle to wandb
    logger.info(f'Uploading {pickle_file} to WandB.')
    artifact = wandb.Artifact(name=f'{model_name}_artifact', type='model')
    artifact.add_file(pickle_file)
    run.log_artifact(artifact)

    #removing pickle file
    os.remove(pickle_file)


if __name__ == '__main__':
    pass




