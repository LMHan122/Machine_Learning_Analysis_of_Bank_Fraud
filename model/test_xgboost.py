import pandas as pd
import os
import logging
import pickle
import wandb
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)-20s %(message)s", filemode="a")
logger = logging.getLogger()

def preprocess_test_df(run):
    '''
    Retrieving the test dataframe from WandB and preprocessing the test dataframe.
    :return:X_test, y_test
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

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    return X, y


if __name__ == '__main__':
    #loading pickle file
    logger.info('Pulling model pickle file')
    run = wandb.init(project='credit_card_fraud')
    artifact = run.use_artifact('lhan122-student/credit_card_fraud/XGBClassifier_artifact:v7', type='model')
    artifact_dir = artifact.download()

    #training and evaluating model
    xgb_model = pd.read_pickle(os.path.join(artifact_dir, 'XGBClassifier.pkl'))

    #preprocessing the test data
    X, y = preprocess_test_df(run)

    #making predictions and evaluating
    logger.info('Making predictions')
    y_pred = xgb_model.predict(X)

    #uploading model to wandb
    logger.info('Uploading model to WandB')
    model_artifact = wandb.Artifact(
        name='XGBClassifier_tested',
        type='model',
        description='Tested XGBoost model with updated preprocessing and feature engineering',
        metadata={'framework': 'xgboost', 'test_dataset': 'credit_card_fraud/test_data:v1'}
    )
    model_artifact.add_file(os.path.join(artifact_dir, 'XGBClassifier.pkl'))
    run.log_artifact(model_artifact)

    logger.info('Logging results')
    report = classification_report(y, y_pred, output_dict=True)
    wandb.log({"classification_report": report})

    logger.info('Logging metrics')
    wandb.log({
        "test_accuracy": report["accuracy"],
        "precision_class_1": report["1"]["precision"],
        "recall_class_1": report["1"]["recall"]
    })

    #finding important features
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    important_features_path = os.path.join(os.getcwd(), 'important_features.csv')
    feature_importance.to_csv(important_features_path, index=False)

    logger.info('Uploading df to WandB')
    feature_artifact = wandb.Artifact(
        name='important_features',
        type='dataset',
        description='Feature importance df',
    )
    feature_artifact.add_file(important_features_path)
    run.log_artifact(feature_artifact)
    run.finish()

    logger.info('Done')


















