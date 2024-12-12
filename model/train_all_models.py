from shared_utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)-20s %(message)s", filemode="a"
)
logger = logging.getLogger()

if __name__ == "__main__":
    # getting training data
    df, run = fetch_train_df()

    # preprocessing data from training dataset
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # logistic regression model
    logger.info("Logistic Regression model with class_weight='balanced'")
    lr_model = LogisticRegression(class_weight="balanced", random_state=10)
    train_evaluate_model(lr_model, X_train, X_test, y_train, y_test, run)

    # random forest model
    logger.info("Random Forest Classifier with 250 trees - rest default parameters.")
    rf_model = RandomForestClassifier(n_estimators=250, random_state=10)
    train_evaluate_model(rf_model, X_train, X_test, y_train, y_test, run)

    # xgboost model
    logger.info("XGB model with gbtree, learning_rate=0.3, max_depth=5, and GPU.")
    xgb_model = XGBClassifier(
        booster="gbtree",
        learning_rate=0.3,
        max_depth=5,
        tree_method="hist",
        device="cuda",
        random_state=10,
        eval_metric="auc",
    )
    train_evaluate_model(xgb_model, X_train, X_test, y_train, y_test, run)

    # closing wandb run
    run.finish()

    logger.info("The models have been trained successfully.")
