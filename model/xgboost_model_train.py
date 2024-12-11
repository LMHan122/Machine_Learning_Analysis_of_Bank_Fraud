from shared_utils import *
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)-20s %(message)s", filemode="a")
logger = logging.getLogger()


#The code below failed, will attempt to modify the parameters and run the model on it's own
#logger.info("XGB model with gbtree. eta=0.3, max_depth=5, and GPU.")
#xgb_model = XGBClassifier(booster="gbtree", eta=0.3, max_depth=5, device='cuda', random_state=10)
#train_evaluate_model(xgb_model, X_train, y_train, X_test, y_test, run)


if __name__ == "__main__":
    #getting training data
    df, run = fetch_train_df()

    #preprocessing data from training dataset
    X_train, X_test, y_train, y_test = preprocess_data(df)

    logger.info("XGB model with gbtree, learning_rate=0.3, max_depth=5, and GPU.")
    xgb_model = XGBClassifier(
        booster="gbtree",
        learning_rate=0.3,
        max_depth=5,
        tree_method="gpu_hist",
        random_state=10,
        eval_metric="auc",
    )
    train_evaluate_model(xgb_model, X_train, X_test, y_train, y_test, run)