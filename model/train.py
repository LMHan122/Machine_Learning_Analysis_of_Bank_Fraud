from shared_utils import *
from xgboost import XGBClassifier
import wandb
import os
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-20s %(message)s", filemode="a")
logger = logging.getLogger()

run = wandb.init(project='credit_card_fraud')
config = wandb.config
artifact = wandb.use_artifact("lhan122-student/credit_card_fraud/train_data:latest", type="dataset")
artifact_dir = artifact.download()
file_path = os.path.join(artifact_dir, "train_data.parquet")
df = pd.read_parquet(file_path)

#preprocessing data from training dataset
X_train, X_test, y_train, y_test = preprocess_data(df)
xgb_model = XGBClassifier(
    booster=config.booster,
    learning_rate=config.learning_rate,
    max_depth=config.max_depth,
    tree_method=config.tree_method,
    random_state=config.random_state,
    eval_metric=config.eval_metric,
)
precision, recall, f1, auc = train_evaluate_model(xgb_model, X_train, X_test, y_train, y_test, run)

#logging metrics using wandb.log and not just logger
wandb.log({
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "auc": auc
})

logger.info("Sweep run complete")
run.finish()