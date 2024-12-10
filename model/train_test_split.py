from sklearn.model_selection import train_test_split
import logging
import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-20s %(message)s",
                    filemode="a")
logger = logging.getLogger()

def train_test_split():
    run = wandb.init(project="credit_card_fraud", save_code=True)


#FIXME: This need to be completely corrected. I got side-tracked trying to fix issues on WandB
    try:
        # grabbing the dataset from WandB
        logger.info("Pulling cleaned dataset from WandB")
        artifact = run.use_artifact(
            "lhan122-student/credit_card_fraud/cleaned_credit_card_data:latest",
            type="dataset",
        )
        artifact_dir = artifact.download()
        file_path = os.path.join(artifact_dir, "cleaned_credit_card_fraud.parquet")
        df = pd.read_parquet(file_path)
        logger.info(f"Dataset loaded successfully with shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        run.finish()
        raise







# Download input artifact. This will also note that this script is using this
# particular version of the artifact
logger.info(f"Fetching artifact {args.input}")
artifact_local_path = run.use_artifact(args.input).file()

df = pd.read_csv(artifact_local_path)

logger.info("Splitting trainval and test")
trainval, test = train_test_split(
    df,
    test_size=args.test_size,
    random_state=args.random_seed,
    stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
)