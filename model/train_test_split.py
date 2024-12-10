from sklearn.model_selection import train_test_split
import logging
import pandas as pd
import wandb

logging.basicConfig(
    level=logging.INFO, format="%(asctime)-20s %(message)s", filemode="a"
)
logger = logging.getLogger()

run = wandb.init(job_type="train_val_test_split")


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