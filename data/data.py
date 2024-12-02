import kagglehub
import shutil

# Download latest version
path = kagglehub.dataset_download("priyamchoksi/credit-card-transactions-dataset")

# Define the destination directory
destination_dir = r"/data"

# Move the files
shutil.move(path, destination_dir)