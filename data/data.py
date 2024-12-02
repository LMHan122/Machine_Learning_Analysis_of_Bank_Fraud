import kagglehub
import shutil

# Download latest version
path = kagglehub.dataset_download("priyamchoksi/credit-card-transactions-dataset")

# Define the destination directory
destination_dir = r"C:\Users\eelil\OneDrive\Desktop\Capstone\Machine_Learning_Analysis_of_Bank_Fraud\data"

# Move the files
shutil.move(path, destination_dir)