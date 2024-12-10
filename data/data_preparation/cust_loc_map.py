import pandas as pd
import logging
import os
import wandb
from geopy.geocoders import Nominatim
from time import time,sleep

logging.basicConfig(
    level=logging.INFO, format="%(asctime)-20s %(message)s", filemode="a"
)
logger = logging.getLogger()

# grabbing cleaned dataset from WandB
logging.info('Getting the dataset from WandB')
run = wandb.init(project="credit_card_fraud", save_code=True)
artifact = run.use_artifact(
    "lhan122-student/credit_card_fraud/cleaned_credit_card_data:latest", type="dataset"
)
artifact_dir = artifact.download()
file_path = os.path.join(artifact_dir, "cleaned_credit_card_fraud.parquet")
df = pd.read_parquet(file_path)


# creating map for customer location
cust_loc = df[["city", "state"]].drop_duplicates()

# getting lat and long values for city and state
geolocator = Nominatim(user_agent="geoapi")


# creating function to be applied
def get_lat_long(city, state):
    try:
        location = geolocator.geocode(f"{city}, {state}")
        if location:
            return location.latitude, location.longitude

        else:
            return None, None
    except Exception as e:
        logger.error(f"Error fetching location for {city}, {state}: {e}")
        return None, None
    finally:
        sleep(1)

logger.info('Getting the geocoding file')
# apply geocoding
cust_loc["lat_long"] = cust_loc.apply(
    lambda x: get_lat_long(x["city"], x["state"]), axis=1
)
cust_loc[["cust_lat", "cust_long"]] = pd.DataFrame(
    cust_loc["lat_long"].tolist(), index=cust_loc.index
)

# uploading it to WandB as well for record keeping
cust_loc_file = "cust_loc.parquet"
cust_loc.to_parquet(cust_loc_file, index=False)

artifact = wandb.Artifact(
    name="cust_loc_data",
    type="dataset",
    description="Lat and Long values for customer's city and state.",
)
artifact.add_file(cust_loc_file)
run.log_artifact(artifact)
