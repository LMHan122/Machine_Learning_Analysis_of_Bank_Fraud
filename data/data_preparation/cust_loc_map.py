import pandas as pd
import logging
import os
import wandb
from geopy.geocoders import Nominatim
from time import time, sleep

logging.basicConfig(
    level=logging.INFO, format="%(asctime)-20s %(message)s", filemode="a"
)
logger = logging.getLogger()


def get_lat_long(city, state):
    """
    This function is used to get the latitude and longitude of a city
    and state combination using the geopy API.
    :param city: The city column in the df
    :param state: The state column in the df
    :return: Nothing - the vaules are appended onto the df
    """
    geolocator = Nominatim(user_agent="geoapi")

    try:
        location = geolocator.geocode(f"{city}, {state}")
        if location:
            sleep(1)
            return location.latitude, location.longitude

        else:
            return None, None
    except Exception as e:
        logger.error(f"Error fetching location for {city}, {state}: {e}")
        return None, None


def cust_loc_map():
    """
    This function is used to get the location of each city and state
    using the get_lat_long function and use it to create two new columns
    and populating the values.
    :return: None
    """
    # grabbing cleaned dataset from WandB
    logger.info("Getting the dataset from WandB")
    run = wandb.init(project="credit_card_fraud")
    artifact = run.use_artifact(
        "lhan122-student/credit_card_fraud/cleaned_credit_card_data:latest",
        type="dataset",
    )
    artifact_dir = artifact.download()
    file_path = os.path.join(artifact_dir, "cleaned_credit_card_fraud.parquet")
    df = pd.read_parquet(file_path)

    # creating map for customer location
    logger.info("Creating side df of unique city and state combinations")
    cust_loc = df[["city", "state"]].drop_duplicates()

    # getting lat and long values for city and state
    logger.info("Getting the geocoding file")
    cust_loc["lat_long"] = cust_loc.apply(
        lambda x: get_lat_long(x["city"], x["state"]), axis=1
    )
    cust_loc[["cust_lat", "cust_long"]] = pd.DataFrame(
        cust_loc["lat_long"].tolist(), index=cust_loc.index
    )

    # uploading it to WandB as well for record keeping
    logger.info("Uploading file to WandB.")
    cust_loc_file = "cust_loc.parquet"
    cust_loc.to_parquet(cust_loc_file, index=False)

    artifact = wandb.Artifact(
        name="cust_loc_data",
        type="dataset",
        description="Lat and Long values for customer's city and state.",
    )
    artifact.add_file(cust_loc_file)
    run.log_artifact(artifact)
    run.finish()
    #removing local copy
    os.remove(cust_loc_file)

if __name__ == "__main__":
    cust_loc_map()
