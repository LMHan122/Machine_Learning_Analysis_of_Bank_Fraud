from data.data_preparation.cust_loc_map import get_lat_long, cust_loc_map
import os
import logging

#setting up logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)-20s %(message)s", filemode="a"
)
logger = logging.getLogger()


if __name__ == '__main__':




    #if city and state location values don't already exist
    #new file will be created getting these values
    logger.info('Checking if latitude and longitude map exists')
    cust_map_file_path = 'data/data_preparation/cust_loc.parquet'
    if not os.path.exists(cust_map_file_path):
        logger.info('Getting City and State latitude and longitude values.')
        cust_loc_map()
