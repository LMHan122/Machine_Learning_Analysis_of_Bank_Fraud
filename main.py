from data.data_preparation.cust_loc_map import get_lat_long
import os




if __name__ == '__main__':


    if not os.path.exists('data/data_preparation/cust_loc.parquet'):
        get_lat_long()
