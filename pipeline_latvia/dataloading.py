
from utils import *
import rasterio
import numpy as np
import pandas as pd
import os

def read_nfi_data(nfi_data_path):
    nfi_data = pd.read_csv(nfi_data_path)
    return nfi_data

def extend_nfi_data(nfi_data):
    nfi_data['field_month'] = nfi_data.apply(get_field_month, axis=1)
    nfi_data['scan_month'] = nfi_data.apply(get_field_month, axis=1)
    nfi_data['time_diff'] = nfi_data.apply(get_time_diff, axis=1)
    nfi_data['north_processed'] = nfi_data.apply(get_north, axis=1)
    nfi_data['east_processed'] = nfi_data.apply(get_east, axis=1)
    return nfi_data

def read_raster_data(raster_data_path, nfi_data, predictor_columns, target_columns):
    raster_images_paths = []
    raster_images = []
    for index, row in nfi_data.iterrows():

        raster_image = rasterio.open(os.path.join(raster_data_path, str(row['Description']) + '.tif')).read()
        raster_images_paths.append(os.path.join(raster_data_path, str(row['Description']) + '.tif'))
        raster_images.append(raster_image)
        targets.append(list(np.array(nfi_data[target_columns].iloc[index])))

    return np.array(raster_images),  nfi_data

