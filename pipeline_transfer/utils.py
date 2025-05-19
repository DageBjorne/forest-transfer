
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from config import *
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import rasterio

def calculate_percentage_rmse(preds, ground_truth):
    mse = mean_squared_error(ground_truth, preds)
    rmse = sqrt(mse)
    mean_target = np.mean(ground_truth)
    percentage_rmse = (rmse / mean_target) * 100
    return percentage_rmse

def compute_rmse(preds, ground_truth):
    rmse = np.sqrt(np.mean(np.square(preds-ground_truth)))
    return rmse
    filtered_data = data[data['area_code'] == area_code]
    return filtered_data

def compute_bias(preds, ground_truth):
    return np.mean(preds) - np.mean(ground_truth)


def create_train_test_split_for_regression(data, seed, test_size):
    data_train, data_test = train_test_split(data, test_size = test_size, random_state=seed)
    return data_train, data_test

def create_train_val_test_split(data, seed, test_size):
    data_temp, data_test = train_test_split(data, test_size=test_size, random_state=seed)
    data_train, data_val = train_test_split(data_temp, test_size=val_size, random_state=seed)
    return data_train, data_val, data_test

def prepare_datasets_for_cnn(data_train, data_val, data_test):
    train_images, train_extra_predictors, train_targets = [], [], []
    for index, row in data_train.iterrows():
        train_images.append(rasterio.open(row['raster_image_path']).read())
        train_extra_predictors.append(row[predictor_columns])
        train_targets.append(row[target_columns])
    val_images, val_extra_predictors, val_targets = [], [], []
    for index, row in data_val.iterrows():
        val_images.append(rasterio.open(row['raster_image_path']).read())
        val_extra_predictors.append(row[predictor_columns])
        val_targets.append(row[target_columns])
    test_images, test_extra_predictors, test_targets = [], [], []
    for index, row in data_test.iterrows():
        test_images.append(rasterio.open(row['raster_image_path']).read())
        test_extra_predictors.append(row[predictor_columns])
        test_targets.append(row[target_columns])

    #convert to numpys
    train_images = np.array(train_images).astype(np.float32)
    train_extra_predictors = np.array(train_extra_predictors).astype(np.float32)
    train_targets = np.array(train_targets).astype(np.float32)

    val_images = np.array(val_images).astype(np.float32)
    val_extra_predictors = np.array(val_extra_predictors).astype(np.float32)
    val_targets = np.array(val_targets).astype(np.float32)

    test_images = np.array(test_images).astype(np.float32)
    test_extra_predictors = np.array(test_extra_predictors).astype(np.float32)
    test_targets = np.array(test_targets).astype(np.float32)

    return train_images, train_extra_predictors, train_targets, val_images, val_extra_predictors, val_targets, test_images, test_extra_predictors, test_targets

#Ugly solution for transfer learning purposes
def prepare_datasets_for_cnn_target(data_train, data_val, data_test):
    train_images, train_extra_predictors, train_targets = [], [], []
    for index, row in data_train.iterrows():
        train_images.append(rasterio.open(row['raster_image_path']).read())
        train_extra_predictors.append(row[predictor_columns])
        train_targets.append(row[target_columns_target])
    val_images, val_extra_predictors, val_targets = [], [], []
    for index, row in data_val.iterrows():
        val_images.append(rasterio.open(row['raster_image_path']).read())
        val_extra_predictors.append(row[predictor_columns])
        val_targets.append(row[target_columns_target])
    test_images, test_extra_predictors, test_targets = [], [], []
    for index, row in data_test.iterrows():
        test_images.append(rasterio.open(row['raster_image_path']).read())
        test_extra_predictors.append(row[predictor_columns])
        test_targets.append(row[target_columns_target])

    #convert to numpys
    train_images = np.array(train_images).astype(np.float32)
    train_extra_predictors = np.array(train_extra_predictors).astype(np.float32)
    train_targets = np.array(train_targets).astype(np.float32)

    val_images = np.array(val_images).astype(np.float32)
    val_extra_predictors = np.array(val_extra_predictors).astype(np.float32)
    val_targets = np.array(val_targets).astype(np.float32)

    test_images = np.array(test_images).astype(np.float32)
    test_extra_predictors = np.array(test_extra_predictors).astype(np.float32)
    test_targets = np.array(test_targets).astype(np.float32)

    return train_images, train_extra_predictors, train_targets, val_images, val_extra_predictors, val_targets, test_images, test_extra_predictors, test_targets

def fit_imputer_and_scaler(train_images, train_extra_predictors):
    # Initializing imputers and scalers
    image_imputer = SimpleImputer(strategy='constant', fill_value=0)
    #image_imputer = KNNImputer(n_neighbors=50)
    extra_predictor_imputer = SimpleImputer(strategy='constant', fill_value=0)
    #extra_predictor_imputer = KNNImputer(n_neighbors=50)

    
    image_scaler = StandardScaler()
    extra_predictor_scaler = StandardScaler()

    # Reshaping and fitting on training data (images)
    train_images_reshaped = train_images.reshape((train_images.shape[0], -1))
    image_scaler.fit(train_images_reshaped)  # Fit the scaler on train images
    image_imputer.fit(train_images_reshaped)  # Fit the imputer on train images

    # Reshaping and fitting on training data (extra predictors)
    train_extra_predictors_reshaped = train_extra_predictors.reshape((train_extra_predictors.shape[0], -1))
    extra_predictor_scaler.fit(train_extra_predictors_reshaped)  # Fit the scaler on extra predictors
    extra_predictor_imputer.fit(train_extra_predictors_reshaped)  # Fit the imputer on extra predictors

    # Return the fitted scalers and imputers for later use
    return image_scaler, image_imputer, extra_predictor_scaler, extra_predictor_imputer

def transform_data(images, extra_predictors, image_scaler, image_imputer, extra_predictor_scaler, extra_predictor_imputer):
    # Reshape and transform the test images
    images_reshaped = images.reshape((images.shape[0], -1))
    images_scaled = image_scaler.transform(images_reshaped)
    images_imputed = image_imputer.transform(images_scaled)
    images_final = images_imputed.reshape(images.shape)

    # Reshape and transform the extra predictors
    extra_predictors_reshaped = extra_predictors.reshape((extra_predictors.shape[0], -1))
    extra_predictors_scaled = extra_predictor_scaler.transform(extra_predictors_reshaped)
    extra_predictors_imputed = extra_predictor_imputer.transform(extra_predictors_scaled)
    extra_predictors_final = extra_predictors_imputed.reshape(extra_predictors.shape)

    return images_final, extra_predictors_final


           


    
