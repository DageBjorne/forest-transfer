from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cnn_model import *

import pandas as pd
import numpy as np
import rasterio
import os

import torch
import torchvision
from config import *
from augment import *
from cnn_model import *
from utils import *
from augment import *

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, raster_images, extra_predictors, targets, augment=False):
        """
        Args:
            image_paths (list): List of paths to the image files.
            labels (list): List of 5-dimensional labels.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.raster_images = raster_images
        self.extra_predictors = extra_predictors
        self.targets = targets
        self.augment = augment

    def __len__(self):
        return len(self.raster_images)

    def __getitem__(self, idx):
        raster_image = self.raster_images[idx] 
        #raster_image = rasterio.open(raster_image_path).read() 
        #raster_image = np.transpose(raster_image, (1,2,0)) 
        if self.augment:
            raster_image = augment_raster_image(raster_image, include_swapping = include_swapping,
                                               swapping_range = swapping_range, 
                                               rotation_prob = rotation_prob, fliplr_prob = fliplr_prob)
            raster_image = np.ascontiguousarray(raster_image)
        extra_predictor = torch.tensor(self.extra_predictors[idx], dtype=torch.float32) 

        target = torch.tensor(self.targets[idx], dtype=torch.float32) 
        if crop_raster_image:
            raster_image = raster_image[:, crop_nr_border:-crop_nr_border, crop_nr_border:-crop_nr_border]
        return raster_image, extra_predictor, target

def run_cnn_with_results_for_target_column(index_col, train_images, 
                                           train_extra_predictors, train_targets, 
                                           val_images, val_extra_predictors, 
                                           val_targets, test_images, test_extra_predictors, 
                                           test_targets):
    
    #instantiation
    if crop_nr_border == 0:
        net = Net(extra_predictors_dim=60)
    elif crop_nr_border == 2:
        net = Net_5x5(extra_predictors_dim=63)
    elif crop_nr_border == 3:
        net = Net_3x3(extra_predictors_dim=63)
    net.to(device)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    dataset_val = CustomImageDataset(raster_images=val_images, extra_predictors = val_extra_predictors, targets=val_targets[:,index_col])
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=32)

    dataset_test = CustomImageDataset(raster_images=test_images, extra_predictors = test_extra_predictors, targets=test_targets[:,index_col])
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=32)
    
    maximum_val_loss = 10000
    es_count = 0
    train_loss = []
    val_loss = []
    RMSE_singlevar = []
    
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
    
        dataset_train = CustomImageDataset(raster_images=train_images, extra_predictors = train_extra_predictors, 
                                           targets=train_targets[:,index_col], augment = True)
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)


        net.train()
        running_loss = []
        for i, data in enumerate(dataloader_train):
            # get the inputs; data is a list of [inputs, labels]
            inputs, extra_predictors, labels = data
            inputs = inputs.clone().detach().to(device)
            extra_predictors = extra_predictors.clone().detach().to(device)
            labels = labels.clone().detach().to(device)


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs, extra_predictors)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss.append(loss.item())

        running_loss = np.mean(running_loss)  
        train_loss.append(running_loss)  


        net.eval()

        running_loss = []
        preds =  []
        all_labels = []

        for i, data in enumerate(dataloader_val):
            inputs, extra_predictors, labels = data
            inputs = inputs.clone().detach().to(device)
            extra_predictors = extra_predictors.clone().detach().to(device)
            labels = labels.clone().detach().to(device)
            with torch.no_grad():
                outputs = net(inputs, extra_predictors)
            loss = criterion(outputs, labels)


            for output, label in zip(outputs, labels):
                output = output.cpu()
                label = label.cpu()
                preds.append(output.item())
                all_labels.append(label.item())



            running_loss.append(loss.item())

        # Append RMSE values to respective lists
        preds = np.array(preds)
        all_labels = np.array(all_labels)
        RMSE_singlevar_epoch = np.sqrt(np.mean(np.square(preds-all_labels)))
        RMSE_singlevar.append(RMSE_singlevar_epoch)

        running_loss = np.mean(running_loss)
        val_loss.append(running_loss)
        scheduler.step()

        if val_loss[-1] > maximum_val_loss:
            es_count += 1
        else:
            maximum_val_loss = val_loss[-1]
            es_count = 0
            #save best model
            torch.save(net.state_dict(), 'best_model.pth')

        if es_count == patience:
            break
        
    #obtain test rmse  
    #load best model
    net.load_state_dict(torch.load('best_model.pth'))
    net.to(device)
    net.eval()

    running_loss = []
    preds =  []
    all_labels = []

    for i, data in enumerate(dataloader_test):
        inputs, extra_predictors, labels = data
        inputs = inputs.clone().detach().to(device)
        labels = labels.clone().detach().to(device)
        extra_predictors = extra_predictors.clone().detach().to(device)
        with torch.no_grad():
            outputs = net(inputs, extra_predictors)
        loss = criterion(outputs, labels)


        for output, label in zip(outputs, labels):

            output = output.cpu()
            label = label.cpu()
            preds.append(output.item())
            all_labels.append(label.item())


        running_loss.append(loss.item())
    preds = np.array(preds)
    all_labels = np.array(all_labels)
    rmse = compute_rmse(preds, all_labels)
    
    
    return rmse

def run_cnn_with_results(data, seed):
    data_train, data_val, data_test = create_train_val_test_split(data, seed, test_size)
    train_images, train_extra_predictors, train_targets, val_images, val_extra_predictors, val_targets, test_images, test_extra_predictors, test_targets = prepare_datasets_for_cnn(data_train, data_val, data_test)
    image_scaler, image_imputer, extra_predictor_scaler, extra_predictor_imputer = fit_imputer_and_scaler(train_images, train_extra_predictors)
    train_images, train_extra_predictors = transform_data(train_images, train_extra_predictors, image_scaler, image_imputer, extra_predictor_scaler, extra_predictor_imputer)
    val_images, val_extra_predictors = transform_data(val_images, val_extra_predictors, image_scaler, image_imputer, extra_predictor_scaler, extra_predictor_imputer)
    test_images, test_extra_predictors = transform_data(test_images, test_extra_predictors, image_scaler, image_imputer, extra_predictor_scaler, extra_predictor_imputer)
    results = []
    result = run_cnn_with_results_for_target_column(0, train_images, 
                                           train_extra_predictors, train_targets, 
                                           val_images, val_extra_predictors, 
                                           val_targets, test_images, test_extra_predictors, 
                                           test_targets)
    results.append(result)
    result = run_cnn_with_results_for_target_column(1, train_images, 
                                           train_extra_predictors, train_targets, 
                                           val_images, val_extra_predictors, 
                                           val_targets, test_images, test_extra_predictors, 
                                           test_targets)
    results.append(result)
    result = run_cnn_with_results_for_target_column(2, train_images, 
                                           train_extra_predictors, train_targets, 
                                           val_images, val_extra_predictors, 
                                           val_targets, test_images, test_extra_predictors)
 
    results.append(result)
    return results