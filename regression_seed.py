import pandas as pd
import numpy as np
import rasterio
from dataloading import *
from utils import *
import os
import statsmodels.api as sm
from config import *
from regression_models import *
def train_regression(data, target_column, formula):
    formula = target_column + ' ~ ' + formula
    model = sm.formula.ols(formula=formula, data=data).fit()
    return model


def test_regression(data, model):
    preds = model.predict(data)
    return preds

def run_regression_with_results(seed, data):
    rmses = []
    data['Volume_srt'] = np.sqrt(data['Volume'])
    data['Biomassa_above_sqrt'] = np.sqrt(data['Biomassa_above'])

    data_train, data_test = create_train_test_split_for_regression(data, seed, test_size)
    #divide test set into regions
    filtered_data_test_1 = filter_test_data_based_on_region(data_test, 1)
    filtered_data_test_2 = filter_test_data_based_on_region(data_test, 2)
    filtered_data_test_3 = filter_test_data_based_on_region(data_test, 3)
    filtered_data_test_4 = filter_test_data_based_on_region(data_test, 4)
    #train the model on all regions
    target_column = 'Hgv'
    model = train_regression(data_train, target_column, height_formula)
    #evaluate on all regions
    preds = model.predict(data_test)
    rmse = compute_rmse(preds, data_test[target_column])
    #evaluate on different regions
    preds1 = model.predict(filtered_data_test_1)
    rmse1 = compute_rmse(preds1, filtered_data_test_1[target_column])
    preds2 = model.predict(filtered_data_test_2)
    rmse2 = compute_rmse(preds2, filtered_data_test_2[target_column])
    preds3 = model.predict(filtered_data_test_3)
    rmse3 = compute_rmse(preds3, filtered_data_test_3[target_column])
    preds4 = model.predict(filtered_data_test_4)
    rmse4 = compute_rmse(preds4, filtered_data_test_4[target_column])
    rmses.append([rmse, rmse1, rmse2, rmse3, rmse4])

    target_column = 'Dgv'
    rmse_list = []
    models_list = []

    model_candidate = train_regression(data_train, target_column, diameter_formula_1)
    preds = model_candidate.predict(data_test)
    rmse = compute_rmse(preds, data_test[target_column])
    rmse_list.append(rmse)
    models_list.append(model_candidate)

    model_candidate = train_regression(data_train, target_column, diameter_formula_2)
    preds = model_candidate.predict(data_test)
    rmse = compute_rmse(preds, data_test[target_column])
    rmse_list.append(rmse)
    models_list.append(model_candidate)

    model_candidate = train_regression(data_train, target_column, diameter_formula_3)
    preds = model_candidate.predict(data_test)
    rmse = compute_rmse(preds, data_test[target_column])
    rmse_list.append(rmse)
    models_list.append(model_candidate)

    model_candidate = train_regression(data_train, target_column, diameter_formula_4)
    preds = model_candidate.predict(data_test)
    rmse = compute_rmse(preds, data_test[target_column])
    rmse_list.append(rmse)
    models_list.append(model_candidate)

    best_index = np.argmin(rmse_list)
    model = models_list[best_index]

    #evaluate on all regions
    preds = model.predict(data_test)
    rmse = compute_rmse(preds, data_test[target_column])
    #evaluate on different regions
    preds1 = model.predict(filtered_data_test_1)
    rmse1 = compute_rmse(preds1, filtered_data_test_1[target_column])
    preds2 = model.predict(filtered_data_test_2)
    rmse2 = compute_rmse(preds2, filtered_data_test_2[target_column])
    preds3 = model.predict(filtered_data_test_3)
    rmse3 = compute_rmse(preds3, filtered_data_test_3[target_column])
    preds4 = model.predict(filtered_data_test_4)
    rmse4 = compute_rmse(preds4, filtered_data_test_4[target_column])
    rmses.append([rmse, rmse1, rmse2, rmse3, rmse4])

    target_column = 'Basal_area'
    rmse_list = []
    models_list = []
    model_candidate = train_regression(data_train, target_column, basal_area_formula_1)
    preds = model_candidate.predict(data_test)
    rmse = compute_rmse(preds, data_test[target_column])
    rmse_list.append(rmse)
    models_list.append(model_candidate)

    model_candidate = train_regression(data_train, target_column, basal_area_formula_2)
    preds = model_candidate.predict(data_test)
    rmse = compute_rmse(preds, data_test[target_column])
    rmse_list.append(rmse)
    models_list.append(model_candidate)

    best_index = np.argmin(rmse_list)
    model = models_list[best_index]

    #evaluate on all regions
    preds = model.predict(data_test)
    rmse = compute_rmse(preds, data_test[target_column])
    #evaluate on different regions
    preds1 = model.predict(filtered_data_test_1)
    rmse1 = compute_rmse(preds1, filtered_data_test_1[target_column])
    preds2 = model.predict(filtered_data_test_2)
    rmse2 = compute_rmse(preds2, filtered_data_test_2[target_column])
    preds3 = model.predict(filtered_data_test_3)
    rmse3 = compute_rmse(preds3, filtered_data_test_3[target_column])
    preds4 = model.predict(filtered_data_test_4)
    rmse4 = compute_rmse(preds4, filtered_data_test_4[target_column])
    rmses.append([rmse, rmse1, rmse2, rmse3, rmse4])

    target_column_for_regression = 'Volume_srt'
    target_column = 'Volume'
    rmse_list = []
    models_list = []
    model_candidate = train_regression(data_train, target_column_for_regression, volume_formula_1)
    preds = model_candidate.predict(data_test)
    rmse = compute_rmse(np.square(preds), data_test[target_column])
    rmse_list.append(rmse)
    models_list.append(model_candidate)

    model_candidate = train_regression(data_train, target_column_for_regression, volume_formula_2)
    preds = model_candidate.predict(data_test)
    rmse = compute_rmse(np.square(preds), data_test[target_column])
    rmse_list.append(rmse)
    models_list.append(model_candidate)

    model_candidate = train_regression(data_train, target_column_for_regression, volume_formula_3)
    preds = model_candidate.predict(data_test)
    rmse = compute_rmse(np.square(preds), data_test[target_column])
    rmse_list.append(rmse)
    models_list.append(model_candidate)

    model_candidate = train_regression(data_train, target_column_for_regression, volume_formula_4)
    preds = model_candidate.predict(data_test)
    rmse = compute_rmse(np.square(preds), data_test[target_column])
    rmse_list.append(rmse)
    models_list.append(model_candidate)

    best_index = np.argmin(rmse_list)
    model = models_list[best_index]

    #evaluate on all regions
    preds = model.predict(data_test)
    rmse = compute_rmse(np.square(preds), data_test[target_column])
    #evaluate on different regions
    preds1 = model.predict(filtered_data_test_1)
    rmse1 = compute_rmse(np.square(preds1), filtered_data_test_1[target_column])
    preds2 = model.predict(filtered_data_test_2)
    rmse2 = compute_rmse(np.square(preds2), filtered_data_test_2[target_column])
    preds3 = model.predict(filtered_data_test_3)
    rmse3 = compute_rmse(np.square(preds3), filtered_data_test_3[target_column])
    preds4 = model.predict(filtered_data_test_4)
    rmse4 = compute_rmse(np.square(preds4), filtered_data_test_4[target_column])
    rmses.append([rmse, rmse1, rmse2, rmse3, rmse4])

    target_column_for_regression = 'Biomassa_above_sqrt'
    target_column = 'Biomassa_above'
    rmse_list = []
    models_list = []
    model_candidate = train_regression(data_train, target_column_for_regression, biomassa_above_formula_1)
    preds = model_candidate.predict(data_test)
    rmse = compute_rmse(np.square(preds), data_test[target_column])
    rmse_list.append(rmse)
    models_list.append(model_candidate)

    model_candidate = train_regression(data_train, target_column_for_regression, biomassa_above_formula_2)
    preds = model_candidate.predict(data_test)
    rmse = compute_rmse(np.square(preds), data_test[target_column])
    rmse_list.append(rmse)
    models_list.append(model_candidate)

    best_index = np.argmin(rmse_list)
    model = models_list[best_index]

    #evaluate on all regions
    preds = model.predict(data_test)
    rmse = compute_rmse(np.square(preds), data_test[target_column])
    #evaluate on different regions
    preds1 = model.predict(filtered_data_test_1)
    rmse1 = compute_rmse(np.square(preds1), filtered_data_test_1[target_column])
    preds2 = model.predict(filtered_data_test_2)
    rmse2 = compute_rmse(np.square(preds2), filtered_data_test_2[target_column])
    preds3 = model.predict(filtered_data_test_3)
    rmse3 = compute_rmse(np.square(preds3), filtered_data_test_3[target_column])
    preds4 = model.predict(filtered_data_test_4)
    rmse4 = compute_rmse(np.square(preds4), filtered_data_test_4[target_column])
    rmses.append([rmse, rmse1, rmse2, rmse3, rmse4])
    return rmses



