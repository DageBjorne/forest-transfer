from utils import *
from regression_seed import *
from xgboost_seed import *
from CNN_concat_seed import *
from dataloading import *
from utils import *
import pandas as pd

nfi_data_path = r'Image_data\merged_data_cleaned_without_nans.csv'
raster_data_path = r'Image_data\ALS_raster_metrics_plot_center_cut_90x90'
saving_path = 'cv_data_nonans.csv'
if os.path.exists(saving_path) == False:
    cv_data = pd.DataFrame(columns = ['seed', 
                                     'regression', 'xgboost', 'cnn'])

    cv_data.to_csv(saving_path)

cv_data = pd.read_csv(saving_path, index_col = [0])

new_cv_row = pd.DataFrame(columns = ['seed', 
                                    'regression', 'xgboost', 'cnn'])

nfi_data = read_nfi_data(nfi_data_path)
seed_list = [1,2,3,4,5,6,7,8,9,10]
for seed in seed_list:
    new_cv_row.at[0, 'seed'] = seed
    results = run_regression_with_results(seed, nfi_data)
    new_cv_row.at[0, 'regression'] = results
    print(results)
    results = run_xg_boost_with_results_for_all_target_columns(seed, nfi_data)
    new_cv_row.at[0, 'xgboost'] = results
    print(results)
    results = run_cnn_with_results(nfi_data, seed)
    new_cv_row.at[0, 'cnn'] = results
    print(results)
    cv_data = pd.concat([cv_data, new_cv_row], ignore_index=True)
    cv_data.to_csv(saving_path)
