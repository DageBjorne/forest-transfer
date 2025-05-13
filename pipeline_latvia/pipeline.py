from utils import *
from regression_seed import *
from xgboost_seed import *
from CNN_concat_seed import *
from dataloading import *
from utils import *
import pandas as pd

nfi_data_path = r'./Latvian_Image_data/merged_data_cleaned.csv'
raster_data_path = r'./Latvian_Image_data/tif_surrounding_processed'
saving_path = './res/results.csv'
if os.path.exists(saving_path) == False:
    cv_data = pd.DataFrame(columns = ['seed', 
                                      'xgboost', 'cnn'])

    cv_data.to_csv(saving_path)

cv_data = pd.read_csv(saving_path, index_col = [0])

new_cv_row = pd.DataFrame(columns = ['seed', 
                                    'xgboost', 'cnn'])

nfi_data = read_nfi_data(nfi_data_path)
seed_list = [3,4,5]
for seed in seed_list:
    new_cv_row.at[0, 'seed'] = seed
    results = run_xg_boost_with_results_for_all_target_columns(seed, nfi_data)
    new_cv_row.at[0, 'xgboost'] = results
    print(results)
    results = run_cnn_with_results(nfi_data, seed)
    new_cv_row.at[0, 'cnn'] = results
    print(results)
    cv_data = pd.concat([cv_data, new_cv_row], ignore_index=True)
    cv_data.to_csv(saving_path)
