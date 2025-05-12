target_columns = ['H_AVERAGE', 'D_AVERAGE', 'VOLUME']

predictor_columns_regression = ['zq30', 'zq80','zq90', 'zq95', 
                                'pzabovezmean', 'zsd', 'n', 'p1th', 
                                'p2th', 'p3th','p4th','itot']

predictor_columns = ['zmax', 'zmean', 'zsd',
    'zskew', 'zkurt', 'zentropy', 'pzabovezmean', 'pzabove2', 'zq5', 'zq10',
    'zq15', 'zq20', 'zq25', 'zq30', 'zq35', 'zq40', 'zq45', 'zq50', 'zq55',
    'zq60', 'zq65', 'zq70', 'zq75', 'zq80', 'zq85', 'zq90', 'zq95',
    'zpcum1', 'zpcum2', 'zpcum3', 'zpcum4', 'zpcum5', 'zpcum6', 'zpcum7',
    'zpcum8', 'zpcum9', 'itot', 'imax', 'imean', 'isd', 'iskew', 'ikurt',
    'ipground', 'ipcumzq10', 'ipcumzq30', 'ipcumzq50', 'ipcumzq70',
    'ipcumzq90', 'p1th', 'p2th', 'p3th', 'p4th', 'p5th', 'pground', 'n',
    'area', 'l_1', 'l_2', 't_3', 't_4']


test_size = 0.05
val_size = 0.15

#cnn settings
BATCH_SIZE = 64
EPOCHS = 50
learning_rate = 5e-05
patience = 5

#augmentation settings
include_swapping = False
swapping_range = [0,15]
rotation_prob = 0.25
fliplr_prob = 0.5
crop_raster_image = False
crop_nr_border = 0 #1 for removing one border, 2for etc...
