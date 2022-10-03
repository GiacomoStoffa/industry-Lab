import numpy as np
import joblib

targets = ['livello', 'portata']
prevs = [1, 3, 5, 7, 14, 21, 28, 42, 56, 70, 84, 98, 112]
offsets = [1, 3, 5, 7, 14, 21, 28]



def import_files(base_path, targets, prevs):
  X_train_tot, X_val_tot, X_test_tot = {}, {}, {}
  y_train_tot, y_val_tot, y_test_tot = {}, {}, {}

  for target in targets:
    X_train_tmp1, X_val_tmp1, X_test_tmp1 = {}, {}, {}
    y_train_tmp1, y_val_tmp1, y_test_tmp1 = {}, {}, {}

    for prev in prevs:
      X_train_tmp2, X_val_tmp2, X_test_tmp2 = {}, {}, {}
      y_train_tmp2, y_val_tmp2, y_test_tmp2 = {}, {}, {}

      for offset in offsets:
        X_train_tmp2[offset] = np.load(base_path+'Data/Prepared/offset'+str(offset)+'_prev'+str(prev)+'/train_prep_'+target+'_x.npy')
        X_val_tmp2[offset] = np.load(base_path+'Data/Prepared/offset'+str(offset)+'_prev'+str(prev)+'/val_prep_'+target+'_x.npy')
        X_test_tmp2[offset] = np.load(base_path+'Data/Prepared/offset'+str(offset)+'_prev'+str(prev)+'/test_prep_'+target+'_x.npy')
        y_train_tmp2[offset] = np.load(base_path+'Data/Prepared/offset'+str(offset)+'_prev'+str(prev)+'/train_prep_'+target+'_y.npy')
        y_val_tmp2[offset] = np.load(base_path+'Data/Prepared/offset'+str(offset)+'_prev'+str(prev)+'/val_prep_'+target+'_y.npy')
        y_test_tmp2[offset] = np.load(base_path+'Data/Prepared/offset'+str(offset)+'_prev'+str(prev)+'/test_prep_'+target+'_y.npy')

      X_train_tmp1[prev] = X_train_tmp2
      X_val_tmp1[prev] = X_val_tmp2
      X_test_tmp1[prev] = X_test_tmp2
      y_train_tmp1[prev] = y_train_tmp2
      y_val_tmp1[prev] = y_val_tmp2
      y_test_tmp1[prev] = y_test_tmp2
    
    X_train_tot[target] = X_train_tmp1
    X_val_tot[target] = X_val_tmp1
    X_test_tot[target] = X_test_tmp1
    y_train_tot[target] = y_train_tmp1
    y_val_tot[target] = y_val_tmp1
    y_test_tot[target] = y_test_tmp1

  return {
    'X_train': X_train_tot, 
    'X_val': X_val_tot, 
    'X_test': X_test_tot, 
    'y_train': y_train_tot, 
    'y_val': y_val_tot, 
    'y_test': y_test_tot, 
  }



def get_data(data_tot, target, prev, offset):
  X_train = data_tot['X_train'][target][prev][offset].copy()
  X_val = data_tot['X_val'][target][prev][offset].copy()
  X_test = data_tot['X_test'][target][prev][offset].copy()
  y_train = data_tot['y_train'][target][prev][offset].copy()
  y_val = data_tot['y_val'][target][prev][offset].copy()
  y_test = data_tot['y_test'][target][prev][offset].copy()
  return X_train, X_val, X_test, y_train, y_val, y_test



def get_scaler(base_path):
  return {
      'livello': joblib.load(base_path+'Models/scaler_livello.joblib'), 
      'portata': joblib.load(base_path+'Models/scaler_portata.joblib') 
  }