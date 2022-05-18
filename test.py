"""
 @file test.py
 @brief Script for testing
 @author Yisen Liu
 Copyright (C) 2022 Institute of Intelligent Manufacturing, Guangdong Academy of Sciences. All right reserved.
"""

########################################################################
# import python-library
########################################################################
import csv
import os
import random
import sys

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics

import common as com
import torch_model

########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
#######################################################################

def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


def load_normal_test_data(itr):

    # load_normal_data
    print('loading normal test data ...')
    data_file = os.path.join(param["data_directory"],'blueberry_healthy.npy')
    normal_data = np.load(data_file)

    #split train and test 
    random.seed(itr)
    shuffle_index = list(range(normal_data.shape[0]))  # shuffle
    random.shuffle(shuffle_index)
    data_size = normal_data.shape[0]
    normal_test_data = normal_data.copy()[shuffle_index[int(1 / 2 * data_size):]]
    y_true_normal = np.zeros((normal_test_data.shape[0]))

    normal_test_data_output = make_pc_images(normal_test_data)

    return normal_test_data_output, y_true_normal


def load_abnormal_test_data(itr):

    # load_abnormal_data
    data_file = os.path.join(param["data_directory"],'blueberry_bruise_new.npy')
    abnormal_data_1 = np.load(data_file)
    print('bruise:%d'%abnormal_data_1.shape[0])

    data_file = os.path.join(param["data_directory"],'blueberry_chiling.npy')
    abnormal_data_2 = np.load(data_file)
    print('chilling:%d'%abnormal_data_2.shape[0])

    data_file = os.path.join(param["data_directory"],'blueberry_infection.npy')
    abnormal_data_3 = np.load(data_file)
    print('infection:%d'%abnormal_data_3.shape[0])

    data_file = os.path.join(param["data_directory"],'blueberry_wrinkled.npy')
    abnormal_data_4 = np.load(data_file)
    abnormal_data_4 = abnormal_data_4[60:]
    print('wrinkled:%d'%abnormal_data_4.shape[0])

    abnormal_test_data = np.concatenate([abnormal_data_1,abnormal_data_2,abnormal_data_3,abnormal_data_4],axis=0)
    print(abnormal_test_data.shape)

    abnormal_test_data_output = make_pc_images(abnormal_test_data)
    y_true_abnormal = np.ones((abnormal_test_data_output.shape[0]))
    abnormal_size = [abnormal_data_1.shape[0],abnormal_data_2.shape[0],abnormal_data_3.shape[0],abnormal_data_4.shape[0]]

    return abnormal_test_data_output, y_true_abnormal, abnormal_size


def load_normal_train_data(itr):
    # load_normal_data

    data_file = os.path.join(param["data_directory"],'blueberry_healthy.npy')
    normal_data = np.load(data_file)

    #split train and test 
    random.seed(itr)
    shuffle_index = list(range(normal_data.shape[0]))  # shuffle
    random.shuffle(shuffle_index)
    normal_train_data = normal_data.copy()[shuffle_index[0:int(1 / 2 * normal_data.copy().shape[0])]]

    normal_train_data_output = make_pc_images(normal_train_data)

    return normal_train_data_output


def test_step(test_data):
    with torch.no_grad() :
        pred,fc_feature = model(test_data)
        pred = torch.softmax(pred,dim=-1)
        # print(pred)
    return pred, fc_feature


def make_pc_images(data):

    #load pca model
    pca_model_file_path = f"{param['model_directory']}/pca_model_{fruit_type}_{itr}itr.model"
    pca = joblib.load(pca_model_file_path)

    data_pca = np.zeros((data.shape[0],60,60,10))
    for i in range (data.shape[0]):
        nonzero_idx = np.nonzero(data[i,:,:,100]) 
        nonzero_idx = np.array(nonzero_idx)
        nonzero_size = nonzero_idx[0].size  
        data_eff = np.zeros((nonzero_size, data.shape[3]))  
        for k in range(0,nonzero_size): 
            w_idx = nonzero_idx[0,k]
            h_idx = nonzero_idx[1,k]
            data_eff[k,:] = data[i,w_idx,h_idx,:]
        data_eff_pca = pca.transform(data_eff)

        max_value = np.load(os.path.join(param["model_directory"],f'pca_max_{itr}itr.npy'))
        min_value = np.load(os.path.join(param["model_directory"],f'pca_min_{itr}itr.npy'))
        
        #normalization
        for j in range (data_eff_pca.shape[1]):
            data_eff_pca[:,j] = (data_eff_pca[:,j]-min_value[j])/(max_value[j]-min_value[j])

        for k in range(0,nonzero_size): 
            w_idx = nonzero_idx[0,k]
            h_idx = nonzero_idx[1,k]
            data_pca[i,w_idx,h_idx,:] = data_eff_pca[k,:]

    result_data = data_pca[:,:,:,0:5]

    return result_data


########################################################################
# main test.py
########################################################################
if __name__ == "__main__":

    # make output result directory
    os.makedirs(param["result_directory"], exist_ok=True)
    device = torch.device(0)

    # initialize lines in csv for anomaly detection results 
    csv_lines = []
    csv_lines.append(["AUC", "F1 score","acc_normal","acc_bruise","acc_chilling","acc_infection","acc_wrinkled"])

    print("============== MODEL LOAD ==============")
    pca_nm = 5
    auc_total = np.zeros((10))
    f1_total = np.zeros((10))
    acc_normal_total = np.zeros((10))
    acc_bruise_total = np.zeros((10))
    acc_chilling_total = np.zeros((10))
    acc_infection_total = np.zeros((10))
    acc_wrinkled_total = np.zeros((10))
    fruit_type = 'blueberry'
    cosinesimilarity = nn.CosineSimilarity(dim=-1)

    for itr in range (10):
        # set model path
        
        SS_model_file_path = f'model/SS_model_{fruit_type}_{pca_nm}pc_{itr}itr_model.pkl'

        # load test file
        normal_test_data,y_true_normal = load_normal_test_data(itr)
        abnormal_test_data,y_true_abnormal,abnormal_size = load_abnormal_test_data(itr)
        normal_train_data = load_normal_train_data(itr)
        normal_test_data = normal_test_data.reshape((-1,60,60,5))
        abnormal_test_data = abnormal_test_data.reshape((-1,60,60,5))
        normal_train_data = normal_train_data.reshape((-1,60,60,5))

        y_true_normal = np.zeros((normal_test_data.shape[0]))
        y_true_abnormal = np.ones((abnormal_test_data.shape[0]))
        y_true = np.concatenate([y_true_normal,y_true_abnormal],axis=0)
        
        normal_test_data = np.transpose(normal_test_data,[0,3,2,1])
        normal_test_data = normal_test_data.reshape((-1,1,60,60))
        
        abnormal_test_data = np.transpose(abnormal_test_data,[0,3,2,1])
        abnormal_test_data = abnormal_test_data.reshape((-1,1,60,60))

        normal_train_data = np.transpose(normal_train_data,[0,3,2,1])
        normal_train_data = normal_train_data.reshape((-1,1,60,60))

        test_data = np.concatenate((normal_test_data, abnormal_test_data),axis=0)

        # setup anomaly score file path
        anomaly_score_csv = f"{param['result_directory']}/anomaly_score_{fruit_type}_{itr}itr.csv"
        #initialize anomaly score list 
        anomaly_score_list = []

        print("\n============== BEGIN TEST ==============")
        # load model file

        model = torch_model.ss_model().to(device)
        model.load_state_dict(torch.load(SS_model_file_path))

        model.eval()    
        
        test_data = torch.from_numpy(test_data).float().to(device)
        pred_pc, feature = test_step(test_data)
        feature = feature.reshape((-1,16*5))

        normal_train_data = torch.from_numpy(normal_train_data).float().to(device)
        pred_train, feature_train = test_step(normal_train_data)

        feature_train = feature_train.reshape((-1,16*5))

        feature_cosine_errors = []
        for i in range(feature.shape[0]):
            cos_simil = cosinesimilarity(feature[i], feature_train)
            feature_cosine_errors.append(cos_simil.mean().item())

        errors = np.array(feature_cosine_errors)
        y_pred = -errors
 
        # save anomaly scores
        for i in range(y_true.shape[0]):
            anomaly_score_list.append([y_true[i], y_pred[i]])

        save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)

        print("\n============ END OF TEST ============")

        #caculate AUC 
        auc = metrics.roc_auc_score(y_true, y_pred)
        print('auc:',auc)

        #decision_making
        decision = np.zeros((y_pred.shape[0]))
        index = numpy.argsort(y_pred)
        normal_num = normal_test_data.shape[0] // 5
        decision[index[0:normal_num]] = 0
        decision[index[normal_num:]] = 1
        
        #caculate F1 score
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, decision).ravel()
        prec = tp / np.maximum(tp + fp, sys.float_info.epsilon)
        recall = tp / np.maximum(tp + fn, sys.float_info.epsilon)
        f1 = 2.0 * prec * recall / np.maximum(prec + recall, sys.float_info.epsilon)
        print('f1:',f1)

        #caculate Acc
        acc_normal = 1-np.sum(decision[0:y_true_normal.shape[0]])/y_true_normal.shape[0]
        acc_bruise = np.sum(decision[y_true_normal.shape[0]:y_true_normal.shape[0]+abnormal_size[0]])/abnormal_size[0]
        acc_chilling = np.sum(decision[y_true_normal.shape[0]+abnormal_size[0]:y_true_normal.shape[0]+np.sum(abnormal_size[:2])])/abnormal_size[1]
        acc_infection = np.sum(decision[y_true_normal.shape[0]+np.sum(abnormal_size[:2]):y_true_normal.shape[0]+np.sum(abnormal_size[:3])])/abnormal_size[2]
        acc_wrinkled = np.sum(decision[y_true_normal.shape[0]+np.sum(abnormal_size[:3]):])/abnormal_size[-1]
        
        print('acc_normal:',acc_normal)
        print('acc_bruise:',acc_bruise)
        print('acc_chilling:',acc_chilling)
        print('acc_infection:',acc_infection)
        print('acc_wrinkled:',acc_wrinkled)

        csv_lines.append(['itr'+str(itr), auc, f1,acc_normal,acc_bruise,acc_chilling,acc_infection,acc_wrinkled])

        auc_total[itr] = auc
        f1_total[itr] = f1
        acc_normal_total[itr] = acc_normal
        acc_bruise_total[itr] = acc_bruise
        acc_chilling_total[itr] = acc_chilling
        acc_infection_total[itr] = acc_infection
        acc_wrinkled_total[itr] = acc_wrinkled

csv_lines.append(['total_mean', np.mean(auc_total), np.mean(f1_total),np.mean(acc_normal_total),np.mean(acc_bruise_total),np.mean(acc_chilling_total),np.mean(acc_infection_total),np.mean(acc_wrinkled_total)])

#calculate 95_interval
auc_interval = 1.96*np.std(auc_total)/(10**0.5)
f1_interval = 1.96*np.std(f1_total)/(10**0.5)
acc_normal_interval = 1.96*np.std(acc_normal_total)/(10**0.5)
acc_bruise_interval = 1.96*np.std(acc_bruise_total)/(10**0.5)
acc_chilling_interval = 1.96*np.std(acc_chilling_total)/(10**0.5)
acc_infection_interval = 1.96*np.std(acc_infection_total)/(10**0.5)
acc_wrinkled_interval = 1.96*np.std(acc_wrinkled_total)/(10**0.5)
csv_lines.append(['95_interval', np.mean(auc_interval), np.mean(f1_interval),np.mean(acc_normal_interval),np.mean(acc_bruise_interval),np.mean(acc_chilling_interval),np.mean(acc_infection_interval),np.mean(acc_wrinkled_interval)])

# save results
result_path = f"{param['result_directory']}/{param['result_file']}"
save_csv(save_file_path=result_path, save_data=csv_lines)
