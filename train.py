"""
 @file train.py
 @brief Script for training
 @author Yisen Liu
 Copyright (C) 2022 Institute of Intelligent Manufacturing, Guangdong Academy of Sciences. All right reserved.
"""

########################################################################
# import default python-library
########################################################################
import os
import random

import joblib
import numpy as np
import torch
import torch.nn as nn
from skimage.transform import rotate
from sklearn.decomposition import PCA

import common as com
import torch_model

########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################


def load_normal_data(itr):

	data_file = os.path.join(param["data_directory"],'blueberry_healthy.npy')
	normal_data = np.load(data_file)
	print(normal_data.shape) #(808, 60, 60, 181)

	# #split train and test
	data_size = normal_data.shape[0]
	random.seed(itr)
	shuffle_index = list(range(data_size))  # shuffle
	random.shuffle(shuffle_index)
	normal_train_data = normal_data.copy()[shuffle_index[0:int(1 / 2 * data_size)]]

	#train pca model
	data_eff_all = np.zeros((0,normal_train_data.shape[-1]))

	for i in range (normal_train_data.shape[0]):
		nonzero_idx = np.nonzero(normal_train_data[i,:,:,100])
		nonzero_idx = np.array(nonzero_idx)
		nonzero_size = nonzero_idx[0].size

		data_eff = np.zeros((nonzero_size,normal_train_data.shape[-1]))

		for k in range(0,nonzero_size):
			w_idx = nonzero_idx[0,k]
			h_idx = nonzero_idx[1,k]
			data_eff[k,:] = normal_train_data[i,w_idx,h_idx,:]

		data_eff_all = np.append(data_eff_all,data_eff,axis=0)

	pca = PCA(n_components=10)
	print('training pca model')

	pca_model = pca.fit(data_eff_all)

	print('PCA_explained_ratio:',pca_model.explained_variance_ratio_)
	pca_model_file_path = f"{param['model_directory']}/pca_model_{fruit_type}_{itr}itr.model"
	joblib.dump(pca_model,pca_model_file_path)
	
	data_pca_all = pca.transform(data_eff_all)

	max_value = np.zeros((10))
	min_value = np.zeros((10))

	for i in range (10):
		max_value[i] = np.max(data_pca_all[:,i])
		min_value[i] = np.min(data_pca_all[:,i])

	data_file = os.path.join(param["model_directory"],f'pca_max_{itr}itr.npy')
	np.save(data_file,max_value)
	data_file = os.path.join(param["model_directory"],f'pca_min_{itr}itr.npy')
	np.save(data_file,min_value)
	print(max_value)
	print(min_value)

	# make pc images
	print('making pca pic')
	data_pca = np.zeros((normal_train_data.shape[0],60,60,10))
	for i in range (normal_train_data.shape[0]):
		nonzero_idx = np.nonzero(normal_train_data[i,:,:,100]) 
		nonzero_idx = np.array(nonzero_idx)
		nonzero_size = nonzero_idx[0].size  
		data_eff = np.zeros((nonzero_size, normal_train_data.shape[3]))  
		for k in range(0,nonzero_size): 
			w_idx = nonzero_idx[0,k]
			h_idx = nonzero_idx[1,k]
			data_eff[k,:] = normal_train_data[i, w_idx, h_idx, :]
		data_eff_pca = pca.transform(data_eff)
		
		#normalization
		for j in range (data_eff_pca.shape[1]):
			data_eff_pca[:,j] = (data_eff_pca[:,j] - min_value[j])/(max_value[j] - min_value[j])

		for k in range(0,nonzero_size): 
			w_idx = nonzero_idx[0,k]
			h_idx = nonzero_idx[1,k]
			data_pca[i,w_idx,h_idx,:] = data_eff_pca[k,:]

	train_data = data_pca[:,:,:,0:5]
	class_num = train_data[:,:,:,0].shape[0]
	label = np.array([i for i in range(5)]*class_num)

	train_data_pca = np.transpose(train_data,[0,3,2,1])
	train_data_pca = train_data_pca.reshape((-1,train_data_pca.shape[-2],train_data_pca.shape[-1]))
	
	return train_data_pca, label


def train_step(train_normal_data,train_label,step,batch_size,model,optimizer,criterion):
		data = train_normal_data[step*batch_size:(step+1)*batch_size]
		label = train_label[step*batch_size:(step+1)*batch_size]
		# zero_grad
		optimizer.zero_grad()
		pred,_ = model(data)
		loss = criterion(pred,label)

		# backward
		loss.backward()
		optimizer.step()

		return loss


########################################################################
# main train.py
########################################################################
if __name__ == "__main__":
	device = torch.device(0)
	# make output directory
	os.makedirs(param["model_directory"], exist_ok=True)
	
	fruit_type = 'blueberry'
	pca_nm = 5
	batch_size = 256
	epoch = 150

	# train self-supervised model
	print("============== SS MODEL TRAINING ==============")

	
	# 10 random sampling runs
	for itr in range (10):

		model = torch_model.ss_model().to(device)
		print(model)

		SS_model_file_path = f'model/SS_model_{fruit_type}_{pca_nm}pc_{itr}itr_model.pkl'

		#load data
		train_normal_data,train_label = load_normal_data(itr)
		
		train_normal_data_rotate_90 = np.zeros_like(train_normal_data)
		train_normal_data_rotate_180 = np.zeros_like(train_normal_data)
		train_normal_data_rotate_270 = np.zeros_like(train_normal_data)

		for i in range (train_normal_data.shape[0]):
			train_normal_data_rotate_90[i] = rotate(train_normal_data[i],90)
			train_normal_data_rotate_180[i] = rotate(train_normal_data[i],180)
			train_normal_data_rotate_270[i] = rotate(train_normal_data[i],270)

		train_normal_data = np.concatenate((train_normal_data,train_normal_data_rotate_90,train_normal_data_rotate_180,train_normal_data_rotate_270),axis=0)
		train_label = np.concatenate([train_label]*4,axis=0)
		
		train_normal_data = torch.from_numpy(train_normal_data).float().to(device)
		train_normal_data = train_normal_data.unsqueeze(1)
		train_label = torch.from_numpy(train_label).long().to(device)

		#define optimizer and loss
		optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
		criterion= nn.CrossEntropyLoss().to(device)

		#training model
		model.train(True)
		
		indexs = [i for i in range(train_normal_data.shape[0])]

		print(f'begin training,{itr}itr' )

		for i in range(epoch):
			np.random.shuffle(indexs)
			train_normal_data = train_normal_data[indexs]
			train_label = train_label[indexs]

			for step in range(train_normal_data.shape[0]//batch_size):
				if (step+1)*batch_size >= train_normal_data.shape[0]:
					break
				loss = train_step(train_normal_data,train_label,step,batch_size,model,optimizer,criterion)

			print(f'epoch {i} loss:{loss:.5f}')

			# save model
			if loss<0.1:
				torch.save(model.state_dict(), SS_model_file_path)
				break

			if i == 150:
				torch.save(model.state_dict(), SS_model_file_path)
