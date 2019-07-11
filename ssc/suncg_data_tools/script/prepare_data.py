import sys
sys.path.append('../lib')
import SUNCGData
import numpy as np
import os
import msgpack
import msgpack_numpy as m
m.patch()
from multiprocessing import Pool as ThreadPool 
import torch

gridsize = np.asarray([240,144,240])
downscale = 4
gridsize_downscale = (gridsize / downscale).astype(int)
suncg_depthbin_dir = '../../data/depthbin/'
total_gpu_num = 1 
#total_gpu_num = torch.cuda.device_count()
output_dir = '../../data/'


if not os.path.exists(output_dir+'msg/'):
	print('prepare suncg datas ...')
	os.system('mkdir -p '+ output_dir+'msg/train ' + output_dir+'msg/test')

def getDataFiles(data_lists):
	data_files = SUNCGData.getDataFiles(data_lists)
	data_files_valid = []
	for depth_file in data_files:
		bin_file = depth_file[:-3] + 'bin'
		if os.path.isfile(depth_file) and os.path.isfile(bin_file):
			data_files_valid.append(depth_file)
	print('total vaild files:' + str(len(data_files_valid)))
	return data_files_valid

def loadDataFile(depth_file, gpu_num):
	bin_file = depth_file[:-3] + 'bin'
	label_full = SUNCGData.getLabels(bin_file).reshape(gridsize[::-1])
	fliped_tsdf = SUNCGData.getTSDF(depth_file, bin_file, gpu_num).reshape(gridsize[::-1])
	if downscale > 1:
		label_downscale = SUNCGData.downsampleLabel(label_full).reshape(gridsize_downscale[::-1])
	else:
		label_downscale = label_full
	return fliped_tsdf, label_downscale

num_thread = 12

train_data_lists=[]
train_data_lists.append(suncg_depthbin_dir+'SUNCGtrain_1_500')
train_data_lists.append(suncg_depthbin_dir+'SUNCGtrain_501_1000')
train_data_lists.append(suncg_depthbin_dir+'SUNCGtrain_1001_2000')
train_data_lists.append(suncg_depthbin_dir+'SUNCGtrain_1001_3000')
train_data_lists.append(suncg_depthbin_dir+'SUNCGtrain_3001_5000')
train_data_lists.append(suncg_depthbin_dir+'SUNCGtrain_5001_7000')
train_files = getDataFiles(train_data_lists)

def processTrainData(idx):
	current_data, current_label = loadDataFile(train_files[idx], idx%total_gpu_num)
	input_nz = current_data.nonzero()
	input_val = current_data[input_nz]
	current_label_legal = current_label.copy()
	illegal_idx = (current_label > 254)
	current_label_legal[illegal_idx] = 0
	target_nz = current_label_legal.nonzero()
	target_val = current_label_legal[target_nz]
	msgpack.dump({'input_nz':np.asarray(input_nz).T, 'input_val':input_val, \
		'target_nz':np.asarray(target_nz), 'target_val':target_val}, open(output_dir+'msg/train/'+str(idx)+'.msg', 'wb'), use_single_float=True)
	print(output_dir+'msg/train/'+str(idx)+'.msg')     

pool = ThreadPool(num_thread) 
results = pool.map(processTrainData, range(len(train_files)))
# you can generate only 200 data for a quick test
# results = pool.map(processTrainData, range(200))
pool.close() 
pool.join() 


# generate test data
test_data_lists = []
test_data_lists.append(suncg_depthbin_dir+'SUNCGtest_49700_49884')
test_files = getDataFiles(test_data_lists)

def processTestData(idx):
	current_data, current_label = loadDataFile(test_files[idx], idx%total_gpu_num)
	input_nz = current_data.nonzero()
	input_val = current_data[input_nz]
	current_label_legal = current_label.copy()
	illegal_idx = current_label > 254
	current_label_legal[illegal_idx] = 0
	target_nz = current_label_legal.nonzero()
	target_val = current_label_legal[target_nz]
	msgpack.dump({'input_nz':np.asarray(input_nz).T, 'input_val':input_val, \
		'target_nz':np.asarray(target_nz), 'target_val':target_val}, open(output_dir+'msg/test/'+str(idx)+'.msg', 'wb'), use_single_float=True)
	print(output_dir+'msg/test/'+str(idx)+'.msg')


pool = ThreadPool(num_thread)
pool.map(processTestData, range(len(test_files)))
pool.close() 
pool.join() 

