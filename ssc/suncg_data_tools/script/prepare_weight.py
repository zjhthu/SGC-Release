import sys
sys.path.append('../lib')
import SUNCGData
import numpy as np
import os
import os.path
import msgpack
import msgpack_numpy as m
m.patch()

from multiprocessing import Pool as ThreadPool 
import torch

gridsize = np.asarray([240,144,240])
downscale = 4
gridsize_downscale = (gridsize / downscale).astype(int)
total_gpu_num = 1
#total_gpu_num = torch.cuda.device_count()

outdir = '../../data/weight/'
if not os.path.exists(outdir):
    print('prepare suncg  weights ...')
    os.system('mkdir -p '+outdir+'train')


def getDataFiles(data_lists):
    data_files = SUNCGData.getDataFiles(data_lists)
    data_files_valid = []
    for depth_file in data_files:
        bin_file = depth_file[:-3] + 'bin'
        if os.path.isfile(depth_file) and os.path.isfile(bin_file):
            data_files_valid.append(depth_file)
    print('total vaild files:' + str(len(data_files_valid)))
    return data_files_valid

def loadData(data_filename, downscale, gpu_num):
    bin_filename = data_filename[:-3] + 'bin'
    label_full = SUNCGData.getLabels(bin_filename).reshape(gridsize[::-1])
    if downscale>1:
        label_downscale = SUNCGData.downsampleLabel(label_full).reshape(gridsize_downscale[::-1].astype(int))
    else:
        label_downscale = label_full
    hard_pos = SUNCGData.getHardPos2(label_downscale, 1, gpu_num).reshape(gridsize_downscale[::-1].astype(int))
    complete_tsdf = SUNCGData.getCompleteTSDF(bin_filename, gpu_num).reshape(gridsize[::-1])
    if downscale>1:
        complete_tsdf_downscale = SUNCGData.downsampleTSDF(complete_tsdf).reshape(gridsize_downscale[::-1].astype(int))
    else:
        complete_tsdf_downscale = complete_tsdf
    return hard_pos, complete_tsdf_downscale

def getSegWeight(hard_pos, tsdf):
    weight = {}
    weight['hard_pos'] = (hard_pos == 1).nonzero()
    weight['easy_pos'] = (hard_pos == 0).nonzero()
    weight['hard_neg'] = np.logical_and(hard_pos==-1, np.logical_and(tsdf<0.99, tsdf>0) ).nonzero()
    weight['easy_neg'] = np.logical_and(hard_pos==-1, np.logical_not(np.logical_and(tsdf<0.99, tsdf>0)) ).nonzero()
    # hard_pos = -2 , invalid
    return weight


train_data_lists=[]
data_dir = '../../data/depthbin/'
train_data_lists.append(data_dir+'SUNCGtrain_1_500')
train_data_lists.append(data_dir+'SUNCGtrain_501_1000')
train_data_lists.append(data_dir+'SUNCGtrain_1001_2000')
train_data_lists.append(data_dir+'SUNCGtrain_1001_3000')
train_data_lists.append(data_dir+'SUNCGtrain_3001_5000')
train_data_lists.append(data_dir+'SUNCGtrain_5001_7000')
train_files = getDataFiles(train_data_lists)



train_weight_dir = outdir+'train/'
def processTrainWeight(idx):
    if os.path.exists(train_weight_dir+str(idx)+'_weight.msg'):
        print(train_weight_dir+str(idx)+'_weight.msg exists')
        return
    hard_pos, tsdf = loadData(train_files[idx], downscale, idx%total_gpu_num)
    weight = getSegWeight(hard_pos, tsdf)
    hard_pos_num = np.asarray(weight['hard_pos']).shape[1]
    easy_pos_num = np.asarray(weight['easy_pos']).shape[1]
    hard_neg_num = np.asarray(weight['hard_neg']).shape[1]
    easy_neg_num = np.asarray(weight['easy_neg']).shape[1]
    easy_pos_select = np.random.permutation(easy_pos_num)[:min(hard_pos_num, easy_pos_num)]
    hard_neg_select = np.random.permutation(hard_neg_num)[:min(int(hard_pos_num*5.5), hard_neg_num)]
    easy_neg_select = np.random.permutation(easy_neg_num)[:min(int(hard_pos_num*2), easy_neg_num)]
    print('hard pos num '+str(hard_pos_num))
    print('easy neg num '+str(easy_neg_num))
    msgpack.dump({'hard_pos':np.asarray(weight['hard_pos']).astype(np.uint8), \
                  'easy_pos':np.asarray(weight['easy_pos']).astype(np.uint8)[:,easy_pos_select], \
                  'hard_neg':np.asarray(weight['hard_neg']).astype(np.uint8)[:,hard_neg_select], \
                  'easy_neg':np.asarray(weight['easy_neg']).astype(np.uint8)[:,easy_neg_select], \
                  }, open(train_weight_dir+str(idx)+'_weight.msg', 'wb'))
    print(train_weight_dir+str(idx)+'_weight.msg')

num_thread = 20
train_data_num = len(train_files)
pool = ThreadPool(num_thread) 
results = pool.map(processTrainWeight, range(train_data_num))
# you can generate only 200 data for a quick test
#results = pool.map(processTrainWeight, range(200))
pool.close() 
pool.join() 




