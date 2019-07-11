import torch
from torch.autograd import Variable
from torch.nn import Module
import sparseconvnet as scn
import numpy as np
import time

def GetConvMask(x, filter_size):
#    mask_time = time.time()
    mask = x.getConvMask(filter_size)
#    print('get conv mask time:'+str(time.time()-mask_time))
    return Variable(mask)
    #check 
'''
    mask_check = torch.FloatTensor(x.features.size(0)).fill_(0)
    dense_x = scn.SparseToDenseFunction().apply(x.features,x.metadata,x.spatial_size,3,x.features.size(1))
    locations_x = x.getSpatialLocations()
    offset = []
    for off_x in [-1,0,1]:
        for off_y in [-1,0,1]:
            for off_z in [-1,0,1]:
                offset.append([off_x, off_y, off_z])

    for location_idx in range(locations_x.size(0)):
        batch = locations_x[location_idx,3]
        print('location idx:'+str(location_idx))
        for off in offset:
            locations_offset = (torch.LongTensor(locations_x[location_idx,:3]) + torch.LongTensor(off))
            if ((locations_offset>=0).sum() + (locations_offset<64).sum()) == 6:
                locations_offset = locations_offset.tolist()
                if dense_x[batch,:,locations_offset[0],locations_offset[1],locations_offset[2]].abs().sum().data.cpu().tolist()[0] != 0:
                    #print(dense_x[batch,:,locations_offset[0],locations_offset[1],locations_offset[2]])
                    #print(dense_x[batch,:,locations_offset[0],locations_offset[1],locations_offset[2]].abs().sum().data.cpu().tolist())
                    mask_check[location_idx] = mask_check[location_idx] + 1
        assert mask_check[location_idx] == mask[location_idx], 'mask_check['+str(location_idx)+']:'+str(mask_check[location_idx])+' should equal to mask['+str(location_idx)+']:'+str(mask[location_idx])
    print('mask check pass!!!!!!!!!!')
'''
#    return Variable(mask)

