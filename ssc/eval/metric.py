import torch
import numpy as np
import scipy.io as sio
import h5py
import os

obj_class = ['empty','ceiling','floor','wall','window','chair','bed','sofa','table','tvs','furn','objs']
numoffiles = 470
# dataRootfolder = '/home/ayao/zjh/sscnet/data/'
dataRootfolder = os.path.dirname(__file__)+'/../data/'
groundtruth_path = dataRootfolder + 'eval/SUNCGtest_49700_49884/'
evalvol_path = groundtruth_path

mat_data = sio.loadmat(os.path.dirname(__file__)+'/suncg_eval.mat')
mapIds = mat_data['mapIds'][0]
Filename = mat_data['Filename'][0]
classtoEvaluate = np.arange(1,12)

# reference suncg evaluation_script.m
# In all stages, data is physically saved in c++ order, (x,y,z), x frist
# when loaded in pytorch, shape is (z,y,x), output shape is (n, c, z, y, x)
# when loaded in caffe, output shape is (n, c, z, y, x)
# when loaded or presaved in matlab, shape is (x, y, z)
# python caffe date loaded to matlab using hdf5: (n, c, z, y, x) -> (x, y, z, c, n)
# But!!! when matlab data is loaded to python using sio.loadmat, data physically storage is changed,
# matlab (x, y, z) ----changed-----> to python (x, y, z)
# when loading matlab data to python using h5py, it is not changed
# matlab (x, y, z) ----------------> to python (z, y, x)

def evaluate_prediction(GTV, V, voxels_to_evaluate, classtoEvaluate):
	assert np.all(GTV.shape == V.shape), 'GTV shape'+str(GTV.shape)+'not equal to V shape'+str(V.shape)
	results = {}
	gt = GTV[voxels_to_evaluate]
	prediction = V[voxels_to_evaluate]

	union = np.sum(np.logical_or(gt>0, prediction>0))
	intersection = np.sum(np.logical_and(gt>0, prediction>0))
	results['iou'] = float(intersection) / union

	results['tp_occ'] = np.sum(np.logical_and(gt>0, prediction>0))
	results['fp_occ'] = np.sum(np.logical_and(gt==0, prediction >0))
	results['fn_occ'] = np.sum(np.logical_and(gt>0, prediction==0))
	results['tp'] = np.zeros(classtoEvaluate.size)
	results['fp'] = np.zeros(classtoEvaluate.size)
	results['fn'] = np.zeros(classtoEvaluate.size)

	for idx, label in enumerate(classtoEvaluate):
		results['tp'][idx] = np.sum(np.logical_and(gt == label, prediction == label))
		results['fp'][idx] = np.sum(np.logical_and(gt != label, prediction == label))
		results['fn'][idx] = np.sum(np.logical_and(gt == label, prediction != label))
	return results
def evaluate_completion(GTV, V, empty_conf_V, vol, conf_threshold):
	results = {}
	voxels_to_evaluate = np.logical_and(vol<0, vol>=-1)
	gt = GTV[voxels_to_evaluate]
	prediction = V[voxels_to_evaluate]

	union = np.sum(np.logical_or(gt>0, prediction>0))
	intersection = np.sum(np.logical_and(gt>0, prediction>0))
	results['iou'] = float(intersection) / union
	results['tp_occ'] = np.sum(np.logical_and(gt>0, prediction>0))
	results['fp_occ'] = np.sum(np.logical_and(gt==0, prediction >0))
	results['fn_occ'] = np.sum(np.logical_and(gt>0, prediction==0))
	results['precision'] = results['tp_occ'] / (results['tp_occ']+results['fp_occ']+np.finfo(float).eps)
	results['recall'] = results['tp_occ'] / (results['tp_occ']+results['fn_occ']+np.finfo(float).eps)
	return results
	

# predobjTensor: (n, c, z, y, x)
def eval_suncg(predobjTensor):
	resultsFullSeg = []
	resultsFullOcc = []
	numoffiles = predobjTensor.shape[0]
	print(numoffiles)
	for batchId in range(numoffiles):
		ld = h5py.File(groundtruth_path+Filename[batchId][0]+'_gt_d4.mat')
		sceneVox = np.asarray(ld['sceneVox_ds'])
		ld = h5py.File(evalvol_path+Filename[batchId][0]+'_vol_d4.mat')
		vol = np.asarray(ld['flipVol_ds'])
		sceneVox[np.logical_or(sceneVox==255, np.isnan(sceneVox))] = 0
		labelobj = mapIds[sceneVox.astype(int)]
		# get prediction
		predobj_conf = np.squeeze(predobjTensor[batchId,:])
		# test_mat = np.asarray([[1,2,3],[4,5,6]])
		# sio.savemat('check.mat',{'sceneVox_py':sceneVox, 'predobj_conf_py':predobj_conf, 'test_mat':test_mat})		
		predobj = np.argmax(predobj_conf, axis=0)

		nonfree_voxels_to_evaluate = np.logical_or(np.abs(vol)<1, vol==-1)
		resultsFullSeg.append(evaluate_prediction(labelobj, predobj, nonfree_voxels_to_evaluate, classtoEvaluate))
		resultsFullOcc.append(evaluate_completion(labelobj, predobj, predobj_conf[0,:,:,:], vol, []))

	tps = np.ones(classtoEvaluate.size)*np.finfo(float).eps
	fps = np.ones(classtoEvaluate.size)*np.finfo(float).eps
	fns = np.ones(classtoEvaluate.size)*np.finfo(float).eps
	for results in resultsFullSeg:
		tps += results['tp']
		fps += results['fp']
		fns += results['fn']
	full_precision = tps / (tps + fps)
	full_recall = tps / (tps + fns)
	full_iou = tps / (tps + fns + fps)
	print('Semantic Scene Compeltion:\nprec ,recall , IoU\nmean:'+str(np.mean(full_precision))+', '+str(np.mean(full_recall))+', ' +str(np.mean(full_iou)))
	for idx in range(classtoEvaluate.size):
		print(obj_class[idx+1]+ ' '+ str(full_precision[idx]) + ' ' + str(full_recall[idx]) + ' ' + str(full_iou[idx])) 
	# completion
	occ_precisions = [results['precision'] for results in resultsFullOcc]
	occ_recalls = [results['recall'] for results in resultsFullOcc]
	occ_ious = [results['iou'] for results in resultsFullOcc]

	print('Scene Completion:\nprec, recall, IoU\n'+str(np.mean(occ_precisions))+', '+str(np.mean(occ_recalls))+', '+str(np.mean(occ_ious)))
	return np.mean(full_iou), np.mean(occ_ious)


# predLabelTensor: (n, z, y, x)
def eval_suncg2(predLabelTensor):
	resultsFullSeg = []
	resultsFullOcc = []
	numoffiles = predLabelTensor.shape[0]
	print(numoffiles)
	for batchId in range(numoffiles):
		ld = h5py.File(groundtruth_path+Filename[batchId][0]+'_gt_d4.mat')
		sceneVox = np.asarray(ld['sceneVox_ds'])
		ld = h5py.File(evalvol_path+Filename[batchId][0]+'_vol_d4.mat')
		vol = np.asarray(ld['flipVol_ds'])
		sceneVox[np.logical_or(sceneVox==255, np.isnan(sceneVox))] = 0
		labelobj = mapIds[sceneVox.astype(int)]
		# get prediction
		predobj = np.squeeze(predLabelTensor[batchId,:])

		nonfree_voxels_to_evaluate = np.logical_or(np.abs(vol)<1, vol==-1)
		resultsFullSeg.append(evaluate_prediction(labelobj, predobj, nonfree_voxels_to_evaluate, classtoEvaluate))
		resultsFullOcc.append(evaluate_completion(labelobj, predobj, [], vol, []))

	tps = np.ones(classtoEvaluate.size)*np.finfo(float).eps
	fps = np.ones(classtoEvaluate.size)*np.finfo(float).eps
	fns = np.ones(classtoEvaluate.size)*np.finfo(float).eps
	for results in resultsFullSeg:
		tps += results['tp']
		fps += results['fp']
		fns += results['fn']
	full_precision = tps / (tps + fps)
	full_recall = tps / (tps + fns)
	full_iou = tps / (tps + fns + fps)
	print('Semantic Scene Compeltion:\nprec ,recall , IoU\nmean:'+str(np.mean(full_precision))+', '+str(np.mean(full_recall))+', ' +str(np.mean(full_iou)))
	for idx in range(classtoEvaluate.size):
		print(obj_class[idx+1]+ ' '+ str(full_precision[idx]) + ' ' + str(full_recall[idx]) + ' ' + str(full_iou[idx])) 
	# completion
	occ_precisions = [results['precision'] for results in resultsFullOcc]
	occ_recalls = [results['recall'] for results in resultsFullOcc]
	occ_ious = [results['iou'] for results in resultsFullOcc]

	print('Scene Completion:\nprec, recall, IoU\n'+str(np.mean(occ_precisions))+', '+str(np.mean(occ_recalls))+', '+str(np.mean(occ_ious)))
