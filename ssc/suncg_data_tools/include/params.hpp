#ifndef CAFFE_SUNCG_PARAMS_HPP
#define CAFFE_SUNCG_PARAMS_HPP
#include <vector>
const int frame_height = 480;
const int frame_width = 640;
const float vox_unit = 0.02;
const float vox_margin = 0.24;
int vox_size[] = {240, 144, 240};
// int crop_size[] = {240, 144, 240};
// vector<int> data_crop_vox_size(crop_size, crop_size + 3);
const std::vector<int> data_full_vox_size(vox_size, vox_size + 3);

// std::vector<int> data_full_vox_size;
// std::vector<int> data_crop_vox_size;


int label_size[] = {60, 36, 60};
const std::vector<int> label_vox_size(label_size, label_size + 3);
const unsigned int seg_classes = 11;
bool shuffle = true;
bool occ_empty_only = true;
float neg_obj_sample_ratio = 2;
int seg_class_map[] = {0
, 1
, 2
, 3
, 4
, 11
, 5
, 6
, 7
, 8
, 8
, 10
, 10
, 10
, 11
, 11
, 9
, 8
, 11
, 11
, 11
, 11
, 11
, 11
, 11
, 11
, 11
, 10
, 10
, 11
, 8
, 10
, 11
, 9
, 11
, 11
, 11};
std::vector<int> segmentation_class_map(seg_class_map, seg_class_map + 37);
float seg_class_weight[] = { 1
, 1
, 1
, 1
, 1
, 1
, 1
, 1
, 1
, 1
, 1
, 1
, 1
, 1
, 1
, 1
, 1};
const std::vector<float> segmentation_class_weight(seg_class_weight,seg_class_weight + 17);
float occ_class_weight[] = { 10,10};
std::vector<float> occupancy_class_weight(occ_class_weight, occ_class_weight + 2);
float cam_K[9] = {518.8579f, 0.0f, (float)frame_width / 2.0f, 0.0f, 518.8579f, (float)frame_height / 2.0f, 0.0f, 0.0f, 1.0f};

bool with_projection_tsdf = false;
int batch_size = 1;
unsigned int tsdf_type = 1;
//     data_type: TSDF
bool surf_only = false;

#endif //CAFFE_SUNCG_PARAMS_HPP 
