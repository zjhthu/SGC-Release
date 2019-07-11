#ifndef _DOWNSAMPLE_HPP_
#define _DOWNSAMPLE_HPP_
#include <vector>
#include <glog/logging.h>
#include <iostream>
#include <numeric>
// find mode  in an vector besides zeros
template <typename Dtype>
float modeLargerZero(const std::vector<Dtype>& values) {
  Dtype old_mode = 0;
  Dtype old_count = 0;
  for (size_t n = 0; n < values.size(); ++n) {
    if (values[n] > 0 && values[n] < 255) {
      Dtype mode = values[n];
      Dtype count = std::count(values.begin() + n + 1, values.end(), mode);

      if (count > old_count) {
        old_mode = mode;
        old_count = count;
      }
    }
  }
  return old_mode;
}

// find mode of in an vector
template <typename Dtype>
float mode(const std::vector<Dtype>& values) {
  Dtype old_mode = 0;
  Dtype old_count = 0;
  for (size_t n = 0; n < values.size(); ++n) {
    float mode = values[n];
    float count = std::count(values.begin() + n + 1, values.end(), mode);

    if (count > old_count) {
      old_mode = mode;
      old_count = count;
    }
  }
  return old_mode;
}

template<typename Dtype>
void downsampleLabel_cpu(std::vector<int> data_vox_size, std::vector<int> label_vox_size,
					int label_downscale, Dtype * segmentation_label_full, 
					Dtype* segmentation_label_downscale){
  Dtype emptyT = (0.95 * label_downscale * label_downscale * label_downscale);
  for (int i = 0;  i < label_vox_size[0] * label_vox_size[1] * label_vox_size[2]; ++i) {
    int z = floor(i / (label_vox_size[0] * label_vox_size[1]));
    int y = floor((i - (z * label_vox_size[0] * label_vox_size[1])) / label_vox_size[0]);
    int x = i - (z * label_vox_size[0] * label_vox_size[1]) - (y * label_vox_size[0]);

    std::vector<Dtype> field_vals;
    int zero_count = 0;
    for (int tmp_x = x * label_downscale; tmp_x < (x + 1) * label_downscale; ++tmp_x) {
      for (int tmp_y = y * label_downscale; tmp_y < (y + 1) * label_downscale; ++tmp_y) {
        for (int tmp_z = z * label_downscale; tmp_z < (z + 1) * label_downscale; ++tmp_z) {
          int tmp_vox_idx = tmp_z * data_vox_size[0] * data_vox_size[1] + tmp_y * data_vox_size[0] + tmp_x;
          //LOG(INFO)<<"tmp vox idx:"<<tmp_vox_idx;
          field_vals.push_back(segmentation_label_full[tmp_vox_idx]); // MG: TODO: -> Can't this be preallocated since the target size is always [60x36x60]?
          //LOG(INFO)<<"push back";
          if (segmentation_label_full[tmp_vox_idx] < Dtype(0.001f) || segmentation_label_full[tmp_vox_idx] > Dtype(254)) {
            zero_count++;
          }
        }
      }
    }
    if (zero_count > emptyT) {
      segmentation_label_downscale[i] = Dtype(mode(field_vals));
    } else {
      segmentation_label_downscale[i] = Dtype(modeLargerZero(field_vals)); // object label mode without zeros
    }
  }
}

template<typename Dtype>
void downsampleTSDF_cpu(std::vector<int> data_vox_size, std::vector<int> label_vox_size,
					int label_downscale, Dtype * tsdf_data_full, Dtype* tsdf_data_downscale){
  for (int i = 0;  i < label_vox_size[0] * label_vox_size[1] * label_vox_size[2]; ++i) {
    int z = floor(i / (label_vox_size[0] * label_vox_size[1]));
    int y = floor((i - (z * label_vox_size[0] * label_vox_size[1])) / label_vox_size[0]);
    int x = i - (z * label_vox_size[0] * label_vox_size[1]) - (y * label_vox_size[0]);

    std::vector<Dtype> tsdf_vals;
    int one_count = 0;
    for (int tmp_x = x * label_downscale; tmp_x < (x + 1) * label_downscale; ++tmp_x) {
      for (int tmp_y = y * label_downscale; tmp_y < (y + 1) * label_downscale; ++tmp_y) {
        for (int tmp_z = z * label_downscale; tmp_z < (z + 1) * label_downscale; ++tmp_z) {
          int tmp_vox_idx = tmp_z * data_vox_size[0] * data_vox_size[1] + tmp_y * data_vox_size[0] + tmp_x;
          tsdf_vals.push_back(Dtype(tsdf_data_full[tmp_vox_idx])); // MG: TODO: -> Can't this be preallocated since the target size is always [60x36x60]?
          if ((abs(tsdf_data_full[tmp_vox_idx]) == 1) | (abs(tsdf_data_full[tmp_vox_idx]) > 1)){
            one_count ++;
          }
        }
      }
    }

    if (one_count < 0.75 * label_downscale * label_downscale * label_downscale)
      tsdf_data_downscale[i] = std::accumulate(tsdf_vals.begin(), tsdf_vals.end(), 0.0) /tsdf_vals.size();
    else
      tsdf_data_downscale[i] = Dtype(mode(tsdf_vals));
  }
}
#endif
