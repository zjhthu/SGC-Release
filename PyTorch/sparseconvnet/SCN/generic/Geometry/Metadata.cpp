// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Geometry/Metadata.cpp"
#else

#include "Metadata.h"
#include <cstring>
#include <iostream>
//#include "omp.h"
#include <time.h>
#include <sys/time.h>
#include "ValidConvolutionRules.h"

extern "C" void scn_D_(setInputSpatialSize)(void **m,
                                            THLongTensor *spatialSize) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  _m.setInputSpatialSize(spatialSize);
}

extern "C" void scn_D_(batchAddSample)(void **m) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  assert(_m.inputSGs && "Call setInputSpatialSize first, please!");
  _m.inputSGs->resize(_m.inputSGs->size() + 1);
  _m.inputSG = &_m.inputSGs->back();
}

void scn_D_(addPointToSparseGridMapAndFeatures)(SparseGridMap<Dimension> &mp,
                                                Point<Dimension> p,
                                                uInt &nActive, long nPlanes,
                                                THFloatTensor *features,
                                                float *vec, bool overwrite) {
  auto iter = mp.find(p);
  if (iter == mp.end()) {
    iter = mp.insert(std::make_pair(p, nActive++)).first;
    THFloatTensor_resize2d(features, nActive, nPlanes);
    std::memcpy(THFloatTensor_data(features) + (nActive - 1) * nPlanes, vec,
                sizeof(float) * nPlanes);
  } else if (overwrite) {
    std::memcpy(THFloatTensor_data(features) + iter->second * nPlanes, vec,
                sizeof(float) * nPlanes);
  }
}
extern "C" void scn_D_(setInputSpatialLocation)(void **m,
                                                THFloatTensor *features,
                                                THLongTensor *location,
                                                THFloatTensor *vec,
                                                bool overwrite) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto p = LongTensorToPoint<Dimension>(location);
  auto &mp = _m.inputSG->mp;
  auto &nActive = *_m.inputNActive;
  auto nPlanes = vec->size[0];
  scn_D_(addPointToSparseGridMapAndFeatures)(
      mp, p, nActive, nPlanes, features, THFloatTensor_data(vec), overwrite);
}
extern "C" void scn_D_(setInputSpatialLocations)(void **m,
                                                 THFloatTensor *features,
                                                 THLongTensor *locations,
                                                 THFloatTensor *vecs,
                                                 bool overwrite) {
  assert(locations->size[0] == vecs->size[0] and
         "Location.size(0) and vecs.size(0) must be equal!");
  assert((locations->size[1] == Dimension or
          locations->size[1] == 1 + Dimension) and
         "locations.size(0) must be either Dimension or Dimension+1");
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  Point<Dimension> p;
  auto &nActive = *_m.inputNActive;
  auto nPlanes = vecs->size[1];
  auto l = THLongTensor_data(locations);
  auto v = THFloatTensor_data(vecs);

  if (locations->size[1] == Dimension) {
    // add points to current sample
    assert(_m.inputSG);
    auto &mp = _m.inputSG->mp;
    for (uInt idx = 0; idx < locations->size[0]; ++idx) {
      for (int d = 0; d < Dimension; ++d)
        p[d] = *l++;
      scn_D_(addPointToSparseGridMapAndFeatures)(mp, p, nActive, nPlanes,
                                                 features, v, overwrite);
      v += nPlanes;
    }
  }
  if (locations->size[1] == Dimension + 1) {
    // add new samples to batch as necessary
    auto &SGs = *_m.inputSGs;
    for (uInt idx = 0; idx < locations->size[0]; ++idx) {
      for (int d = 0; d < Dimension; ++d)
        p[d] = *l++;
      auto batch = *l++;
      if (batch >= SGs.size()) {
        SGs.resize(batch + 1);
      }
      auto &mp = SGs[batch].mp;
      scn_D_(addPointToSparseGridMapAndFeatures)(mp, p, nActive, nPlanes,
                                                 features, v, overwrite);
      v += nPlanes;
    }
  }
}

// extern "C" void scn_D_(setInputSpatialLocations2)(void **m,
//                                                  THLongTensor *locations_) {
//   assert((locations_->size[1] == 1 + Dimension) and
//          "locations.size(0) must be either Dimension or Dimension+1");
//   SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)

//   auto locations = THLongTensor_data(locations_);
//   // std::vector<long> br;
//   // br.push_back(0);
//   // auto &nActive = *_m.inputNActive;
//   // uInt batchSize = 0;
//   // for (long i = 0; i < locations_->size[0]; ++i){
//   //   if(locations[i*(Dimension+1)+Dimension] != batchSize){
//   //     br.push_back(nActive);
//   //     batchSize++;
//   //   }
//   //   else{
//   //     nActive++;
//   //   }
//   // }
//   // batchSize++;
//   // _m.inputSGs->resize(batchSize);

//   uInt b;
// #pragma omp parallel for private(b)
//   for (b = 0; b < batchSize; b++) {
//     auto &sg = _m.inputSGs->at(b);
//     for (uInt i = br[b]; i < br[b + 1]; i++) {
//       Point<Dimension> x;
//       for (uInt j = 0; j < Dimension; j++) {
//         x[j] = locations[i * Dimension + j];
//       }
//       sg.mp[x] = i;
//     }
//   }
// }
extern "C" void scn_D_(getSpatialLocations)(void **m, THLongTensor *spatialSize,
                                            THLongTensor *locations) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  uInt nActive = _m.getNActive(spatialSize);
  auto &SGs = _m.getSparseGrid(spatialSize);
  uInt batchSize = SGs.size();
  THLongTensor_resize2d(locations, nActive, Dimension + 1);
  THLongTensor_zero(locations);

  auto lD = THLongTensor_data(locations);
  uInt i;
#pragma omp parallel for private(i)
  for (i = 0; i < batchSize; i++) {
    auto &mp = SGs[i].mp;
    for (auto it = mp.begin(); it != mp.end(); ++it) {
      for (uInt d = 0; d < Dimension; ++d) {
        lD[(it->second + SGs[i].ctr) * (Dimension + 1) + d] = it->first[d];
      }
      lD[(it->second + SGs[i].ctr)* (Dimension + 1) + Dimension] = i;
    }
  }
}

extern "C" void scn_D_(getSampleWeight)(void **m, THLongTensor *spatialSize, THFloatTensor *sample_,
                                        THLongTensor *position_, THFloatTensor *val_) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  uInt nActive = _m.getNActive(spatialSize);
  auto &SGs = _m.getSparseGrid(spatialSize);
  uInt batchSize = SGs.size();

  auto sample = THFloatTensor_data(sample_);// 1 sample, 0 reserved
  std::vector<std::vector<long>> rules(nActive, std::vector<long>(26));
  std::vector<int> count(nActive);
  std::vector<long> total_count(batchSize);
  //std::vector<long> kernel_size(Dimension);
  long kernel_size[Dimension];
  for (int d = 0; d < Dimension; d++)
    kernel_size[d]=3;

  uInt i;
#pragma omp parallel for private(i)
  for (i = 0; i < batchSize; i++) {
    auto &mp = SGs[i].mp;
    total_count[i] = 0;
    for (auto it = mp.begin(); it != mp.end(); ++it) {
      long it_idx = it->second+SGs[i].ctr;
      int sample_neighbors = 0;
      //std::cout << "it_idx:"<<it_idx <<",nActive:"<<nActive;
      /*if (it_idx >= nActive){
          std::cout << "false" << std::endl;
          exit(0);
      }*/
      //std::cout << ",sample[it_idx]:"<<sample[it_idx]<<std::endl;
      if(sample[it_idx] == 0){   
        auto inRegion = InputRegionCalculator_Valid<Dimension>(it->first, kernel_size);
        for (auto inputPoint : inRegion) {
          auto inputIter = mp.find(inputPoint);
          if (inputIter != mp.end()){
            long inputIter_idx = inputIter->second + SGs[i].ctr;
            if (sample[inputIter_idx] == 1) {
              rules[it_idx][sample_neighbors] = inputIter_idx;
              sample_neighbors++;
            } 
          }
        }
        //std::cout<<"sampple neighbors:"<<sample_neighbors<<std::endl;
        total_count[i] += sample_neighbors;
        if (sample_neighbors == 0){
          sample[it_idx] = 1;// no neighbers -> sample
        }
      }
      //std::cout << "count it idx:"<<count[it_idx]<<std::endl;
      count[it_idx] = sample_neighbors;
      //count[it_idx] = 0;
    }// end of it of mp
  }// end of batch
  //exit(0);
  //std::cout << "before total"<<std::endl; 
  long total = 0;
  for (int b = 0; b < batchSize; b++){
    total+=total_count[b];
  }
  //std::cout << "total:" << total <<std::endl;
  THLongTensor_resize2d(position_, total, 2);
  THLongTensor_zero(position_);
  auto position = THLongTensor_data(position_);
  THFloatTensor_resize1d(val_, total);
  THFloatTensor_zero(val_);
  auto val = THFloatTensor_data(val_);
  long reserved_indx = 0;
  //std::cout << "nActive"<<std::endl;
  for (long i = 0; i < nActive; ++i){
    if(count[i] > 0){
      //std::cout << "count i:" << i << std::endl;
      for(int count_idx = 0; count_idx < count[i]; count_idx++){
        position[reserved_indx*2] = i;
        position[reserved_indx*2+1] = rules[i][count_idx];
        val[reserved_indx] = 1.0 / count[i];
        reserved_indx++;
      }
    }
  }
}

extern "C" void scn_D_(getLocationsIndexInRef)(void **m, void **m_ref, THLongTensor *spatialSize,
                                            THLongTensor *indexs_) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m_ref)
  auto &SGs = _m.getSparseGrid(spatialSize);
  auto &SGs_ref = _m_ref.getSparseGrid(spatialSize);
  uInt batchSize = SGs.size();

  auto indexs = THLongTensor_data(indexs_);

  uInt i;
#pragma omp parallel for private(i)
  for (i = 0; i < batchSize; i++) {
    auto &mp = SGs[i].mp;
    auto &mp_ref = SGs_ref[i].mp;
    for (auto it = mp.begin(); it != mp.end(); ++it) {
      auto hit = mp_ref.find(it->first);
      // if(hit == mp_ref.end()){
      //   // indexs[it->second + SGs[i].ctr] = -1;
      //   // break;
      // }
      // else{
      if (hit != mp_ref.end()){
        indexs[it->second + SGs[i].ctr] = hit->second + SGs_ref[i].ctr;
      }
    }
  }
}
extern "C" long scn_D_(concatTensor)(void **m, void **in1, void **in2, THLongTensor *spatialSize,
                                     THLongTensor *idx1_, THLongTensor *idx2_) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, in1)
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, in2)
  auto &SGs_1 = _in1.getSparseGrid(spatialSize);
  auto &SGs_2 = _in2.getSparseGrid(spatialSize);
  uInt batchSize = SGs_1.size();
  _m.inputSGs->resize(batchSize);
  auto &SGs = _m.getSparseGrid(spatialSize);

  auto idx1 = THLongTensor_data(idx1_);
  auto idx2 = THLongTensor_data(idx2_);
  uInt i;
#pragma omp parallel for private(i)
  for (i = 0; i < batchSize; i++) {
    auto &mp = SGs[i].mp;
    auto &mp1 = SGs_1[i].mp;
    auto &mp2 = SGs_2[i].mp;
    SGs[i].ctr = 0;
    for (auto it = mp1.begin(); it != mp1.end(); ++it){
      mp[it->first] = SGs[i].ctr;
      idx1[it->second + SGs_1[i].ctr] = SGs[i].ctr;
      SGs[i].ctr++;
    }
    for (auto it = mp2.begin(); it != mp2.end(); ++it){
      auto hit = mp.find(it->first);
      if (hit == mp.end()){
        mp[it->first] = SGs[i].ctr;
        idx2[it->second + SGs_2[i].ctr] = SGs[i].ctr;
        SGs[i].ctr++;
      }
      else{
        idx2[it->second + SGs_2[i].ctr] = hit->second;
      }
    }
  }
  auto &nActive = *_m.inputNActive;
  for (uInt i = 0; i < batchSize; i++) {
    uInt tmp = nActive;
    nActive += SGs[i].ctr;
    SGs[i].ctr = tmp;
  }

#pragma omp parallel for private(i)
  for (i = 0; i < batchSize; i++) {
    auto mp1 = SGs_1[i].mp;
    auto mp2 = SGs_2[i].mp;
    for (auto it = mp1.begin(); it != mp1.end(); ++it){
      idx1[it->second + SGs_1[i].ctr] += SGs[i].ctr;
    }
    for (auto it = mp2.begin(); it != mp2.end(); ++it){
      idx2[it->second + SGs_2[i].ctr] += SGs[i].ctr;
    }
  }
  return long(nActive);
}
extern "C" void scn_D_(extractStructure)(void **m, THLongTensor *spatialSize, 
                                         THLongTensor *label_, THLongTensor *subset_, THLongTensor *kernel_size_) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto &SGs = _m.getSparseGrid(spatialSize);
  uInt batchSize = SGs.size();

  auto label = THLongTensor_data(label_);
  auto subset = THLongTensor_data(subset_);
  auto kernel_size = THLongTensor_data(kernel_size_);
  uInt i;
#pragma omp parallel for private(i)
  for (i = 0; i < batchSize; i++) {
    auto &mp = SGs[i].mp;
    for (auto it = mp.begin(); it != mp.end(); ++it) {
      if (label[it->second + SGs[i].ctr] > 0){
        auto inRegion = InputRegionCalculator_Valid<Dimension>(it->first, kernel_size);
        for (auto inputPoint : inRegion) {
          auto inputIter = mp.find(inputPoint);
          if (inputIter != mp.end()) {
            subset[inputIter->second + SGs[i].ctr] = 1;
          }
        }
      }
    }
  }
}

extern "C" void scn_D_(extractStructure2)(void **m, THLongTensor *spatialSize, THFloatTensor *p_, 
                                         THLongTensor *label_, THLongTensor *subset_, THLongTensor *kernel_size_, float p_threshold) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto &SGs = _m.getSparseGrid(spatialSize);
  uInt batchSize = SGs.size();

  auto p = THFloatTensor_data(p_);
  auto label = THLongTensor_data(label_);
  auto subset = THLongTensor_data(subset_);
  auto kernel_size = THLongTensor_data(kernel_size_);
  uInt i;
#pragma omp parallel for private(i)
  for (i = 0; i < batchSize; i++) {
    auto &mp = SGs[i].mp;
    for (auto it = mp.begin(); it != mp.end(); ++it) {
      long it_idx = it->second + SGs[i].ctr;
      long it_label = label[it_idx];

      if (it_label > 0){
        auto inRegion = InputRegionCalculator_Valid<Dimension>(it->first, kernel_size);
        for (auto inputPoint : inRegion) {
          auto inputIter = mp.find(inputPoint);
          if (inputIter != mp.end()) {
            long inputIter_idx = inputIter->second + SGs[i].ctr;
            if (label[inputIter_idx] == 0){
              subset[inputIter_idx] = 1;
              subset[it_idx] = 1;
            }
            else if(label[inputIter_idx] != it_label){
              subset[it_idx] = 1;
            }
          }
        }//end of search neighber
        if (p[it_idx] < p_threshold){
          subset[it_idx] = 1;
        }//end of 
      }
    }//end of mp
  }//end of batch
}

extern "C" void
scn_D_(createMetadataForDenseToSparse)(void **m, THLongTensor *spatialSize_,
                                       THLongTensor *nz_, long batchSize) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  _m.clear();
  _m.setInputSpatialSize(spatialSize_);
  _m.inputSGs->resize(batchSize);
  auto &nActive = *_m.inputNActive;
  nActive = nz_->size[0];

  auto nz = THLongTensor_data(nz_);
  auto spatialSize = THLongTensor_data(spatialSize_);

  std::vector<uInt> br(batchSize + 1);
  if (batchSize == 1) {
    br[1] = nActive;
  } else {
    long b = 0;
    for (uInt i = 0; i < nActive; i++) {
      long B = nz[i * (Dimension + 1)];
      for (; b < B;)
        br[++b] = i;
    }
    for (; b < batchSize;)
      br[++b] = nActive;
  }
  uInt b;
#pragma omp parallel for private(b)
  for (b = 0; b < batchSize; b++) {
    auto &sg = _m.inputSGs->at(b);
    for (uInt i = br[b]; i < br[b + 1]; i++) {
      Point<Dimension> x;
      for (uInt j = 0; j < Dimension; j++) {
        x[j] = nz[i * (Dimension + 1) + j + 1]; // 0-indexed
      }
      sg.mp[x] = i;
    }
  }
}

// only for set locations of raw input data
extern "C" void
scn_D_(setInputBatchSpatialLocations)(void **m, 
                                      THFloatTensor *features_,
                                      THLongTensor *locations_, 
                                      THFloatTensor *vecs_,
                                      THLongTensor *nz_nums_) {
  assert(locations_->size[0] == vecs_->size[0] and
         "Location.size(0) and vecs.size(0) must be equal!");
  assert(locations_->size[1] == Dimension  and
         "locations.size(1) must be Dimension");
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto batchSize = nz_nums_->size[0];
  _m.inputSGs->resize(batchSize);
  auto &nActive = *_m.inputNActive;
  nActive = locations_->size[0];
  
  auto locations = THLongTensor_data(locations_);
  auto vecs = THFloatTensor_data(vecs_);
  auto nz_nums = THLongTensor_data(nz_nums_);

  std::vector<uInt> br(batchSize + 1);
  br[0] = 0;
  for(long i = 0; i < batchSize; i++){
    br[i+1] = br[i] + nz_nums[i];
  }

  uInt b;
#pragma omp parallel for private(b)
  for (b = 0; b < batchSize; b++) {
    auto &sg = _m.inputSGs->at(b);
    for (uInt i = br[b]; i < br[b + 1]; i++) {
      Point<Dimension> x;
      for (uInt j = 0; j < Dimension; j++) {
        x[j] = locations[i * Dimension + j];
      }
      sg.mp[x] = i;
    }
  }

  auto nPlanes = vecs_->size[1];
  THFloatTensor_resize2d(features_, nActive, nPlanes);
  std::memcpy(THFloatTensor_data(features_), vecs, sizeof(float) * nPlanes * nActive);
}


// tensor is size[0] x .. x size[Dimension-1] x size[Dimension]
// size[0] x .. x size[Dimension-1] == spatial volume
// size[Dimension] == #feature planes
extern "C" void scn_D_(addSampleFromThresholdedTensor)(
    void **m, THFloatTensor *features_, THFloatTensor *tensor_,
    THLongTensor *offset_, THLongTensor *spatialSize_, float threshold) {

  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  auto &nActive = *_m.inputNActive;
  auto &SGs = *_m.inputSGs;
  SGs.resize(SGs.size() + 1);
  auto &sg = SGs.back();

  auto tensor = THFloatTensor_data(tensor_);
  auto offset = THLongTensor_data(offset_);
  auto spatialSize = THLongTensor_data(spatialSize_);
  long *size = tensor_->size;
  auto nPlanes = size[Dimension];
  long volume = 1;
  for (int i = 0; i < Dimension; ++i)
    volume *= size[i];
  THFloatTensor_resize2d(features_, nActive + volume, nPlanes);
  // Increment pointers as we work through the data
  auto features = THFloatTensor_data(features_) + nActive * nPlanes;

  // Active locations
  Point<Dimension> point;
  for (uInt i = 0; i < Dimension; i++)
    point[i] = offset[i];
  for (uInt ctr = 0; ctr < volume; ctr++) {
    bool active = false;
    for (uInt i = 0; i < nPlanes; i++) {
      if (fabs(tensor[i]) > threshold) {
        active = true;
        break;
      }
    }
    for (uInt i = 0; i < Dimension; i++) {
      if (point[i] < 0 or point[i] >= spatialSize[i]) {
        active = false;
        break;
      }
    }
    if (active) {
      sg.mp[point] = nActive++;
      std::memcpy(features, tensor, sizeof(float) * nPlanes);
      features += nPlanes;
    }
    tensor += nPlanes;
    incrementPointInCube<Dimension>(point, size, offset);
  }
  THFloatTensor_resize2d(features_, nActive, nPlanes);
}

// 3x3 valid convolutions, 3x3/2x2 pooling or strided convolutions
extern "C" void scn_D_(generateRuleBooks3s2)(void **m) {
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  long sz[Dimension], str[Dimension], inS[Dimension], outS[Dimension];
  Point<Dimension> p1;
  Point<2 * Dimension> p2;
  Point<3 * Dimension> p3;
  for (int i = 0; i < Dimension; ++i) {
    p1[i] = p2[i] = p3[i] = inS[i] = _m.inputSpatialSize[i];
    p2[i + Dimension] = p3[i + Dimension] = sz[i] = 3;
    p3[i + 2 * Dimension] = str[i] = 2;
  }
  while (true) {
    auto &SGs = _m.grids[p1];
    auto &rb = _m.validRuleBooks[p2];
    if (rb.empty())
      ValidConvolution_SgsToRules(SGs, rb, sz);
    for (int i = 0; i < Dimension; ++i)
      if (p1[i] < 3 or p1[i] % 2 != 1)
        return;
      else
        p1[i] = outS[i] = (inS[i] - 1) / 2;
    auto &SGs2 = _m.grids[p1];
    auto &rb2 = _m.ruleBooks[p3];
    if (rb2.empty())
      _m.nActive[p1] = Convolution_InputSgsToRulesAndOutputSgs_OMP(
          SGs, SGs2, rb2, sz, str, inS, outS);
    for (int i = 0; i < Dimension; ++i)
      p2[i] = p3[i] = inS[i] = outS[i];
  }
}

// 3x3 valid convolutions, 2x2 pooling or strided convolutions
extern "C" void scn_D_(generateRuleBooks2s2)(void **m) {

  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  long s2[Dimension], s3[Dimension], inS[Dimension], outS[Dimension];
  Point<Dimension> p1;
  Point<2 * Dimension> p2;
  Point<3 * Dimension> p3;
  for (int i = 0; i < Dimension; ++i) {
    p1[i] = p2[i] = p3[i] = inS[i] = _m.inputSpatialSize[i];
    p2[i + Dimension] = s3[i] = 3;
    p3[i + Dimension] = p3[i + 2 * Dimension] = s2[i] = 2;
  }
  while (true) {    
    auto &SGs = _m.grids[p1];
    auto &rb = _m.validRuleBooks[p2];
    // struct timeval valid_start, valid_end;
    // gettimeofday(&valid_start, NULL);
    ValidConvolution_SgsToRules_OMP(SGs, rb, s3);
    // gettimeofday(&valid_end, NULL);
    // double vaild_run_time = ((valid_end.tv_sec  - valid_start.tv_sec) * 1000000u + valid_end.tv_usec - valid_start.tv_usec) / 1.e6;
    // std::cout << "valid convolution spend time:" << vaild_run_time << std::endl;
    
    for (int i = 0; i < Dimension; ++i)
      if (p1[i] < 2 or p1[i] % 2 != 0)
        return;
      else
        p1[i] = outS[i] = inS[i] / 2;
    auto &SGs2 = _m.grids[p1];
    auto &rb2 = _m.ruleBooks[p3];
    if (rb2.empty()){
      _m.nActive[p1] = Convolution_InputSgsToRulesAndOutputSgs_OMP(
          SGs, SGs2, rb2, s2, s2, inS, outS);
    }
    for (int i = 0; i < Dimension; ++i)
      p2[i] = p3[i] = inS[i] = outS[i];
  }
}


extern "C" void scn_D_(getConvMask)(void **m, THLongTensor *inputSize, THLongTensor *filterSize, THFloatTensor *mask_){
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)

  RuleBook rb = _m.getValidRuleBook(inputSize, filterSize, true);
  auto mask = THFloatTensor_data(mask_);
  for (auto rbIdx = 0; rbIdx != rb.size(); ++rbIdx){
    auto rbIt = rb[rbIdx];
    for(long rbItIdx = 0; rbItIdx < rbIt.size(); rbItIdx+=2){
      mask[rbIt[rbItIdx]] ++;
    }
  }
}

extern "C" void scn_D_(getValidRules)(void **m, THLongTensor *inputSize, THLongTensor *filterSize){
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  // try{
  //     std::cout << "get valid rules c++"<<std::endl;
  _m.getValidRuleBook(inputSize, filterSize, true);
  // }
  // catch(const std::exception &exc){
  //     std::cerr << exc.what();
  //     std::cout<<"error in getValidRules"<<std::endl;
  // }
}

extern "C" void scn_D_(getConvRulesAndOutput)(void **m, THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize, THLongTensor *stride){
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  // try{
  //     std::cout << "get conv rules c++" << std::endl;
  _m.getRuleBook(inputSize, outputSize, filterSize, stride, true);
  // }
  // catch(const std::exception &exc){
  //     std::cerr << exc.what();
  //     std::cout<<"error in get conv Rules"<<std::endl;
  // }

}

extern "C" void scn_D_(getConvRules2AndOutput)(void **m, THLongTensor *inputSize, THLongTensor *outputSize, THLongTensor *filterSize, THLongTensor *stride){
  SCN_INITIALIZE_AND_REFERENCE(Metadata<Dimension>, m)
  _m.getRuleBook2(outputSize, inputSize, filterSize, stride, true);
}

extern "C" void scn_D_(freeMetadata)(void **m) {
  SCN_DELETE(Metadata<Dimension>, m)
}

#endif
