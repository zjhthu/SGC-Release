#ifndef _WEIGHT_HPP_
#define _WEIGHT_HPP_

#include "device_alternate.hpp"

template <typename Dtype> 
__global__ 
void DetermineHardPosKernel (int * label_vox_size_GPU, Dtype* segmentation_label_downscale_gpu, Dtype* hard_pos_gpu, int margin){
    int vox_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (vox_idx >= label_vox_size_GPU[0] * label_vox_size_GPU[1] * label_vox_size_GPU[2]){
        return;
    }
    int z = int((vox_idx / ( label_vox_size_GPU[0] * label_vox_size_GPU[1]))%label_vox_size_GPU[2]) ;
    int y = int((vox_idx / label_vox_size_GPU[0]) % label_vox_size_GPU[1]);
    int x = int(vox_idx % label_vox_size_GPU[0]);

    hard_pos_gpu[vox_idx] = -1;// default, neg example
    if (segmentation_label_downscale_gpu[vox_idx] == 255){
        hard_pos_gpu[vox_idx] = -2;
        return;
    }
    if (segmentation_label_downscale_gpu[vox_idx] >0 && segmentation_label_downscale_gpu[vox_idx] <255){
        Dtype pos_label = segmentation_label_downscale_gpu[vox_idx];
        int same_label_count = 0;
        for (int iix = max(0,x-margin); iix <= min((int)label_vox_size_GPU[0]-1,x+margin); iix++){
          for (int iiy = max(0,y-margin); iiy <= min((int)label_vox_size_GPU[1]-1,y+margin); iiy++){
            for (int iiz = max(0,z-margin); iiz <= min((int)label_vox_size_GPU[2]-1,z+margin); iiz++){
              int iidx = iiz * label_vox_size_GPU[0] * label_vox_size_GPU[1] + iiy * label_vox_size_GPU[0] + iix;
              if (segmentation_label_downscale_gpu[iidx] == pos_label){
                same_label_count++;
              }
            }
          }
        }
        int threshold = (margin*2+1)*(margin*2+1)*(margin*2+1);
        if(same_label_count == threshold)
          hard_pos_gpu[vox_idx] = 0; // easy pos example
        else if (same_label_count < threshold)
          hard_pos_gpu[vox_idx] = 1; // hard pos example
        return;
    }
}

/*-------------------- determine hard pos example --------------------*/
template <typename Dtype> 
void DetermineHardPos (const int * label_vox_size, int num_label_voxels, Dtype * segmentation_label_downscale,  Dtype * hard_pos, int margin) {

    Dtype * segmentation_label_downscale_GPU;
    CUDA_CHECK(cudaMalloc(&segmentation_label_downscale_GPU, num_label_voxels * sizeof(Dtype)));
    CUDA_CHECK(cudaMemcpy(segmentation_label_downscale_GPU, segmentation_label_downscale, 
                        num_label_voxels * sizeof(Dtype), cudaMemcpyHostToDevice));
    int * label_vox_size_GPU;
    CUDA_CHECK(cudaMalloc(&label_vox_size_GPU, 3 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(label_vox_size_GPU, label_vox_size, 3 * sizeof(int), cudaMemcpyHostToDevice));

    Dtype * hard_pos_gpu;
    CUDA_CHECK(cudaMalloc(&hard_pos_gpu, num_label_voxels * sizeof(Dtype)));

    int THREADS_NUM = 1024;
    int BLOCK_NUM = int((num_label_voxels + size_t(THREADS_NUM) - 1) / THREADS_NUM);
    DetermineHardPosKernel<<< BLOCK_NUM, THREADS_NUM >>>(label_vox_size_GPU, segmentation_label_downscale_GPU, hard_pos_gpu, margin);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(hard_pos, hard_pos_gpu, num_label_voxels * sizeof(Dtype), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(segmentation_label_downscale_GPU));
    CUDA_CHECK(cudaFree(label_vox_size_GPU));
    CUDA_CHECK(cudaFree(hard_pos_gpu));
}



#endif
