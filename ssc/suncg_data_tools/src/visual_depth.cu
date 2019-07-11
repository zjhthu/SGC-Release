#include "headers/device_alternate.hpp"
#include "headers/params.hpp"
#include "headers/suncg_io.hpp"
#include "headers/tsdf.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>

void SavePts2Ply(const std::string &filename, float* pts, int num_points);
void ReadDepthImage(const std::string &filename, float * depth_data, int frame_width, int frame_height);
void SaveVoxLabel2Ply(const std::string &filename, int* vox_size, int label_downscale, float * vox_label);
void ComputeGrid(float* cam_info_CPU, float* vox_info_CPU, float* cam_info_GPU, float* vox_info_GPU, float* depth_data_GPU,  float* vox_binary_GPU);

int main(int argc, char * argv[]){
	// if (argc != 4){
	// 	std::cout << "[usage] depth2ply depth_name cam_K_name out_ply_name" << std::endl;
	// 	return 1;
	// }
	std::string depth_name = argv[1];
	std::string bin_filename = argv[2];
	std::string raw_ply_name = argv[3];
	std::string binary_ply_name = argv[4];

	// read depth image
	float *depth_data = new float[frame_height * frame_width];
	float *depth_data_GPU;
	CUDA_CHECK(cudaMalloc(&depth_data_GPU,
	                frame_height * frame_width * sizeof(float)));

	ReadDepthImage(depth_name, depth_data, frame_width, frame_height);
	CUDA_CHECK(cudaMemcpy(depth_data_GPU, depth_data, 
            frame_height * frame_width * sizeof(float),
                      cudaMemcpyHostToDevice));

	// read vox origin and cam pose
	float vox_origin[3];
	float cam_pose[16];
	std::ifstream fid(bin_filename, std::ios::binary);
	for (int i = 0; i < 3; ++i){
	fid.read((char*)&vox_origin[i], sizeof(float));
	}
	for (int i = 0; i < 16; ++i){
	fid.read((char*)&cam_pose[i], sizeof(float));
	}
	// get cam info and vox info
	float cam_info[27];
	cam_info[0] = float(frame_width);
	cam_info[1] = float(frame_height);
	for (int i = 0; i < 9; ++i)
	cam_info[i + 2] = cam_K[i];
	for (int i = 0; i < 16; ++i)
	cam_info[i + 11] = cam_pose[i];

	float * raw_points = new float[frame_width*frame_height*3];
	int point_idx = 0;
	for (int y = 0; y < frame_height; y++){
		for (int x = 0; x < frame_width; x++){
			float point_depth = depth_data[x + y*frame_width];
			if (point_depth > 0){
				float point_cam[3] = {0};
				point_cam[0] =  (x - cam_K[2])*point_depth/cam_K[0];
				point_cam[1] =  (y - cam_K[5])*point_depth/cam_K[4];
				point_cam[2] =  point_depth;

				float point_base[3] = {0};

				point_base[0] = cam_pose[0 * 4 + 0]* point_cam[0] + cam_pose[0 * 4 + 1]*  point_cam[1] + cam_pose[0 * 4 + 2]* point_cam[2];
				point_base[1] = cam_pose[1 * 4 + 0]* point_cam[0] + cam_pose[1 * 4 + 1]*  point_cam[1] + cam_pose[1 * 4 + 2]* point_cam[2];
				point_base[2] = cam_pose[2 * 4 + 0]* point_cam[0] + cam_pose[2 * 4 + 1]*  point_cam[1] + cam_pose[2 * 4 + 2]* point_cam[2];

				point_base[0] = point_base[0] + cam_pose[0 * 4 + 3];
				point_base[1] = point_base[1] + cam_pose[1 * 4 + 3];
				point_base[2] = point_base[2] + cam_pose[2 * 4 + 3];
                
                int z = (int)floor((point_base[0] - vox_origin[0])/vox_unit);
                int x = (int)floor((point_base[1] - vox_origin[1])/vox_unit);
                int y = (int)ceil((point_base[2] - vox_origin[2])/vox_unit)+2;

                if( x >= 0 && x < vox_size[0] && y >= 0 && y < vox_size[1] && z >= 0 && z < vox_size[2]){
				    raw_points[point_idx*3] = x;
				    raw_points[point_idx*3+1] = y;
				    raw_points[point_idx*3+2] = z;
				    point_idx++;
                }
			}
		}
	}
	SavePts2Ply(raw_ply_name, raw_points, point_idx);


	float vox_info[8];
	vox_info[0] = vox_unit;
	vox_info[1] = vox_margin;
	for (int i = 0; i < 3; ++i)
	vox_info[i + 2] = float(data_full_vox_size[i]);
	vox_info[5] = vox_origin[0];
	vox_info[6] = vox_origin[1];
	vox_info[7] = vox_origin[2];

	float *cam_info_GPU;
	float *vox_info_GPU;
	CUDA_CHECK(cudaMalloc(&cam_info_GPU, 27 * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&vox_info_GPU, 8 * sizeof(float)));
	CUDA_CHECK(cudaMemcpy(cam_info_GPU, cam_info, 27 * sizeof(float),
	                    cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(vox_info_GPU, vox_info, 8 * sizeof(float),
	                    cudaMemcpyHostToDevice));

	float *vox_binary_GPU;
	int num_voxels = data_full_vox_size[0]*data_full_vox_size[1]*data_full_vox_size[2];
	CUDA_CHECK(cudaMalloc(&vox_binary_GPU, num_voxels * sizeof(float)));
	GPU_set_value(num_voxels, vox_binary_GPU, float(0));
	ComputeGrid(cam_info, vox_info, cam_info_GPU, vox_info_GPU, depth_data_GPU, vox_binary_GPU);

	float * vox_binary_CPU = new float [num_voxels];
	CUDA_CHECK(cudaMemcpy(vox_binary_CPU, vox_binary_GPU, num_voxels * sizeof(float),
	                    cudaMemcpyDeviceToHost));

	SaveVoxLabel2Ply(binary_ply_name, &vox_size[0], 1, vox_binary_CPU);


	CUDA_CHECK(cudaFree(vox_binary_GPU));
	CUDA_CHECK(cudaFree(cam_info_GPU));
	CUDA_CHECK(cudaFree(vox_info_GPU));
	CUDA_CHECK(cudaFree(depth_data_GPU));

	delete [] raw_points;
	delete [] depth_data;
	delete [] vox_binary_CPU;
	return 0;
}

void ComputeGrid(float* cam_info_CPU, float* vox_info_CPU,
                 float* cam_info_GPU, float* vox_info_GPU,
                 float* depth_data_GPU,  float* vox_binary_GPU) {

  int frame_width  = cam_info_CPU[0];
  int frame_height = cam_info_CPU[1];
  int vox_size[3];
  for (int i = 0; i < 3; ++i)
    vox_size[i] = vox_info_CPU[i + 2];
  int num_crop_voxels = vox_size[0] * vox_size[1] * vox_size[2];

  // from depth map to binaray voxel representation 
  depth2Grid<<<frame_width,frame_height>>>(cam_info_GPU, vox_info_GPU, depth_data_GPU, vox_binary_GPU);
  CUDA_CHECK(cudaGetLastError());
}

// Save voxel volume labels to point cloud ply file for visualization
void SavePts2Ply(const std::string &filename, float* pts, int num_points) {
  // Create header for ply file
  FILE *fp = fopen(filename.c_str(), "w");
  fprintf(fp, "ply\n");
  fprintf(fp, "format binary_little_endian 1.0\n");
  fprintf(fp, "element vertex %d\n", num_points);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "end_header\n");

  // Create point cloud content for ply file
  for (int i = 0; i < num_points; ++i) {
	fwrite(&pts[i*3], sizeof(float), 1, fp);
	fwrite(&pts[i*3+1], sizeof(float), 1, fp);
	fwrite(&pts[i*3+2], sizeof(float), 1, fp);
  }
  fclose(fp);
}

// Save voxel volume labels to point cloud ply file for visualization
void SaveVoxLabel2Ply(const std::string &filename, int* vox_size, int label_downscale, float * vox_label) {

  // Count total number of points in point cloud
  int num_points = 0;
  for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; ++i)
    if (CPUStorage2ComputeT(vox_label[i]) > 0)
      num_points++;

  // Create header for ply file
  FILE *fp = fopen(filename.c_str(), "w");
  fprintf(fp, "ply\n");
  fprintf(fp, "format binary_little_endian 1.0\n");
  fprintf(fp, "element vertex %d\n", num_points);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "property uchar red\n");
  fprintf(fp, "property uchar green\n");
  fprintf(fp, "property uchar blue\n");
  fprintf(fp, "end_header\n");

  // Create different colors for each class
  const int num_classes = 36;
  int class_colors[num_classes * 3];
  for (int i = 0; i < num_classes; ++i) {
    class_colors[i * 3 + 0] = (int)(std::round(GenRandFloat(0.0f, 255.0f)));
    class_colors[i * 3 + 1] = (int)(std::round(GenRandFloat(0.0f, 255.0f)));
    class_colors[i * 3 + 2] = (int)(std::round(GenRandFloat(0.0f, 255.0f)));
  }

  // Create point cloud content for ply file
  for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; ++i) {

    // If class of voxel non-empty, add voxel coordinates to point cloud
    if (CPUStorage2ComputeT(vox_label[i]) > 0) {

      // Compute voxel indices in int for higher positive number range
      int z = floor(i / (vox_size[0] * vox_size[1]));
      int y = floor((i - (z * vox_size[0] * vox_size[1])) / vox_size[0]);
      int x = i - (z * vox_size[0] * vox_size[1]) - (y * vox_size[0]);

      // Convert voxel indices to float, and save coordinates to ply file
      float float_x = (float)x * (float)label_downscale + (float)label_downscale / 2;
      float float_y = (float)y * (float)label_downscale + (float)label_downscale / 2;
      float float_z = (float)z * (float)label_downscale + (float)label_downscale / 2;
      fwrite(&float_x, sizeof(float), 1, fp);
      fwrite(&float_y, sizeof(float), 1, fp);
      fwrite(&float_z, sizeof(float), 1, fp);

      // Save color of class into voxel
      unsigned char color_r = (unsigned char) class_colors[(int)CPUStorage2ComputeT(vox_label[i]) * 3 + 0];
      unsigned char color_g = (unsigned char) class_colors[(int)CPUStorage2ComputeT(vox_label[i]) * 3 + 1];
      unsigned char color_b = (unsigned char) class_colors[(int)CPUStorage2ComputeT(vox_label[i]) * 3 + 2];
      fwrite(&color_r, sizeof(unsigned char), 1, fp);
      fwrite(&color_g, sizeof(unsigned char), 1, fp);
      fwrite(&color_b, sizeof(unsigned char), 1, fp);
    }
  }
  fclose(fp);
}

void ReadDepthImage(const std::string &filename, float * depth_data,
                    int frame_width, int frame_height) {
  cv::Mat depth_image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_ANYDEPTH);
  unsigned short * depth_raw = new unsigned short[frame_height * frame_width];
  for (int i = 0; i < frame_height * frame_width; ++i) {
    depth_raw[i] = ((((unsigned short)depth_image.data[i * 2 + 1]) << 8) +
        ((unsigned short)depth_image.data[i * 2 + 0]));
    depth_raw[i] = (depth_raw[i] << 13 | depth_raw[i] >> 3);
    depth_data[i] = ((float)depth_raw[i] / 1000.0f);
  }
  delete [] depth_raw;
}

