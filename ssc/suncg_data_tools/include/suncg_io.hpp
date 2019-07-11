#ifndef _SUNCG_IO_HPP_
#define _SUNCG_IO_HPP_
#include "params.hpp"
#include <iostream>
#include <glog/logging.h>
#include <vector>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <dirent.h>
#include <random>

#define ComputeT float
#define StorageT float
#define CPUStorage2ComputeT(x) (x)
// Generate random float
float GenRandFloat(float min, float max){
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(min, max - 0.0001);
  return dist(mt);
}
template <typename Dtype>
void ReadVoxLabel(const std::string &filename, Dtype *vox_origin,
                  Dtype *cam_pose, Dtype *occupancy_label_fullsize,
                  std::vector<int> segmentation_class_map,
                  Dtype *segmentation_label_fullsize) {

  std::ifstream fid(filename, std::ios::binary);
  if(!fid){
    LOG(INFO)<<"could not open file:" << filename;
  }
  for (int i = 0; i < 3; ++i){
    fid.read((char*)&vox_origin[i], sizeof(Dtype));
  }
  for (int i = 0; i < 16; ++i){
    fid.read((char*)&cam_pose[i], sizeof(Dtype));
  }
  std::vector<unsigned int> scene_vox_RLE;
  while (!fid.eof()) {
    int tmp;
    fid.read((char*)&tmp, sizeof(int));
    if (!fid.eof())
      scene_vox_RLE.push_back(tmp);
  }
  int vox_idx = 0;
  bool logout = false;
  for (size_t i = 0; i < scene_vox_RLE.size() / 2; ++i) {
    unsigned int vox_val = scene_vox_RLE[i * 2];
    unsigned int vox_iter = scene_vox_RLE[i * 2 + 1];

    for (size_t j = 0; j < vox_iter; ++j) {
      if ((vox_val == 255) || (vox_val >= segmentation_class_map.size())) {
        segmentation_label_fullsize[vox_idx] = Dtype(255);
        occupancy_label_fullsize[vox_idx] = Dtype(0.0f);
        if((vox_val >= segmentation_class_map.size()) && (vox_val != 255)){
          if(!logout){
            LOG(INFO) << "label:" << vox_val <<  "out of range!!!";
            logout = true;
          }
        }
      } else {
        segmentation_label_fullsize[vox_idx] = Dtype(segmentation_class_map[vox_val]);
        if (vox_val > 0)
          occupancy_label_fullsize[vox_idx] = Dtype(1.0f);
        else
          occupancy_label_fullsize[vox_idx] = Dtype(0.0f);
      }
      vox_idx++;
    }
  }
}

template <typename Dtype>
void ReadDepthImage(const std::string &filename, Dtype * depth_data,
                    int frame_width, int frame_height) {
  cv::Mat depth_image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_ANYDEPTH);
  unsigned short * depth_raw = new unsigned short[frame_height * frame_width];
  for (int i = 0; i < frame_height * frame_width; ++i) {
    depth_raw[i] = ((((unsigned short)depth_image.data[i * 2 + 1]) << 8) +
        ((unsigned short)depth_image.data[i * 2 + 0]));
    depth_raw[i] = (depth_raw[i] << 13 | depth_raw[i] >> 3);
    depth_data[i] = Dtype((float)depth_raw[i] / 1000.0f);
  }
  delete [] depth_raw;
}


// Save voxel volume labels to point cloud ply file for visualization
void SaveVoxLabel2Ply(const std::string &filename, std::vector<int> vox_size, int label_downscale, StorageT * vox_label) {

  // Count total number of points in point cloud
  int num_points = 0;
  for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; ++i)
    if (CPUStorage2ComputeT(vox_label[i]) > 0  && CPUStorage2ComputeT(vox_label[i]) < 255)
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
    if (CPUStorage2ComputeT(vox_label[i]) > 0 && CPUStorage2ComputeT(vox_label[i]) < 255) {

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

// Read in camera List

struct SceneMeta{
    std::string sceneId;
    int roomId;
    int floorId;
    std::vector< std::vector<float> > extrinsics;
    int activityId;
    //std::vector<AnnoMeta> annotationList;
};

std::vector <SceneMeta> readList(std::string file_list){
  std::cout<<"loading file "<<file_list<<"\n";
  FILE* fp = fopen(file_list.c_str(),"rb");
  if (fp==NULL) { std::cout<<"fail to open file: "<<file_list<<std::endl; exit(EXIT_FAILURE); }

  std::vector <SceneMeta> sceneMetaList;
  float vx, vy, vz, tx, ty, tz, ux, uy, uz, rx, ry, rz;
  int idx = 0;
  while (feof(fp)==0) {
    SceneMeta Scene;
    int cameraPoseLen = 0;
    int numAct = 0;
    int anno_Id =0;
    char SceneId[100];
    int res = fscanf(fp, "%s %d %d %d %d %d %d", SceneId, &Scene.floorId, &Scene.roomId, &Scene.activityId, &anno_Id, &numAct, &cameraPoseLen);

    if (res==7){
      Scene.sceneId.append(SceneId);
      Scene.extrinsics.resize(cameraPoseLen);

      for (int i = 0; i < cameraPoseLen; i++){
        int res1 = fscanf(fp, "%f%f%f%f%f%f%f%f%f%f%f%f", &vx, &vy, &vz, &tx, &ty, &tz, &ux, &uy, &uz, &rx, &ry, &rz);
        Scene.extrinsics[i].resize(16);
        Scene.extrinsics[i][0]  = rx; Scene.extrinsics[i][1]  = -ux; Scene.extrinsics[i][2]  = tx;  Scene.extrinsics[i][3]  = vx;
        Scene.extrinsics[i][4]  = ry; Scene.extrinsics[i][5]  = -uy; Scene.extrinsics[i][6]  = ty;  Scene.extrinsics[i][7]  = vy;
        Scene.extrinsics[i][8]  = rz; Scene.extrinsics[i][9]  = -uz; Scene.extrinsics[i][10] = tz;  Scene.extrinsics[i][11] = vz;
        Scene.extrinsics[i][12] = 0 ; Scene.extrinsics[i][13] = 0;   Scene.extrinsics[i][14] = 0;   Scene.extrinsics[i][15] = 1;
      }
      sceneMetaList.push_back(Scene);
      idx++;
    }else{
      break;
    }
  }
  fclose(fp);
  std::cout<<"finish loading scene "<<"\n";
  return sceneMetaList;
}


// Check if file exists
bool FileExists(const std::string &filepath) {
  std::ifstream file(filepath);
  return (!file.fail());
}

// Return all files in directory using search string
void GetFilesInDir(const std::vector<std::string> &directories, std::vector<std::string> &file_list, const std::string &search_string) {
  DIR *dir;
  struct dirent *ent;
  for (int i =0;i<directories.size();i++){
    std::string directory = directories[i];
    if ((dir = opendir (directory.c_str())) != NULL) {
      while ((ent = readdir (dir)) != NULL) {
        std::string filename(ent->d_name);
        if (filename.find(search_string) != std::string::npos && filename != "." && filename != ".."){
          filename = directory +"/"+ filename;
          file_list.push_back(filename);
        }
      }
      closedir (dir);
    } else {
      perror ("Error: could not look into directory!");
    }
  }
  LOG(INFO) << "total number of files: "<<file_list.size();
  std::sort(file_list.begin(),  file_list.end());
  //std::cin.ignore();
}

void GetFilesByList(const std::vector<std::string> &directories, std::vector<std::string> &file_list, std::string cameraListFile){
  for (int i =0;i<directories.size();i++){
    std::string list_file = directories[i] + "/" + cameraListFile;
    std::vector <SceneMeta>  sceneMetaList =readList(list_file);
    int curr_frame = 0;
    for (int j = 0; j < sceneMetaList.size(); j++){
      char buff2[100];
      sprintf(buff2, "%08d_%s_fl%03d_rm%04d_%04d",j, sceneMetaList[j].sceneId.c_str(), sceneMetaList[j].floorId,sceneMetaList[j].roomId,curr_frame);
      std::string  fileDepth = directories[i] + "/" + buff2 + ".png";
      file_list.push_back(fileDepth);
    }
    std::cout<<"read in :"<< list_file<< " : "<<sceneMetaList.size()<<std::endl;

  }
  std::cout<<"Total number of data : "<< file_list.size() <<std::endl;
}

void GetFiles(const std::vector<std::string> &directories, std::vector<std::string> &file_list,  const std::string cameraListFile, const std::string &search_string){
  std::string list_file = directories[0] + "/" + cameraListFile;
  if (FileExists(list_file)) {
    std::cout<<"List file exist : "<< list_file <<std::endl;
    GetFilesByList( directories, file_list, cameraListFile);
  }else{
    std::cout<<"List file not exist : "<< list_file <<std::endl;
    GetFilesInDir( directories, file_list, search_string);
  }
}
#endif
