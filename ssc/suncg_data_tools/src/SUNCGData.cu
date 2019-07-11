#include <vector>
#include <iostream>
#include <glog/logging.h>
#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>
#include "boost/python/stl_iterator.hpp"
#include "params.hpp"
#include "downsample.hpp"
#include "suncg_io.hpp"
#include "tsdf.hpp"
#include "device_alternate.hpp"
#include "weight.hpp"

using namespace boost::python;
// reference https://jianfengwang.wordpress.com/tag/python-numpy-cc/
boost::python::object downsampleLabel(PyObject* label_full){
  PyArrayObject* label_full_array = PyArray_GETCONTIGUOUS((PyArrayObject*)label_full);
  float* label_full_ptr = (float*)PyArray_DATA(label_full_array);
  int label_downscale = (data_full_vox_size[0] / label_vox_size[0]);
  int num_label_downscale = label_vox_size[0] * label_vox_size[1] * label_vox_size[2];
  float* label_downscale_ptr = new float[num_label_downscale];
  downsampleLabel_cpu(data_full_vox_size, label_vox_size, label_downscale, label_full_ptr, label_downscale_ptr);
 
  std::vector<float>  vec_down(label_downscale_ptr, label_downscale_ptr + num_label_downscale);
  npy_intp size_down = vec_down.size();
  float * data_down = size_down ? const_cast<float *>(&vec_down[0]) : static_cast<float *>(NULL); 
  PyObject * pyObj_down = PyArray_SimpleNewFromData( 1, &size_down, NPY_FLOAT, data_down );
  boost::python::handle<> handle_down( pyObj_down );
  boost::python::numeric::array arr_down( handle_down );  
  
  Py_DECREF((PyObject*)label_full_array);
  delete[] label_downscale_ptr;
  return arr_down.copy();
}

boost::python::object downsampleLabel2(PyObject* label_full, int label_downscale){
  PyArrayObject* label_full_array = PyArray_GETCONTIGUOUS((PyArrayObject*)label_full);
  float* label_full_ptr = (float*)PyArray_DATA(label_full_array);
  npy_intp *full_shape = PyArray_DIMS(label_full_array);
  std::vector<int> label_full_size(3);
  std::vector<int> label_downscale_size(3);
  for (int i=0; i<3; ++i){
      label_full_size[i] = full_shape[i];
      label_downscale_size[i] = full_shape[i] / label_downscale;
    }
  int num_label_downscale = label_downscale_size[0] * label_downscale_size[1] * label_downscale_size[2];
  float* label_downscale_ptr = new float[num_label_downscale];
  downsampleLabel_cpu(label_full_size, label_downscale_size, label_downscale, label_full_ptr, label_downscale_ptr);
 
  std::vector<float>  vec_down(label_downscale_ptr, label_downscale_ptr + num_label_downscale);
  npy_intp size_down = vec_down.size();
  float * data_down = size_down ? const_cast<float *>(&vec_down[0]) : static_cast<float *>(NULL); 
  PyObject * pyObj_down = PyArray_SimpleNewFromData( 1, &size_down, NPY_FLOAT, data_down );
  boost::python::handle<> handle_down( pyObj_down );
  boost::python::numeric::array arr_down( handle_down );  
  
  Py_DECREF((PyObject*)label_full_array);
  delete[] label_downscale_ptr;
  return arr_down.copy();
}

boost::python::object downsampleTSDF(PyObject* tsdf_full){
  PyArrayObject* tsdf_full_array =  PyArray_GETCONTIGUOUS((PyArrayObject*)tsdf_full);
  float* tsdf_full_ptr = (float*)PyArray_DATA(tsdf_full_array);
  
  int tsdf_downscale = (data_full_vox_size[0] / label_vox_size[0]);
  int num_tsdf_downscale = label_vox_size[0] * label_vox_size[1] * label_vox_size[2];
  float* tsdf_downscale_ptr = new float[num_tsdf_downscale];
  downsampleTSDF_cpu(data_full_vox_size, label_vox_size, tsdf_downscale, tsdf_full_ptr, tsdf_downscale_ptr);
  
  std::vector<float>  vec_down(tsdf_downscale_ptr, tsdf_downscale_ptr + num_tsdf_downscale);
  npy_intp size_down = vec_down.size();
  float * data_down = size_down ? const_cast<float *>(&vec_down[0]) : static_cast<float *>(NULL); 
  PyObject * pyObj_down = PyArray_SimpleNewFromData( 1, &size_down, NPY_FLOAT, data_down );
  boost::python::handle<> handle_down( pyObj_down );
  boost::python::numeric::array arr_down( handle_down );  
  
  Py_DECREF((PyObject*)tsdf_full_array);
  delete[] tsdf_downscale_ptr;
  return arr_down.copy();
}
boost::python::object getLabels(const std::string &filename){
  int num_full_voxels = data_full_vox_size[0] * data_full_vox_size[1] * data_full_vox_size[2];
  int num_label_voxels = label_vox_size[0] * label_vox_size[1] * label_vox_size[2];

  float vox_origin[3];
  float cam_pose[16];
  float *occupancy_label_full = new float[num_full_voxels];
  float *segmentation_label_full = new float[num_full_voxels];

  ReadVoxLabel(filename, vox_origin, cam_pose, occupancy_label_full, segmentation_class_map, segmentation_label_full);
  // SaveVoxLabel2Ply( outPath + file_without_extension + ".ply", label_vox_size, label_downscale, segmentation_label_downscale);
  
  std::vector<float>  vec_full(segmentation_label_full, segmentation_label_full + num_full_voxels);
  npy_intp size_full = vec_full.size();
  float * data_full = size_full ? const_cast<float *>(&vec_full[0]) : static_cast<float *>(NULL); 
  PyObject * pyObj_full = PyArray_SimpleNewFromData( 1, &size_full, NPY_FLOAT, data_full );
  boost::python::handle<> handle_full( pyObj_full );
  boost::python::numeric::array arr_full( handle_full );  

  // Free memory
  delete[] occupancy_label_full;
  delete[] segmentation_label_full;
  return arr_full.copy();
}

boost::python::object getTSDF(const std::string & depth_filename, const std::string & bin_filename, int gpu_num){
  cudaSetDevice(gpu_num);
  // read depth image
  float *depth_data = new float[frame_height * frame_width];
  float *depth_data_GPU;
  CUDA_CHECK(cudaMalloc(&depth_data_GPU,
                    frame_height * frame_width * sizeof(float)));

  ReadDepthImage(depth_filename, depth_data, frame_width, frame_height);
  CUDA_CHECK(cudaMemcpy(depth_data_GPU, depth_data, 
            frame_height * frame_width * sizeof(float),
                      cudaMemcpyHostToDevice));
  // read vox origin and cam pose
  float vox_origin[3];
  float cam_pose[16];
  std::ifstream fid(bin_filename, std::ios::binary);
  if(!fid){
    LOG(INFO)<<"could not open file" << bin_filename;
    exit(0);
  }
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

  float *tsdf_data_GPU;
  int num_voxels = data_full_vox_size[0]*data_full_vox_size[1]*data_full_vox_size[2];
  CUDA_CHECK(cudaMalloc(&tsdf_data_GPU, num_voxels * sizeof(float)));
  GPU_set_value(num_voxels, tsdf_data_GPU, float(1.0));
  ComputeTSDF(cam_info, vox_info, cam_info_GPU, vox_info_GPU, depth_data_GPU, tsdf_data_GPU);
  // transform tsdf to fliped tsdf
  int THREADS_NUM = 1024;
  int BLOCK_NUM = int((num_voxels + size_t(THREADS_NUM) - 1) / THREADS_NUM);
  tsdfTransform <<< BLOCK_NUM, THREADS_NUM >>> (vox_info_GPU, tsdf_data_GPU);

  float * tsdf_data_CPU = new float[num_voxels];
  // copy back to cpu
  CUDA_CHECK(cudaMemcpy(tsdf_data_CPU, tsdf_data_GPU,
                        num_voxels * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // handle python obj
  std::vector<float>  vec(tsdf_data_CPU, tsdf_data_CPU + num_voxels);
  npy_intp data_size = vec.size();
  float * tsdf_data = data_size ? const_cast<float *>(&vec[0]) : static_cast<float *>(NULL); 
  PyObject * pyObj = PyArray_SimpleNewFromData( 1, &data_size, NPY_FLOAT, tsdf_data );
  boost::python::handle<> handle( pyObj );
  boost::python::numeric::array arr( handle );
  
  // release memory
  delete[] depth_data;
  delete[] tsdf_data_CPU;
  CUDA_CHECK(cudaFree(depth_data_GPU));
  CUDA_CHECK(cudaFree(cam_info_GPU));
  CUDA_CHECK(cudaFree(vox_info_GPU));
  CUDA_CHECK(cudaFree(tsdf_data_GPU));
  
  return arr.copy();
}

boost::python::object getCompleteTSDF(const std::string & bin_filename, int gpu_num){
  cudaSetDevice(gpu_num);

  // read label
  int num_full_voxels = data_full_vox_size[0] * data_full_vox_size[1] * data_full_vox_size[2];

  float vox_origin[3];
  float cam_pose[16];
  float *occupancy_label_full = new float[num_full_voxels];
  float *segmentation_label_full = new float[num_full_voxels];

  ReadVoxLabel(bin_filename, vox_origin, cam_pose, occupancy_label_full, segmentation_class_map, segmentation_label_full);

  float *occupancy_label_full_GPU;
  CUDA_CHECK(cudaMalloc(&occupancy_label_full_GPU,
                    num_full_voxels * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(occupancy_label_full_GPU, occupancy_label_full, 
            num_full_voxels * sizeof(float), cudaMemcpyHostToDevice));
 
  float vox_info[8];
  vox_info[0] = vox_unit;
  vox_info[1] = vox_margin;
  for (int i = 0; i < 3; ++i)
    vox_info[i + 2] = float(data_full_vox_size[i]);
  // vox info[5~7] is not needed for computing complete tsdf

  float *vox_info_GPU;
  CUDA_CHECK(cudaMalloc(&vox_info_GPU, 8 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(vox_info_GPU, vox_info, 8 * sizeof(float),
                        cudaMemcpyHostToDevice));

  float *complete_tsdf_GPU;
  CUDA_CHECK(cudaMalloc(&complete_tsdf_GPU, num_full_voxels * sizeof(float)));
  int THREADS_NUM = 1024;
  int BLOCK_NUM = int((num_full_voxels + size_t(THREADS_NUM) - 1) / THREADS_NUM);
  CompleteTSDF<<<BLOCK_NUM, THREADS_NUM>>>(vox_info_GPU, occupancy_label_full_GPU, complete_tsdf_GPU);

  float *complete_tsdf_CPU = new float[num_full_voxels];
  // copy back to cpu
  CUDA_CHECK(cudaMemcpy(complete_tsdf_CPU, complete_tsdf_GPU,
                        num_full_voxels * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // handle python obj
  std::vector<float>  vec(complete_tsdf_CPU, complete_tsdf_CPU + num_full_voxels);
  npy_intp data_size = vec.size();
  float * tsdf_data = data_size ? const_cast<float *>(&vec[0]) : static_cast<float *>(NULL); 
  PyObject * pyObj = PyArray_SimpleNewFromData( 1, &data_size, NPY_FLOAT, tsdf_data );
  boost::python::handle<> handle( pyObj );
  boost::python::numeric::array arr( handle );
  
  // release memory
  delete[] occupancy_label_full;
  delete[] segmentation_label_full;
  delete[] complete_tsdf_CPU;
  CUDA_CHECK(cudaFree(occupancy_label_full_GPU));
  CUDA_CHECK(cudaFree(vox_info_GPU));
  CUDA_CHECK(cudaFree(complete_tsdf_GPU));
  
  return arr.copy();
}
// reference https://gist.github.com/marcinwol/b8df949bf8009cf856a3
boost::python::list getDataFiles(const boost::python::object& iterable) {

  std::vector<std::string> data_lists = std::vector<std::string>( boost::python::stl_input_iterator< std::string >( iterable ),
                             boost::python::stl_input_iterator< std::string >( ) );
  /*for(int i = 0; i < data_lists.size(); i++){
    LOG(INFO) << data_lists[i];
  }*/
  std::vector<std::string> data_filenames;
  GetFiles(data_lists, data_filenames, "camera_list_train.list", "0000.png");

  typename std::vector<std::string>::iterator iter;
  boost::python::list data_files;
  for (iter = data_filenames.begin(); iter != data_filenames.end(); ++iter) {
    data_files.append(*iter);
  }
  return data_files;
}

boost::python::object getHardPos(PyObject* label_downscale, int margin, int gpu_num){
  cudaSetDevice(gpu_num);
  PyArrayObject* label_downscale_array =  PyArray_GETCONTIGUOUS((PyArrayObject*)label_downscale);
  float* label_downscale_ptr = (float*)PyArray_DATA(label_downscale_array);
  
  int num_label_voxels = label_vox_size[0] * label_vox_size[1] * label_vox_size[2];
  float* hard_pos = new float[num_label_voxels];
  DetermineHardPos(&label_vox_size[0], num_label_voxels, label_downscale_ptr, hard_pos, margin);

  std::vector<float>  vec(hard_pos, hard_pos + num_label_voxels);
  npy_intp size = vec.size();
  float * data = size ? const_cast<float *>(&vec[0]) : static_cast<float *>(NULL); 
  PyObject * pyObj = PyArray_SimpleNewFromData( 1, &size, NPY_FLOAT, data );
  boost::python::handle<> handle( pyObj );
  boost::python::numeric::array arr( handle );  
  
  Py_DECREF((PyObject*)label_downscale_array);
  delete[] hard_pos;
  return arr.copy();
}
boost::python::object getHardPos2(PyObject* label_downscale, int margin, int gpu_num){
  cudaSetDevice(gpu_num);
  PyArrayObject* label_downscale_array =  PyArray_GETCONTIGUOUS((PyArrayObject*)label_downscale);
  float* label_downscale_ptr = (float*)PyArray_DATA(label_downscale_array);
  npy_intp *shape = PyArray_DIMS(label_downscale_array);
  std::vector<int> label_size(3);
  for (int i=0; i<3; ++i){
      label_size[i] = shape[i];
  }
  int num_label_voxels = label_size[0] * label_size[1] * label_size[2];
  float* hard_pos = new float[num_label_voxels];
  DetermineHardPos(&label_size[0], num_label_voxels, label_downscale_ptr, hard_pos, margin);

  std::vector<float>  vec(hard_pos, hard_pos + num_label_voxels);
  npy_intp size = vec.size();
  float * data = size ? const_cast<float *>(&vec[0]) : static_cast<float *>(NULL); 
  PyObject * pyObj = PyArray_SimpleNewFromData( 1, &size, NPY_FLOAT, data );
  boost::python::handle<> handle( pyObj );
  boost::python::numeric::array arr( handle );  
  
  Py_DECREF((PyObject*)label_downscale_array);
  delete[] hard_pos;
  return arr.copy();
}

#if PY_VERSION_HEX >= 0x03000000
void *
#else
void
#endif
initialize()
{
  import_array();
}

BOOST_PYTHON_MODULE(SUNCGData)
{
  initialize();
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  boost::python::def("downsampleLabel",downsampleLabel);
  boost::python::def("downsampleLabel2",downsampleLabel2);
  boost::python::def("downsampleTSDF",downsampleTSDF);
  boost::python::def("getLabels",getLabels);
  boost::python::def("getTSDF",getTSDF);
  boost::python::def("getCompleteTSDF",getCompleteTSDF);
  boost::python::def("getDataFiles",getDataFiles);
  boost::python::def("getHardPos",getHardPos);
  boost::python::def("getHardPos2",getHardPos2);
}

