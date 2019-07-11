### SGC: *Efficient Semantic Scene Completion Network with Spatial Group Convolution*
Created by Jiahui Zhang, Hao Zhao, Anbang Yao, Yurong Chen, Li Zhang and Hongen Liao

### Citation
If you find our work useful in your research, please consider citing:

        @inproceedings{zhang2018efficient,
          title={Efficient Semantic Scene Completion Network with Spatial Group Convolution},
          author={Zhang, Jiahui and Zhao, Hao and Yao, Anbang and Chen, Yurong and Zhang, Li and Liao, Hongen},
          booktitle={European Conference on Computer Vision (ECCV)},
          year={2018}
        }

### Introduction
This work is based on our ECCV'18 paper. You can find the paper <a href="http://openaccess.thecvf.com/content_ECCV_2018/papers/Jiahui_Zhang_Efficient_Semantic_Scene_ECCV_2018_paper.pdf">here</a> for a quick overview. SGC is designed for accelerating the computation of 3D dense prediction tasks. We conduct experiments on the SUNCG dataset, achieving state-of-the-art performance (84.5% of IoU for scene completion and 70.5% IoU for semantic scene completion) and fast speed.

In this repository we release code on SUNCG dataset.

### Installation

0. Install <a href="https://github.com/facebookresearch/SparseConvNet.git">SparseConvNet</a>. This is a modified version of SparseConvNet. So you need to compile it yourself.

    Install the required packages:
    ```Shell
    pip install torch==0.3.1
    pip install git+https://github.com/pytorch/tnt.git@master
    pip install msgpack
    pip install msgpack_numpy
    pip install cffi
    sudo apt-get install libsparsehash-dev
    pip install matplotlib

    ```

    Compile SparseConvNet:

    ```Shell
    cd Pytorch
    python setup.py develop
    ```

1. Install SUNCG data toolbox. We provide a python wrapper for the <a href="https://github.com/shurans/sscnet">C++ functions</a> about SUNCG Dataset. 

    Compile SUNCG data toolbox.
    We have tested on boost 1.58.0 and python2
    ```Shell
    cd ssc/suncg_data_tools
    mkdir build
    cd build
    cmake ..
    make
    ```

### Generate data for training and testing
0. Download SUNCG data, refer to <a href="https://github.com/shurans/sscnet">SSCNet</a>. 
    ```Shell
    cd ssc/
    mkdir data
    wget http://sscnet.cs.princeton.edu/sscnet_release/data/depthbin_eval.zip
    unzip depthbin_eval.zip
    wget http://sscnet.cs.princeton.edu/sscnet_release/data/SUNCGtrain.zip
    unzip SUNCGtrain.zip
    mv SUNCGtrain* depthbin/
    ```

1. Prepare data used in our project.
    It will take a long time and generate about 700G data.
    ```Shell
    cd ssc/suncg_data_tools/script
    python prepare_data.py
    python prepare_weight.py
    ```

### Usage

Pretrained model are provided in ssc/baseline/log and ssc/sgc-pattern4/log.

0. For baseline network without using SGC:

    ```Shell
    cd ssc/baseline
    python sscnet
    ```
   
1. For network with SGC:

    ```Shell
    cd ssc/sgc-pattern4
    python sscnet
    ```
