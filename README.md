# Simple Baselines for Human Pose Estimation and Tracking
<p align="center">
<img src="/figs/1.jpg" width="400" height="250"> <img src="/figs/2.jpg" width="400" height="250">
<img src="/figs/3.jpg" width="400" height="300"> <img src="/figs/4.jpg" width="400" height="300">
</p>

## Introduction

This repo is **[TensorFlow](https://www.tensorflow.org)** implementation of **[Simple Baselines for Human Pose Estimation and Tracking (ECCV 2018)](https://arxiv.org/abs/1804.06208)** for **2D multi-person pose estimation** from a single RGB image.

**What this repo provides:**
* [TensorFlow](https://www.tensorflow.org) implementation of [Simple Baselines for Human Pose Estimation and Tracking](https://arxiv.org/abs/1804.06208).
* Flexible and simple code.
* Compatibility for most of the publicly available 2D multi-person pose estimation datasets including **[MPII](http://human-pose.mpi-inf.mpg.de/), [PoseTrack 2018](https://posetrack.net/), and [MS COCO 2017](http://cocodataset.org/#home)**.
* Human pose estimation visualization code (modified from [Detectron](https://github.com/facebookresearch/Detectron)).


## Dependencies
* [TensorFlow](https://www.tensorflow.org/)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cuDNN](https://developer.nvidia.com/cudnn)
* [Anaconda](https://www.anaconda.com/download/)
* [COCO API](https://github.com/cocodataset/cocoapi)

This code is tested under Ubuntu 16.04, CUDA 9.0, cuDNN 7.1 environment with Titan X GPUs (12GB VRAM).

Python 3.6.5 version with Anaconda 3 is used for development.

## Directory

### Root
The `${POSE_ROOT}` is described as below.
```
${POSE_ROOT}
|-- data
|-- lib
|-- main
|-- tool
|-- output
`-- requirement.txt
```
* `data` contains data loading codes and soft links to images and annotations directories.
* `lib` contains kernel codes for 2d multi-person pose estimation system.
* `main` contains high-level codes for training or testing the network.
* `tool` contains dataset converter. I set MS COCO as reference format and provide mpii2coco and posetrack2coco converting code.
* `output` contains log, trained models, visualized outputs, and test result.
* Run `pip install -r requirement.txt` to install required modules.

### Data
You need to follow directory structure of the `data` as below.
```
${POSE_ROOT}
|-- data
|-- |-- MPII
|   `-- |-- dets
|       |   |-- human_detection.json
|       |-- annotations
|       |   |-- train.json
|       |   `-- test.json
|       `-- images
|           |-- 000001163.jpg
|           |-- 000003072.jpg
|-- |-- PoseTrack
|   `-- |-- dets
|       |   |-- human_detection.json
|       |-- annotations
|       |   |-- train2018.json
|       |   |-- val2018.json
|       |   `-- test2018.json
|       |-- original_annotations
|       |   |-- train/
|       |   |-- val/
|       |   `-- test/
|       `-- images
|           |-- train/
|           |-- val/
|           `-- test/
|-- |-- COCO
|   `-- |-- dets
|       |   |-- human_detection.json
|       |-- annotations
|       |   |-- person_keypoints_train2017.json
|       |   |-- person_keypoints_val2017.json
|       |   `-- image_info_test-dev2017.json
|       `-- images
|           |-- train2017/
|           |-- val2017/
|           `-- test2017/
`-- |-- imagenet_weights
|       |-- resnet_v1_50.ckpt
|       |-- resnet_v1_101.ckpt
|       `-- resnet_v1_152.ckpt
```
* In the `tool`, run `python mpii2coco.py` to convert MPII annotation files to MS COCO format (`MPII/annotations`).
* In the `tool`, run `python posetrack2coco.py` to convert PoseTrack annotation files to MS COCO format (`PoseTrack/annotations`).
* In the training stage, GT human bbox is used, and `human_detection.json` is used in testing stage which should be prepared before testing.
* `human_detection.json` should follow [MS COCO format](http://cocodataset.org/#format-results).
* Download imagenet pre-trained resnet models from [tf-slim](https://github.com/tensorflow/models/tree/master/research/slim) and place it in the `data/imagenet_weights`.
* Except for `annotation` of the MPII and PoseTrack, all other directories are original version of downloaded ones.
* If you want to add your own dataset, you have to convert it to [MS COCO format](http://cocodataset.org/#format-data).
* You can change default directory structure of `data` by modifying `dataset.py` of each dataset folder.

### Output
You need to follow the directory structure of the `output` folder as below.
```
${POSE_ROOT}
|-- output
|-- |-- log
|-- |-- model_dump
|-- |-- result
`-- |-- vis
```
* Creating `output` folder as soft link form is recommended instead of folder form because it would take large storage capacity.
* `log` folder contains training log file.
* `model_dump` folder contains saved checkpoints for each epoch.
* `result` folder contains final estimation files generated in the testing stage.
* `vis` folder contains visualized results.
* You can change default directory structure of `output` by modifying `main/config.py`.

## Running TF-SimpleHumanPose
### Config
Except for special cases, you only need to modify codes in `main`.

In the `main/config.py`, you can change settings of the model including dataset to use, network backbone, and input size and so on.


### Train
In the `main` folder, run
```bash
python train.py --gpu 0-1
```
to train the network on the GPU 0-1. 

If you want to continue experiment, run 
```bash
python train.py --gpu 0-1 --continue
```

### Test
Place trained model at the `output/model_dump/$DATASET/` and human detection result (`human_detection.json`) to `data/$DATASET/dets/`.

In the `main` folder, run 
```bash
python test.py --gpu 0-1 --test_epoch 140
```
to test the network on the GPU 0-1 with 140th epoch trained model.

## Results
Here I report the performance of the model from this repo and [the original paper](https://arxiv.org/abs/1804.06208). Also, I provide pre-trained models and human detection results.
 
As this repo outputs compatible output files for MS COCO and PoseTrack, you can directly use [cocoapi](https://github.com/cocodataset/cocoapi) or [poseval]( https://github.com/leonid-pishchulin/poseval) to evaluate result on the MS COCO or PoseTrack dataset. You have to convert the produced `json` file to `mat` file to evaluate on MPII dataset following [this](http://human-pose.mpi-inf.mpg.de/#evaluation).

### Results on COCO val2017
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|---|---|---|---|---|---|---|---|---|---|
| 256x192_resnet50 (this repo) | 69.7 | 87.9 | 77.4 | 66.9 | 76.1 | 76.3 | 92.9 | 83.1 | 72.0 | 82.4 |
| 256x192_resnet50 ([original repo](https://github.com/Microsoft/human-pose-estimation.pytorch)) | 70.4 | 88.6 | 78.3 | 67.1 | 77.2 | 76.3 | 92.9 | 83.4 | 72.1 | 82.4 |

Note that there are some differences between the model from my repo and [original repo](https://github.com/Microsoft/human-pose-estimation.pytorch)
* Both of them is trained with 32 mini-batch per GPU, however mine used 2 GPUs while theirs used 4 GPUs (2x smaller total mini-batch size)
* I used human detection results of 55.3 AP, while theirs used that of 56.4 AP on human class of COCO val2017 dataset.

### Results on PoseTrack2018 validation set
Coming soon!

### Pre-trained models, pose estimation and human detection results
* 256x192_resnet50_COCO [[model](https://cv.snu.ac.kr/research/TF-SimpleHumanPose/human_pose_model/256x192_resnet50_COCO.zip)] [[pose_result](https://cv.snu.ac.kr/research/TF-SimpleHumanPose/human_pose_result/person_keypoints_256x192_resnet50_val2017_results.json)]
* Human detection on COCO val2017 (55.3 AP on human class) [[det_result](https://cv.snu.ac.kr/research/TF-SimpleHumanPose/human_det_result/human_detection_coco_val2017.json)]
* Human detection on COCO test-dev2017 (57.2 AP on human class) [[det_result](https://cv.snu.ac.kr/research/TF-SimpleHumanPose/human_det_result/human_detection_coco_test-dev2017.json)]
* Other human detection results on COCO val2017 [[Detectron_MODEL_ZOO](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md)]

## Acknowledgements
This repo is largely modified from [TensorFlow repo of CPN](https://github.com/chenyilun95/tf-cpn) and [PyTorch repo of Simple](https://github.com/Microsoft/human-pose-estimation.pytorch). Some hyperparameters such as learning rate are changed from [original work](https://arxiv.org/abs/1804.06208) because of framework conversion from [PyTorch](https://pytorch.org) to [TensorFlow](https://pytorch.org).

## Reference
[1] Xiao, Bin, Haiping Wu, and Yichen Wei. "Simple Baselines for Human Pose Estimation and Tracking". ECCV 2018.
