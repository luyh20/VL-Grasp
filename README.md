# VL-Grasp
Official Implementation for paper "VL-Grasp: a 6-Dof Interactive Grasp Policy for Language-Oriented Objects in Cluttered Indoor Scenes" IROS 2023. 

[arxiv](https://arxiv.org/abs/2308.00640)

The VL-Grasp is a interactive grasp policy combined with visual grounding and 6-dof grasp pose detection tasks. The robot can adapt to various observation views and more diverse indoor scenes to grasp the target according to a human's language command by applying the VL-Grasp. Meanwhile, we build a new visual grounding dataset specially designed for the robot interaction grasp task, called RoboRefIt.

![teaser](logs/pipeline.png)

## Download the RoboRefIt Dataset


You can download **RoboRefIt** from [Google Drive](https://drive.google.com/file/d/1pdGF1HaU_UiKfh5Z618hy3nRjVbq_VuW/view?usp=sharing).
The data directories should like this:

```
RoboRefIt/
├── data/
│   └── final_dataset/
│       ├── train
│       ├── testA
│       └── testB
```

The homepage of the **RoboRefIt** dataset is at [RoboRefIt](https://luyh20.github.io/RoboRefIt.github.io/). More details and statistics information about the dataset can be found in the homepage.


## Requirements

References：
```shell
python=3.7.16
torch=1.7.1+cu110
torchvision=0.8.2+cu110
torchaudio=0.7.2
```

And others:
```shell
cd RoboRefIt
pip install -r requirements.txt

cd GraspNet
pip install -r requirements.txt

cd GraspNet/knn
python setup.py install 

cd GraspNet/pointnet2
python setup.py install
```

## Training 
There are two stages of model training.

### Visual Grounding Network

Training with the RoboRefIt dataset.
Please download the dataset and allocate the dataset parameters at "./RoboRefIt/main_vg.py".

```shell
cd RoboRefIt
sh train_roborefit.sh
```
Thansks for the [RefTR model](https://github.com/ubc-vision/RefTR).

### 6-Dof Grasp Pose Detection Network

Training with the [GraspNet-1Billion dataset](https://graspnet.net/datasets.html).
Please download the dataset and allocate the dataset parameters at "./GraspNet/train.py".

```shell
cd GraspNet
sh train.sh
```
Thansks for the [FGC-GraspNet model](https://github.com/luyh20/FGC-GraspNet).

## Model

There are Google Drive links, checkpoint_best_r50.pth [https://drive.google.com/file/d/1HJQKnuiG5J02PZBQJc2KsW3RozmJYH-L/view?usp=sharing] 
and checkpoint_fgc.tar [https://drive.google.com/file/d/1x4e23njqi4A_S_LlZPCUHjHT9CqFPzkc/view?usp=sharing].

## Demo 

```shell
python main.py
```

## BibTeX
```
@inproceedings{lu2023vl,
  title={VL-Grasp: a 6-Dof Interactive Grasp Policy for Language-Oriented Objects in Cluttered Indoor Scenes},
  author={Lu, Yuhao and Fan, Yixuan and Deng, Beixing and Liu, Fangfu and Li, Yali and Wang, Shengjin},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={976--983},
  year={2023},
  organization={IEEE}
}
```



