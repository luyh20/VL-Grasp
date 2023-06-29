# RoboRefIt
RoboRefIt: An RGB-D Visual Grounding Dataset with Cluttered Indoor Scenes for Human-Robot Interaction.

[Homepage](https://luyh20.github.io/RoboRefIt.github.io/)


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


## RefTR Model for RoboRefIt

The RefTR model is from https://github.com/ubc-vision/RefTR.

### Requirements

References：
```shell
python=3.6.13
torch=1.8.0+cu111
torchvision=0.9.0+cu111
torchaudio=0.8.0
```

And others:
```shell
pip install -r requirements.txt
```

### Training and Evaluation

Training:
```shell
sh train_roborefit.sh
```

Evaluation：
```shell
sh test_roborefit.sh
```

### Model Zoo

You can download the models with ResNet-50 backbone and ResNet-101 backbone in [here](https://drive.google.com/u/0/uc?id=1P3TRJhM0DYZxeZtpY4kYFi9iX6AGh1u3&export=download). These models take RGB images as input. 


