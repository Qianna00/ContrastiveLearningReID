![Python >=3.5](https://img.shields.io/badge/Python->=3.6-blue.svg)
![PyTorch >=1.6](https://img.shields.io/badge/PyTorch->=1.6-yellow.svg)


## Requirements

### Installation

```shell
git clone https://github.com/Qianna00/ContrastiveLearningReID.git
cd cl
python setup.py develop
```

### Prepare Datasets

```shell
cd examples && mkdir data
```
Download the person datasets Market-1501,MSMT17,PersonX,DukeMTMC-reID, the vehicle dataset VeRi-776 and the vessel dataset VesselReID
Then unzip them under the directory like

```
ContrastiveLearningReID/examples/data
├── market1501
│   └── Market-1501-v15.09.15
├── msmt17
│   └── MSMT17_V1
├── personx
│   └── PersonX
├── dukemtmcreid
│   └── DukeMTMC-reID
├── veri
│   └── VeRi
└── vesselreid
    └── VesselReID
```

### Prepare ImageNet Pre-trained Models for IBN-Net

When training with the backbone of [IBN-ResNet](https://arxiv.org/abs/1807.09441), you need to download the ImageNet-pretrained model from this [link](https://drive.google.com/drive/folders/1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S) and save it under the path of `examples/pretrained/`.

ImageNet-pretrained models for **ResNet-50** will be automatically downloaded in the python script.

## Training

We utilize 4 GTX-1080TI GPUs for training. For more parameter configuration, please check **`run_code.sh`**.

**examples:**

Market-1501:

1. Using DBSCAN:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16
```


2. Using InfoMap:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16
```

MSMT17:

1. Using DBSCAN:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d msmt17 --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16
```

2. Using InfoMap:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d msmt17 --iters 400 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16
```

DukeMTMC-reID:

1. Using DBSCAN:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d dukemtmcreid --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16
```

2. Using InfoMap:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d dukemtmcreid --iters 200 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16
```

VeRi-776

1. Using DBSCAN:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d veri --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16 --height 224 --width 224
```

2. Using InfoMap:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_infomap.py -b 256 -a resnet50 -d veri --iters 400 --momentum 0.1 --eps 0.5 --k1 15 --k2 4 --num-instances 16 --height 224 --width 224
```

## Evaluation

We utilize 1 GTX-1080TI GPU for testing. **Note that**

+ use `--width 128 --height 256` (default) for person datasets, `--height 224 --width 224` for vehicle datasets,  and `--height 180 --width 256` for vessel datasets

+ use `-a resnet50` (default) for the backbone of ResNet-50, and `-a resnet_ibn50a` for the backbone of IBN-ResNet.

To evaluate the model, run:
```shell
CUDA_VISIBLE_DEVICES=0 \
python examples/test.py \
  -d $DATASET --resume $PATH
```

**Some examples:**
```shell
### Market-1501 ###
CUDA_VISIBLE_DEVICES=0 \
python examples/test.py \
  -d market1501 --resume logs/spcl_usl/market_resnet50/model_best.pth.tar
```

# Acknowledgements

Thanks to Alibaba for opening source of his excellent works  [ClusterContrast](https://github.com/alibaba/cluster-contrast-reid). 
