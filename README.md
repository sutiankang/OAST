# Unsupervised Video Object Segmentation with Online Adversarial Self-Tuning

## Prerequisites
The training and testing experiments are conducted using PyTorch 1.10.1 with two GeForce RTX 2080Ti GPUs with 11GB Memory.
- Python 3.8
```
conda create -n mp-vos python=3.8
```
Other minor Python modules can be installed by running
```
pip install -r requirements.txt
```

## Train

### Download Datasets
In the paper, we use the following three public available dataset for training. Here are some steps to prepare the data:
- [DAVIS-2016](https://davischallenge.org/davis2017/code.html): We use all the data in the train subset of DAVIS-2016. However, please download DAVIS-2017 dataset, it will automatically choose the subset of DAVIS-2016 for training. We use all training data as labeled data in DAVIS-2016.
- [YouTubeVOS-2018](https://youtube-vos.org/dataset/): We sample the training data every 10 frames as labeled data in YoutubeVOS-2018. You can sample any number of frames to train the model by modifying parameter ```--youtube_stride```.
- [FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets/moseg.en.html): We only use FBMS dataset for evaluating.
- [Youtube-objects](https://data.vision.ee.ethz.ch/cvl/youtube-objects/): We apply val frames as unlabeled data in Youtube-objects.

Note that these datasets are all public.

### Prepare Optical Flow
Please following the instruction of [RAFT](https://github.com/princeton-vl/RAFT) to prepare the optical flow. Note that this repository is from [RAFT: Recurrent All Pairs Field Transforms for Optical Flow (ECCV 2020)](https://arxiv.org/pdf/2003.12039.pdf).

The structure of datasets is as follows:
```
|—— Datasets
  |—— YouTubeVOS-2018
    |—— train
      |—— images
        |—— 00000.jpg
        |—— 00005.jpg
        |—— ...
      |—— flows
        |—— 00000.jpg
        |—— 00005.jpg
        |—— ...
      |—— labels
        |—— 00000.png
        |—— 00005.png
        |—— ...
    |—— val
      |—— images
      |—— flows
      |—— labels    
  |—— DAVIS-2016
    |—— train
      |—— images
      |—— flows
      |—— labels    
    |—— val
      |—— images
      |—— flows
      |—— labels
  |—— Youtube-objects  
    |—— val
      |—— images
      |—— flows
  |—— FBMS
    |—— train
      |—— images
      |—— flows
      |—— labels    
    |—— val
      |—— images
      |—— flows
      |—— labels    
```

### Prepare pretrained backbone
The pre-trained backbone can be downloaded from [MobileViT xxs backbone](https://anonymfile.com/oebzD/mobilevit-xxspth.tar) and [MobileViT s backbone](https://anonymfile.com/Epe4D/mobilevit-spth.tar) and put it into the ```pretrained``` folder.

### Train
- First, train the model using the YouTubeVOS-2018, DAVIS-2016 datasets as labeled data and Youtube-objects as unlabeled data.
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 unlabeled_train.py --epochs 30 --data_dir /your/data/path --pretrained /your/pretrained/path --sync-bn
```
- Second, finetune the model using the DAVIS-2016 dataset as labeled data and Youtube-objects as unlabeled data.
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 unlabeled_train.py --epochs 25 --data_dir /your/data/path --labeled_datasets DAVIS-2016 --pretrained /your/pretrained/path --finetune /your/first_stage/path --sync-bn
```

### Online Finetuning
- Run following to start online finetuning.
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 online_finetuning.py --epochs 10 --data_dir /your/data/path --pretrained /your/pretrained/path --weight /your/trained/weight --sync-bn
```

## Test

-   We can produce segmentation results in ```test.py```.
```
python test.py --data_dir /your/data/path --save_dir /save/reuslts/path --weights /your/final/weight/path
```

## Final weight

- The final weight can be downloaded from [OAST](https://anonymfile.com/Rd49K/online-finetuningbest.pth).

## Evaluation Metrics

- We use the standard UVOS evaluation toolbox from [DAVIS-2016 benchmark](https://github.com/davisvideochallenge/davis-matlab/tree/davis-2016) and VSOD evaluation toolbox from [DAVSOD benchmark](https://github.com/DengPingFan/DAVSOD). Note that the two toolboxes are from official repositories. 
