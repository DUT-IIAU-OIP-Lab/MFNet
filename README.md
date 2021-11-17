# MFNet
Source code for "**MFNet: Multi-filter Directive Network for Weakly Supervised Salient Object Detection**", accepted in ICCV-2021, poster.

[Yongri Piao](http://ice.dlut.edu.cn/yrpiao/), Jian Wang, Miao Zhang and [Huchuan Lu](http://ice.dlut.edu.cn/lu/publications.html).  IIAU-OIP Lab.

![image](https://github.com/DUTyimmy/MFNet/blob/main/fig/overall%20framework.png)

## Prerequisites
### environment
  - Windows 10
  - Torch 1.8.1
  - CUDA 10.0
  - Python 3.7.4
  - other environment requirment can be found in requirments.txt 

### training data
link: https://pan.baidu.com/s/1omTCChQFWwNFhQ79AVD8rg.    code: oipw

### testing datasets
link: https://pan.baidu.com/s/1PBzDP1Hnf3RIvpARmxn2yA.    code: oipw

## Training
### 1st training stage
Case1: please refer to [this repository](https://github.com/DUTyimmy/generatePGT).

Case2: We also upload ready-made pseudo labels in **Training data** (the link above), you can directly use our offered two kinds of pseudo labels for convenience. CAMs are also presented if you needed.

### 2nd training stage

#### 1, setting the training data to the proper root as follows:

```
MF_code -- data -- DUTS-Train -- image -- 10553 samples

                -- ECSSD (not necessary) 
                
                -- pseudo labels -- label0_0 -- 10553 pseudo labels
                
                                 -- label1_0 -- 10553 pseudo labels
```
#### 2, training
```Run main.py```

Here you can set ECCSD dataset as validation set for optimal results by setting ```--val``` to ```True```, of course it is not necessary in our work.

## Testing
```Run test_code.py```

You need to configure your desired testset in ```--test_root```.  Here you can also perform PAMR and CRF on saliency maps for a furthur refinements if you want, by setting ```--pamr``` and ```--crf``` to True. **Noting** that the results in our paper do not adopt these post-process for a fair comparison.

The evaluation code can be found in [here](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox).

## Saliency maps & Checkpoint
We offer our saliency maps and checkpoints on various backbones (including DenseNet-169, ResNet-101, ResNet-50 and VGG-16) for more convenient comparison in the future. The results in our paper are all come from the model based on DenseNet-169, and we also highly recommend the following researchers adopt same backbone for a more fair and convenient comparison.
### Saliency maps
link: https://pan.baidu.com/s/1IRTEaEicYaCJ2TYjZV1lZA.    code: oipw
### Checkpoints
link: https://pan.baidu.com/s/1lQm-MY4uwZZTW5NJTTXeyA.    code: oipw

## Contact me
If you have any questions, please contact me: [jiangnanyimi@163.com].

## Acknowledge
Thanks to pioneering helpful works:

  - [MSW](https://github.com/zengxianyu/mws/tree/new):  Multi-source weak supervision for saliency detection, CVPR2019, by Yu Zeng et al.
  - [SSSS](https://github.com/visinf/1-stage-wseg):  Single-stage Semantic Segmentation from Image Labels, CVPR2020, by Nikita Araslanov et al.
