# Decoupled Weight Decay Regularization (old title: Fixing Weight Decay Regularization in Adam)

This repository contains the code for the paper Decoupled Weight Decay Regularization (old title: Fixing Weight Decay Regularization in Adam) by Ilya Loshchilov and Frank Hutter, ICLR 2019 [arXiv](https://arxiv.org/abs/1711.05101). 

The code represents a tiny modification of the source code provided for the Shake-Shake regularization by Xavier Gastaldi [arXiv](https://arxiv.org/abs/1705.07485). Since the usage of both is very similar, the introduction and description of the original Shake-Shake code is given below. Please consider to  *first* run the Shake-Shake code and then our code. 

Find below a few examples to train a 26 2x96d "Shake-Shake-Image" ResNet on CIFAR-10 with 1 GPU.
To run on 4 GPUs, set `CUDA_VISIBLE_DEVICES=0,1,2,3` and `-nGPU 4`.
For test purposes you may reduce `-nEpochs` from 1500 to e.g. 150 and set `-widenFactor` to 4 to use a smaller network. 
To run on ImageNet32x32, set `-dataset` to imagenet32 and reduce `-nEpochs` to 150.
You may consider to use `-weightDecay=0.05` for CIFAR-10. 

Importantly, please copy with replacement `adam.lua` and `sgd.lua` from `UPDATETORCHFILES` to `YOURTORCHFOLDER/install/share/lua/5.1/optim/`

To run AdamW for `nEpochs=1500` epochs without restarts with initial learning rate `LR=0.001`, normalized weight decay `weightDecay=0.025`   

```
CUDA_VISIBLE_DEVICES=0 th main.lua -algorithmType ADAMW -nEpochs 1500 -Te 1500 -Tmult 2 -widenFactor 6 -LR 0.001 -weightDecay 0.025 -dataset cifar10 -nGPU 1 -depth 26 -irun 1 -batchSize 128 -momentum 0.9 -shareGradInput false -optnet true -netType shakeshake -forwardShake true -backwardShake true -shakeImage true -lrShape cosine -LRdec true
```

To run AdamW for `nEpochs=1500` epochs with restarts, where the first restart will happen after `Te=100` epochs and the second restart after 200 more epochs because `100*Tmult=200`. 

```
CUDA_VISIBLE_DEVICES=0 th main.lua -algorithmType ADAMW -nEpochs 1500 -Te 100 -Tmult 2 -widenFactor 6 -LR 0.001 -weightDecay 0.025 -dataset cifar10 -nGPU 1 -depth 26 -irun 1 -batchSize 128 -momentum 0.9 -shareGradInput false -optnet true -netType shakeshake -forwardShake true -backwardShake true -shakeImage true -lrShape cosine -LRdec true
```

To run SGDW for `nEpochs=150` epochs without restarts with initial learning rate `LR=0.05`, normalized weight decay `weightDecay=0.025`   

```
CUDA_VISIBLE_DEVICES=0 th main.lua -algorithmType SGDW -nEpochs 1500 -Te 1500 -Tmult 2 -widenFactor 6 -LR 0.05 -weightDecay 0.025 -dataset cifar10 -nGPU 1 -depth 26 -irun 1 -batchSize 128 -momentum 0.9 -shareGradInput false -optnet true -netType shakeshake -forwardShake true -backwardShake true -shakeImage true -lrShape cosine -LRdec true
```

To run SGDW for `nEpochs=150` epochs with restarts, where the first restart will happen after `Te=100` epochs and the second restart after 200 more epochs because `100*Tmult=200`. 

```
CUDA_VISIBLE_DEVICES=0 th main.lua -algorithmType SGDW -nEpochs 1500 -Te 100 -Tmult 2 -widenFactor 6 -LR 0.001 -weightDecay 0.025 -dataset cifar10 -nGPU 1 -depth 26 -irun 1 -batchSize 128 -momentum 0.9 -shareGradInput false -optnet true -netType shakeshake -forwardShake true -backwardShake true -shakeImage true -lrShape cosine -LRdec true
```

Acknowledgments: We thank Patryk Chrabaszcz for creating functions dealing with ImageNet32x32 dataset.


# Shake-Shake regularization of 3-branch residual networks

This repository contains the code for the paper [Shake-Shake regularization of 3-branch residual networks](https://openreview.net/forum?id=HkO-PCmYl&noteId=HkO-PCmYl). 

The code is based on [fb.resnet.torch] (https://github.com/facebook/fb.resnet.torch).

## Table of Contents
0. [Introduction](#introduction)
0. [Results](#results)
0. [Usage](#usage)
0. [Contact](#contact)

## Introduction
This method aims at helping computer vision practitioners faced with an overfit problem. The idea is to replace, in a 3-branch ResNet, the standard summation of residual branches by a stochastic affine combination. The largest tested model improves on the best single shot published result on CIFAR-10 by reaching 2.72% test error.

![shake-shake](https://s3.eu-central-1.amazonaws.com/github-xg/architecture3.png)

Figure 1: **Left:** Forward training pass. **Center:** Backward training pass. **Right:** At test time.

## Results
The base network is a 26 2x32d ResNet (i.e. the network has a depth of 26, 2 residual branches and the first residual block has a width of 32). "Shake" means that all scaling coefficients are overwritten with new random numbers before the pass. "Even" means that all scaling coefficients are set to 0.5 before the pass. "Keep" means that we keep, for the backward pass, the scaling coefficients used during the forward pass. "Batch" means that, for each residual block, we apply the same scaling coefficient for all the images in the mini-batch. "Image" means that, for each residual block, we apply a different scaling coefficient for each image in the mini-batch. The numbers in the Table below represent the average of 3 runs except for the 96d models which were run 5 times.

Forward | Backward | Level | 26 2x32d | 26 2x64d | 26 2x96d 
-------|:-------:|:--------:|:--------:|:--------:|:--------:|
Even	|Even	|n\a	|4.13	|3.64	|3.44
Even	|Shake	|Batch	|4.34	|-	|-
Shake	|Keep	|Batch	|3.98	|-	|-
Shake	|Even	|Batch	|3.40	|3.24	|-
Shake	|Shake	|Batch	|3.54	|3.01	|-
Even	|Shake	|Image	|tbd	|-	|-
Shake	|Keep	|Image	|4.07	|-	|-
Shake	|Even	|Image	|tbd	|tbd	|-
Shake	|Shake	|Image 	|3.48	|2.86	|**2.72**

Table 1: Error rates (%) on CIFAR-10

## Usage 
0. Install [fb.resnet.torch] (https://github.com/facebook/fb.resnet.torch), [optnet](https://github.com/fmassa/optimize-net) and [lua-stdlib](https://github.com/lua-stdlib/lua-stdlib).
1. Download Shake-Shake
```
git clone https://github.com/xgastaldi/shake-shake.git
```
2. Copy the elements in the shake-shake folder and paste them in the fb.resnet.torch folder. This will overwrite 5 files (*main.lua*, *train.lua*, *opts.lua*, *checkpoints.lua* and *models/init.lua*) and add 3 new files (*models/shakeshake.lua*, *models/shakeshakeblock.lua* and *models/mulconstantslices.lua*).
3. You can train a 26 2x32d "Shake-Shake-Image" ResNet on CIFAR-10+ using

```
th main.lua -dataset cifar10 -nGPU 1 -batchSize 128 -depth 26 -shareGradInput false -optnet true -nEpochs 1800 -netType shakeshake -lrShape cosine -widenFactor 2 -LR 0.2 -forwardShake true -backwardShake true -shakeImage true
``` 

You can train a 26 2x96d "Shake-Shake-Image" ResNet on 2 GPUs using

```
CUDA_VISIBLE_DEVICES=0,1 th main.lua -dataset cifar10 -nGPU 2 -batchSize 128 -depth 26 -shareGradInput false -optnet true -nEpochs 1800 -netType shakeshake -lrShape cosine -widenFactor 6 -LR 0.2 -forwardShake true -backwardShake true -shakeImage true
```

A widenFactor of 2 corresponds to 32d, 4 to 64d, etc..

### Note
Changes made to fb.resnet.torch files:

*main.lua*  
Ln 17, 54-59, 81-88: Adds a log (courtesy of Sergey Zagoruyko)  

*train.lua*  
Ln 36-38 58-60 206-213: Adds the cosine learning rate function  
Ln 88-89: Adds the learning rate to the elements printed on screen  

*opts.lua*  
Ln 57-62: Adds Shake-Shake options  

*checkpoints.lua*  
Ln 15-16: Adds require 'models/shakeshakeblock' and require 'std'  
Ln 60-61: Avoids using the fb.renet.torch deepcopy (it doesn't seem to be compatible with the BN in shakeshakeblock) and replaces it with the deepcopy from stdlib  
Ln 67-81: Saves only the best model  

*init.lua*  
Ln 91-92: Adds require 'models/mulconstantslices' and require 'models/shakeshakeblock'  

The main model is in *shakeshake.lua*. The residual block model is in *shakeshakeblock.lua*. *mulconstantslices.lua* is just an extension of nn.mulconstant that multiplies elements of a vector with image slices of a mini-batch tensor.

## Contact
xgastaldi.mba2011 at london.edu  
Any discussions, suggestions and questions are welcome!

