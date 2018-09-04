# caffe-faster-rcnn has been deprecated. Please see [Detectron](https://github.com/facebookresearch/Detectron), which includes an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870).

**Special Feature for This Caffe Repository**

- Clone from the official [Caffe](https://github.com/BVLC/caffe), will continuely be up-to-date by the official caffe code
- Faster rcnn joint train, test and evaluate
- Action recognition (Two Stream CNN)

## [Faster R-CNN](https://arxiv.org/abs/1506.01497)

### Disclaimer
The official Faster R-CNN code (written in MATLAB) is [available](https://github.com/ShaoqingRen/faster_rcnn) here. If your goal is to reproduce the results in our NIPS 2015 paper, please use the [official](https://github.com/ShaoqingRen/faster_rcnn) code.

This repository contains a C++ reimplementation of the Python code([py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)). This C++ implementation is built on the offcial [caffe](https://github.com/BVLC/caffe), I will continue to update this code for improvement and up-to-date by offcial caffe.

All following steps, you should do these in the `$CAFFE_ROOT` path.

### Demo
Using `sh example/FRCNN/demo_frcnn.sh`, the will process five pictures in the `examples/FRCNN/images`, and put results into `examples/FRCNN/results`.

Note: You should prepare the trained caffemodel into `models/FRCNN`, such as `ZF_faster_rcnn_final.caffemodel` for ZF model.

### Prepare for training and evaluation
- The list of training data is `examples/FRCNN/dataset/voc2007.trainval`.
- The list of testing data is `examples/FRCNN/dataset/voc2007.test`.
- Create symlinks for the PASCAL VOC dataset `ln -s $YOUR_VOCdevkit_Path $CAFFE_ROOT/VOCdevkit`.

As shown in VGG example `models/FRCNN/vgg16/train_val.proto`, the original pictures should appear at `$CAFFE_ROOT/VOCdevkit/VOC2007/JPEGImages/`. (Check window\_data\_param in FrcnnRoiData)

If you want to train Faster R-CNN on your own dataset, you may prepare custom dataset list.
The format is as below
```
# image-id
image-name
number of boxes
label x1 y1 x2 y2 difficulty
...
```

- The ImageNet pre-trained models can be found in [this link](https://drive.google.com/drive/folders/1xjFL-ZeVzXkY584ZsEnr9O6O3P1Ypjwd?usp=sharing)

### Training
`sh examples/FRCNN/zf/train_frcnn.sh` will start training process of voc2007 data using ZF model.

If you use the provided training script, please make sure:
- VOCdevkit is within $CAFFE\_ROOT and VOC2007 in within VOCdevkit
- ZF pretrain model should be put into models/FRCNN/ as ZF.v2.caffemodel

`examples/FRCNN/convert_model.py` transform the parameters of `bbox_pred` layer by mean and stds values,
because the regression value is normalized during training and we should recover it to obtain the final model.

### Evaluation
`sh examples/FRCNN/zf/test_frcnn.sh` the will evaluate the performance of voc2007 test data using the trained ZF model.

- First Step of This Shell : Test all voc-2007-test images and output results in a text file.
- Second Step of This Shell : Compare the results with the ground truth file and calculate the mAP.

### Detail

Shells and prototxts for different models are listed in the `examples/FRCNN` and `models/FRCNN`

More details in the code:
- `include/api/FRCNN` and `src/api/FRCNN` for demo and test api
- `include/caffe/FRCNN` and `src/caffe/FRCNN` contains all codes related to Faster R-CNN

### Commands, Rebase From Caffe Master

**For synchronous with official caffe**

- git remote add caffe https://github.com/BVLC/caffe.git
- git fetch caffe
- git checkout master
- git rebase caffe/master

**Rebase the dev branch**
- git checkout dev
- git rebase master 
- git push -f origin dev

## QA
- CUB not found, when compile for GPU version, `frcnn_proposal_layer.cu` requires a head file `<cub/cub.cuh>`. CUB is library contained in the official Cuda Toolkit, usually can be found in ` /usr/local/cuda/include/thrust/system/cuda/detail/`. You should add this path in your `Makefile.config` (try `locate ''cub.cuh''` to find cub on your system)
- When Get `error: RPC failed; result=22, HTTP code = 0`, use `git config http.postBuffer 524288000`, increases git buffer to 500mb

## Two-Stream Convolutional Networks for Action Recognition in Video

See codes `src/caffe/ACTION_REC` and `include/caffe/ACTION_REC`

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
    @inproceedings{girshick2015fast,
      title={Fast R-CNN},
      author={Girshick, Ross},
      booktitle={International Conference on Computer Vision},
      pages={1440--1448},
      year={2015}
    }
    @inproceedings{ren2015faster,
      title={Faster {R-CNN}: Towards real-time object detection with region proposal networks},
      author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
      booktitle={Neural Information Processing Systems},
      pages={91--99},
      year={2015}
    }
    @article{ren2017faster,
      title={Faster {R-CNN}: Towards real-time object detection with region proposal networks},
      author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      volume={39},
      number={6},
      pages={1137--1149},
      year={2017},
      publisher={IEEE}
    }
