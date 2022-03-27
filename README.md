# ssd4largeimage

This repository shares scripts for performing object detection on large images.
Object detection uses an implementation called SSD512 based on deep learning.
Since the SSD512 input image is resized to 512x512 pixels before inference is performed, important information may be lost in the case of large images.

The following three types of scripts are mainly provided here:
* Cropping of large images and their labeling information in dieferent size ratio
* Training SSD512
* Making inference with large images by cropping them in dieferent size ratio


## Run scripts
* The deep learning framework is Chainer and ChainerCV.
* The labeling required in the training data assumes the use of labelImg.
* The sample script assumes an input image of 8001 x 8001 pixels.

### 1. Data propressing: Separate images in different ratios.
```
python separate_image.py --data-dir <path to directory that contains image data and xml(labeling info) data> --image-format jpg --output-dir <path to directory of cropped data that for training phase> --window-size 500,1000,2000,4000,8000 --margin 50,100,200,400,800 --thresh 0.05
```

### 2. Training SSD512 
``` 
python train.py --train <path to directory of training data>  --val <path to directory of validation data> --iteration 40000 --val-iteration 200 --lr 1e-4 --gpu 0 --output-dir <path to directory of trained model>
```


### 3. Make inference of large images
```
mkdir <name of output directory>
python inference_image_with_multi-window.py --data-dir <path to directory of inference data> --output-dir <path to directory of inference data> --margin 50,100,200,400,800 --window-size 500,1000,2000,4000,8000 --model  <path to directory of trained model>/<name of model file> --image-format jpg --label-names label.yml  --gpu 0  --batch-size 10  --thresh 0.7
```
