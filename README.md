# Scaled YOLO v4

This repo exaplins how to train [Scaled_YOLOv4](https://arxiv.org/abs/2011.08036) model on your custom dataset. 

### About Scaled YOlO v4
The YOLOv4 object detection neural network based on the CSP approach, scales both up and down and is applicable to small and large networks while maintaining optimal speed and accuracy. We propose a network scaling approach that modifies not only the depth, width, resolution, but also structure of the network. YOLOv4-large model achieves state-of-the-art results: 55.5% AP (73.4% AP50) for the MS COCO dataset at a speed of ~16 FPS on Tesla V100, while with the test time augmentation, YOLOv4-large achieves 56.0% AP (73.3 AP50). To the best of our knowledge, this is currently the highest accuracy on the COCO dataset among any published work. The YOLOv4-tiny model achieves 22.0% AP (42.0% AP50) at a speed of 443 FPS on RTX 2080Ti, while by using TensorRT, batch size = 4 and FP16-precision the YOLOv4-tiny achieves 1774 FPS.
### Model Architecture
![alt text](https://github.com/Mr-TalhaIlyas/Scaled-YOLOv4/blob/master/screens/image-2.png)
### Comparison with other models
![alt text](https://github.com/Mr-TalhaIlyas/Scaled-YOLOv4/blob/master/screens/image.png)
## Dependencies

Some of the main requirements are
```
pytorch
mish-cuda
```

## Roboflow

I'll be using the [Roboflow Scaled YOLOv4](https://github.com/roboflow-ai/ScaledYOLOv4) repo in this tutorial, and go through the steps of how 

1. Prepare your data 
2. Convert data format from PASCAl_VOC to YOLO PyTorch format
3. Installing repo
4. Installing/Troublshooting `mish-cuda`.

Original [Colab Notebook](https://colab.research.google.com/drive/1LDmg0JRiC2N7_tx8wQoBzTB0jUZhywQr#scrollTo=odKEqYtTgbRc)

## Dataset Preparation

First to train an object detection model you need a dataset annotated in proper format so download publically available datasets from [here](https://public.roboflow.com/).
I'd recommend starting by downloading already available dataset. There are alot of format options available in Roboflow but for this repo we need `YOLO v5 PyTorch` as this 
Scaled YOLO v4 repo is also built on top on [YOLOv5](https://github.com/Mr-TalhaIlyas/YOLO-v5) one.

or you can also make you own dataset using `labelimg`. A full tutorial for that is [here](https://github.com/tzutalin/labelImg)
The ouput annotation file for label me is `.xml` format but our yolov4 model can't read that so we need to convert the dataset into proper format

## Dataset Format Conversion
 After labelling the data put both img files and `.xml` files in the same dir.
and run the `voc2yolov5.py` file from scripts

In the first few lines of the `.py` file change following lines accordingly

```python

os.chdir('D:/cw_projects/paprika/paprika_processed/data_final/') # main dir which contains following subdirectories
dirs = ['train', 'test', 'val']

classes = [ "car", "bike","dog","person"] # put your class names
```
After that make your data dir like following

```
📦data
 ┣ 📂test
 ┃ ┣ 📂images
 ┃ ┗ 📂labels
 ┣ 📂train
 ┃ ┣ 📂images
 ┃ ┗ 📂labels
 ┣ 📂valid
 ┃ ┣ 📂images
 ┃ ┗ 📂labels
 ┗ 📜data.yaml
```
the `images` will have all the `.jpg` or `.png` files and `labels` will have `.txt` files for each image and inside `data.yaml` file add following information

```
train: /path2dir/data/train/images
val: /path2dir/data/valid/images

nc: 4
names: [ "car", "bike","dog","person"]
```

## Installation
Getting started wiht the installation make a new conda `env`

```
conda create -n yolo python=3.6.7
```
activate `env` and `cd` to a `dir` where you will keep all your `data` and `scripts`.

### Scaled YOLO v4
Now form inside the `scirpts` dir copy the `jupyter notebooks` and place in the same `dir` you choose first.

Now run the notebook `setup_ScaledYOLOv4.ipynb` sequentelly
Place your data inside the `dir` you choose
Now run the `Scaled_YOLOv4.ipynd` follow steps inside notebook

### mish-cuda

while isntalling mish-cuda you might face following problems

1. `cudalas_v2.h` not found
2. `cudalas_api.h1 not found

So you can solve it as follows

1. first check if you server have this file or not by typing

```
find /usr/local/ -name cublas_v2.h 
# or depending on which file is missing
find /usr/local/ -name cublas_api.h 
```
you will see the dir or cuda versions which have these files so e.g. in my case my server has 3 differec cuda version running and `mish-cuda` needed 10.1 version which was
missing file so I got following output

```
/usr/local/cuda-10.2/targets/x86_64-linux/include/cublas_api.h
/usr/local/cuda-10.0/include/cublas_api.h
```
so you can add this file to `cuda-10.1` dir by either creating a `symblic link` or simply copying the files. I copied the files as follows

```
# export PATH=/usr/local/cuda-10.0/bin:$PATH 
# copy files
sudo cp /usr/local/cuda-10.0/include/cublas_api.h /usr/local/cuda-10.1/targets/x86_64-linux/include/
# or crate symblic link
ln -s /usr/include/cublas_v2.h /usr/local/cuda-10.1/targets/x86_64-linux/include/cublas_v2.h
```

## Evaluation

For in depth evaluation you can run the `my_inference.py` file form the scripts.

