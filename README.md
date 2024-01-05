<div align="center">
  <img width=500 src="./assest/logo.png" alt="logo" />
</div>

UFCD is a Pytorch-based toolbox for three different change detection tasks, including binary change detection (BCD), semantic change detection (SCD), and building damage assessment (BDA).

## 🍓 Quick Start

✈️ `Step 1`: Prerequisites for Python:
- Create a virtual environment in the terminal:
```shell
conda create -n SCD python=3.8
```
- Install necessary packages:
```shell
pip install -r requirements.txt
```

✈️ `Step 2`: Prepare the semantic change detection datasets:
- Download datasets [SECOND](https://ieeexplore.ieee.org/abstract/document/9555824), and [Landsat-SCD](https://figshare.com/articles/figure/Landsat-SCD_dataset_zip/19946135/1)
- The pre-processed datasets can be obtained from [SECOND](http://www.captain-whu.com/PROJECT/SCD/), and the train and test list can be  downloaded from [list](https://github.com/ggsDing/Bi-SRNet/tree/main/datasets).
- The pre-processed Landsat-SCD dataset can be obtained from [Landsat-SCD](https://drive.google.com/file/d/11CkLhakNtfaBH78SGTHxcXKNsBM524H5/view).
- Prepare datasets into the following structure and set their path in `./lib/configs/data_congfig.py`
```
├─train
    ├─im1        ...jpg/png
    ├─im2        ...jpg/png
    ├─label1    ...jpg/png
    ├─label2    ...jpg/png
    └─list     ...txt
├─test
    ├─im1        
    ├─im2        
    ├─label1    
    ├─label2   
    └─list     
```
✈️ `Step 3`:Prepare the building damage assessment dataset:
- Download dataset [xView2](https://openaccess.thecvf.com/content_CVPRW_2019/papers/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.pdf).
- Create the rbg mask from xView2 by using './data/xBD/preoricess_data.py'.
- Create the image list from xView2 by using './data/xBD/capture_image_list.py'.
- Prepare dataset into the following structure and set its path in `./lib/configs/data_congfig.py`
```
├─train
    ├─images        ...jpg/png
    ├─labels        ...json
    ├─masks    ...jpg/png
    ├─targets    ...jpg/png
    ├─image_list.txt     
├─test
    ├─images        ...jpg/png
    ├─labels        ...json
    ├─masks    ...jpg/png
    ├─targets    ...jpg/png
    ├─image_list.txt        
 ```
✈️ `Step 4`:Train/Test:
```shell
sh ./scripts/train.sh  
sh ./scripts/test.sh   
 ```
## 🍓 Currently Supported Models and Datasets

Supported semantic change detection models:
- [x] [A2Net](https://ieeexplore.ieee.org/abstract/document/10034814)
- [x] [SCanNet](https://arxiv.org/abs/2212.05245)
- [x] [TED](https://arxiv.org/abs/2212.05245)
- [x] [BiSRNet](https://ieeexplore.ieee.org/document/9721305)
- [x] [SSCDL](https://ieeexplore.ieee.org/document/9721305)

Supported building damage assessment model:
- [x] [ChangeOS](https://www.sciencedirect.com/science/article/pii/S0034425721003564)

Supported semantic change detection datasets:
- [x] [SECOND](https://ieeexplore.ieee.org/abstract/document/9555824)
- [x] [Landsat-SCD](https://figshare.com/articles/figure/Landsat-SCD_dataset_zip/19946135/1)

Supported building damage assessment dataset:
- [x] [xView2](https://openaccess.thecvf.com/content_CVPRW_2019/papers/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.pdf)



## 🥤 Acknowledgment
This repository is built with the help of the projects [SCanNet](https://github.com/ggsDing/SCanNet), 
and [SRSCDF](https://github.com/walking-shadow/Simple-Remote-Sensing-Change-Detection-Framework) for academic use only.

## 🍎 Ending
If this repository  is helpful to you, please remember to Star~😘.
