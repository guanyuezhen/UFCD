# SCD

## 1. Introduction
Here is a framework for semantic change detection based on Pytorch.

Supported semantic change detection models:
- [x] [A2Net](https://ieeexplore.ieee.org/abstract/document/10034814)
- [x] [SCanNet](https://arxiv.org/abs/2212.05245)
- [x] [TED](https://arxiv.org/abs/2212.05245)
- [x] [BiSRNet](https://ieeexplore.ieee.org/document/9721305)
- [x] [SSCDL](https://ieeexplore.ieee.org/document/9721305)

Supported semantic change detection datasets:
- [x] [SECOND](https://ieeexplore.ieee.org/abstract/document/9555824)
- [x] [Landsat-SCD](https://figshare.com/articles/figure/Landsat-SCD_dataset_zip/19946135/1)

## 2. Usage
+ Prepare the data:
    - Download datasets [SECOND](https://ieeexplore.ieee.org/abstract/document/9555824), and [Landsat-SCD](https://figshare.com/articles/figure/Landsat-SCD_dataset_zip/19946135/1)
    - The pre-processed datasets can be obtained from [SECOND](http://www.captain-whu.com/PROJECT/SCD/), and the train and test list can be  downloaded from [list](https://github.com/ggsDing/Bi-SRNet/tree/main/datasets).
    - The pre-processed Landsat-SCD dataset can be obtained from [Landsat-SCD](https://drive.google.com/file/d/11CkLhakNtfaBH78SGTHxcXKNsBM524H5/view).
    - Prepare datasets into the following structure and set their path in `train.py` and `test.py`
    ```
    ├─Train
        ├─im1        ...jpg/png
        ├─im2        ...jpg/png
        ├─label1    ...jpg/png
        ├─label2    ...jpg/png
        └─list     ...txt
    ├─Test
        ├─im1        
        ├─im2        
        ├─label1    
        ├─label2   
        └─list     
    ```

+ Prerequisites for Python:
    - Creating a virtual environment in the terminal: `conda create -n SCD python=3.8`
    - Installing necessary packages: `pip install -r requirements.txt`

+ Train/Test
    - `sh train.sh`
    - `sh test.sh`

### 3. Change Detection Results on SECOND Dataset
<p align="center">
    <img src="assest/benchmark_results.png"/> <br />
    <em> 
    The best and second best results are highlighted in <font color="#FF0000">red</font> and <font color="#00B0F0">blue</font>, respectively.
    </em>
</p>

### 4. Acknowledgment
This repository is built with the help of the projects [BIT_CD](https://github.com/justchenhao/BIT_CD), 
[SCanNet](https://github.com/ggsDing/SCanNet), 
and [SRSCDF](https://github.com/walking-shadow/Simple-Remote-Sensing-Change-Detection-Framework) for academic use only.

### 5. Citation

Please cite our paper if you find the work useful:

    @article{Li_2023_A2Net,
         author={Li, Zhenglai and Tang, Chang and Liu, Xinwang and Zhang, Wei and Dou, Jie and Wang, Lizhe and Zomaya, Albert Y.},
        journal={IEEE Transactions on Geoscience and Remote Sensing}, 
        title={Lightweight Remote Sensing Change Detection With Progressive Feature Aggregation and Supervised Attention}, 
        year={2023},
        volume={61},
        number={},
        pages={1-12},
        doi={10.1109/TGRS.2023.3241436}
        }