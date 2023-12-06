# A2Net-SCD
This repository contains a simple Python implementation of semantic change detection in remote sensing images based on [A2Net](https://ieeexplore.ieee.org/abstract/document/10034814).


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