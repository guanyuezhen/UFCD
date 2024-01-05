
##  Prepare the datasets

✈️ `1`: Prepare the binary change detection datasets:
- Download datasets [LEVIR](https://justchenhao.github.io/LEVIR/) and [SYSU](https://github.com/liumency/SYSU-CD)
- Crop LEVIR and BCDD datasets into 256x256 patches. 
- Generate list file as `ls -R ./label/* > test.txt`
- Prepare datasets into the following structure and set their path in `./lib/configs/data_congfig.py`
    ```
    ├─train
        ├─A        ...jpg/png
        ├─B        ...jpg/png
        ├─label    ...jpg/png
        └─list     ...txt
    ├─val
        ├─A
        ├─B
        ├─label
        └─list
    ├─test
        ├─A
        ├─B
        ├─label
        └─list
    ```


✈️ `2`: Prepare the semantic change detection datasets:
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
✈️ `3`:Prepare the building damage assessment dataset:
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
