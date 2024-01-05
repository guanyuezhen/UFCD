<div align="center">
  <img width=500 src="./assest/logo.png" alt="logo" />
</div>

UFCD is a Pytorch-based toolbox for three different change detection tasks, including binary change detection (BCD), semantic change detection (SCD), and building damage assessment (BDA).

## 🍓 Quick Start

✈️ `Step 1`: Prerequisites for Python:
- Create a virtual environment in the terminal:
```shell
conda create -n UFCD python=3.8
```
- Install necessary packages:
```shell
pip install -r requirements.txt
```

✈️ `Step 2`: Prepare the change detection datasets following [./data/README.md](./data/README.md).

✈️ `Step 3`:Train/Test:
```shell
sh ./scripts/train.sh  
sh ./scripts/test.sh   
 ```
## 🍓 Currently Supported Models and Datasets

Supported binary change detection models:
| No. | Model  | Paper                                                                                                     | link                                                            |
| --- | ------ | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 1   | A2Net  | Lightweight Remote Sensing Change Detection With Progressive Feature Aggregation and Supervised Attention | [link](https://ieeexplore.ieee.org/abstract/document/10034814/) |
| 2   | TFI-GR | Remote Sensing Change Detection via Temporal Feature Interaction and Guided Refinement                    | [link](https://ieeexplore.ieee.org/abstract/document/9863802)   |

Supported semantic change detection models:
| No. | Model  | Paper                                                                                                     | link                                                            |
| --- | ------ | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 1   | A2Net  | Lightweight Remote Sensing Change Detection With Progressive Feature Aggregation and Supervised Attention | [link](https://ieeexplore.ieee.org/abstract/document/10034814/) |
| 2   | SCanNet/TED | Joint Spatio-Temporal Modeling for the Semantic Change Detection in Remote Sensing Images            | [link](https://arxiv.org/abs/2212.05245)                        |
| 2   | BiSRNet/SSCDL | Bi-Temporal Semantic Reasoning for the Semantic Change Detection in HR Remote Sensing Images       | [link](https://ieeexplore.ieee.org/document/9721305)            |

Supported building damage assessment model:
Supported semantic change detection models:
| No. | Model  | Paper                                                                                                     | link                                                            |
| --- | ------ | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 1   | ChangeOS  | Building Damage Assessment for Rapid Disaster Response with a Deep Object-based Semantic Change Detection Framework: From Natural Disasters to Man-made Disasters | [link](https://www.sciencedirect.com/science/article/pii/S0034425721003564) |
| 2   | ChangeOS-GRM | -           |-                       |

Supported binary change detection datasets:
| No. | Dataset  | -                                                                                                     | link                                                            |
| --- | ------ | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 1   | LEVIR/LEVIR+  | - | [link](https://justchenhao.github.io/LEVIR/) |
| 2   | TFI-GR | -                   | [link](https://ieeexplore.ieee.org/abstract/document/9863802)   |

Supported semantic change detection datasets:
| No. | Dataset  | -                                                                                                     | link                                                            |
| --- | ------ | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 1   | SECOND  | - | [link](https://ieeexplore.ieee.org/abstract/document/9555824) |
| 2   | Landsat-SCD | -                   | [link](https://figshare.com/articles/figure/Landsat-SCD_dataset_zip/19946135/1)   |

Supported building damage assessment dataset:
| No. | Dataset  | -                                                                                                     | link                                                            |
| --- | ------ | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 1   | xView2  | - | [link](https://xview2.org/) |


## 🥤 Acknowledgment
This repository is built with the help of the projects [BIT_CD](https://github.com/justchenhao/BIT_CD), [PytorchDeepLearing](https://github.com/junqiangchen/PytorchDeepLearing), [SCanNet](https://github.com/ggsDing/SCanNet), 
and [SRSCDF](https://github.com/walking-shadow/Simple-Remote-Sensing-Change-Detection-Framework) for academic use only.

## 🍎 Ending
If this repository  is helpful to you, please remember to Star~😘.
