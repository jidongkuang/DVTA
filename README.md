# DVTA
This is an official PyTorch implementation of **"Zero-Shot Skeleton-based Action Recognition with Dual Visual-Text Alignment"**.

# Framework
<!--![SMIE](https://github.com/YujieOuO/SMIE/blob/main/images/pipeline.png)-->

## Requirements
![python = 3.9](https://img.shields.io/badge/python-3.9.18-green)
![torch = 2.1.1+cu118](https://img.shields.io/badge/torch-2.1.1%2Bcu118-yellowgreen)

## Installation
```bash
# Install the python libraries
$ cd DVTA
$ pip install -r requirements.txt
```

## Data Preparation
We apply the same dataset processing as [SMIE](https://github.com/YujieOuO/SMIE).  
<!--!
### Semantic Features
For the Semantic Features, You can download in BaiduYun link: [Semantic Feature](https://pan.baidu.com/s/1y2r15lxGF3i9aPa1ARfRiQ).

The code: smie
* [dataset]_embeddings.npy: based on label names using Sentence-Bert.
* [dataset]_clip_embeddings.npy: based on label names using CLIP.
* [dataset]_des_embeddings.npy: based on label descriptions using Sentence-Bert.

Put the semantic feautures in fold: ./data/language/

### Label Descriptions
Using [ChatGPT](https://chat.openai.com/) to expand each action label name into a complete action description.
The total label descriptions can be found in [folder](https://github.com/YujieOuO/SMIE/tree/main/descriptions).

## Different Experiment Settings
Our DVTA employs two experiment setting.
* SynSE Experiment Setting: two datasets are used, split_5 and split_12 on NTU60, and split_10 and split_24 on NTU120. The visual feature extractor is Shift-GCN. 
* SMIE Experiment Setting: three datasets are used (NTU-60, NTU-120, PKU-MMD), and each dataset have three random splits. The visual feature extractor is classical ST-GCN to minimize the impact of the feature extractor and focus on the connection model.
-->
### SynSE Experiment Setting
Example for training and testing on NTU-60 split_5 data.
```bash
# SynSE Experiment Setting
$ python procedure.py with 'train_mode="sota"'
```

### SMIE Experiment Setting
Example for training and testing on NTU-60 split_1.  
```bash
# Optimized Experiment Setting
$ python procedure.py with 'train_mode="main"'
```

## Acknowledgement
* The codebase is from [SMIE](https://github.com/YujieOuO/SMIE).
  
## Licence
This project is licensed under the terms of the MIT license.

## Contact
For any questions, feel free to contact: jidongkuang@seu.edu.cn
