# Multi-Stream Refinement Network for Human Object Interaction Detection

Human Object Interaction (HOI) detection has always been a crucial problem in comprehensive visual understanding such as activity understanding and human intention prediction. The HOI aims to detect <human, action, object> triplets within an image, where not only all object and human instances are detected, but also the possible interactions between human and objects have to be analyzed. Many existing methods often exploit to integrate the human and object visual features , the spatial layout of human-object pairs, human poses, contextual information, and even object semantic labels into a framework to learn the interactions, proving that all these components can contribute to improve the HOI detection. However, most methods simply concatenate these components that are not explicitly embedded in the feature learning for  HOI detection. In this work, we are trying to model these components explicitly using one uniform multi-stream feature neural network that can effectively learn the visual features of humans, contexts, and objects, while the network receives the attentions from human poses, spatial configurations, and semantic prior knowledge of objects to refine the three types of visual features, respectively. In addition, a graph neural network is employed here to learn the structural features of human-object pairs. We verified our method on V-COCO and HICO-DET datasets with extensive experiments, and compare our method with other state-of-the-art methods. The experimental results demonstrate that our method is effective yet simple, achieving superior performance to those state-of-the-art methods.

## Evaluation results on V-COCO test dataset
<center>

| Method | baseline | spatial-attention | pose-attention | object semantic information | mAP(Scenario1) | mAP(Scenario2) |
|:------:|:--------:|:-----:|:-------:|:-------:|:-------:|:-------:
| a | √ | × | × | × | 48.07 | 52.51 |
| b | √ | √ | × | × | 49.86 | 54.51 |
| c | √ | × | √ | × | 49.27 | 54.10 |
| d | √ | × | × | √ | 49.39 | 54.26 |
| e | √ | √ | √ | × | 51.23 | 56.04 |
| f | √ | v | √ | √ | 51.39 | 56.33 |
| g | √ | v | √ | √ | 50.65 | 55.46 |
| h | √ | × | √ | √ | 52.13 | 57.03 |


**Note**: Ablation study of our method on V-COCO dataset.

##  Result comparisons on V-COCO test dataset 



|      Method      | Backbone         | mAP(Scenario1) | mAP(Scenario2) |
| :--------------: | :----------------: | :---------: | :--------: |
| Gupta et al.[7]  | ResNet-50-FPN    | 31.8      |-         |
|  InteractNet[9]  | ResNet-50-FPN    | 40.0      | 47.98    |
|     GPNN[37]     | ResNet-50        | 44.0      | -        |
|     iCAN[10]     | ResNet-50        | 45.3      | 52.4     |
|     RPNN[30]     | ResNet-50        | 47.5      | -        |
|   Li et al.[42]  | ResNet-50        | 48.7      | -        |
|    VSGNet[12]    | ResNet-152       | 51.8      | 57.0     |
|    PMFNet[14]    | ResNet-50-FPN    | 52.0      | -        |
|    PFNet[32]     | ResNet-50        | 52.8      | -        |
|    Ours          | ResNet-50        | 50.5      | 55.4     |
|     **Ours**     | **ResNet-152**   | **52.1**  |**57.0** |


##  Result comparisons on HICO-DET test dataset in Default setting


|      Method      | Backbone         | mAP(Full) | mAP(Rare)| mAP(None-Rare)|
| :--------------: | ---------------- | --------- | -------- |-------- |
|    HO-RCNN[8]    | CaffeNet         | 7.81      | 5.37     | 8.54  |
|  InteractNet[9]  | ResNet-50-FPN    | 9.94      | 7.16     | 10.77 |
|     GPNN[37]     | ResNet-50        | 13.11     | 9.34     | 14.23 |
|     iCAN[10]     | ResNet-50        | 14.84     | 10.45    | 16.15 |
|   Li et al.[42]  | ResNet-50        | 17.22     | 13.51    | 18.32 |
|     RPNN[30]     | ResNet-50        | 17.35     | 12.78    | 18.71 |
|    PMFNet[14]    | ResNet-50-FPN    | 17.46     | 15.65    | 18.00 |
|    VSGNet[12]    | ResNet-152       | 19.80     | 16.05    | 20.91 |
|    PFNet[32]     | ResNet-50        | 20.05     | 16.66    | 21.07 |
|     Ours         | ResNet-50        | 20.12     | 17.09    | 21.03 |
|     **Ours**     | **ResNet-152**   | **20.99** |**18.12** | **21.85** |

##  Result comparisons on HICO-DET test dataset in Known Object setting


|      Method      | Backbone         | mAP(Full)  | mAP(Rare)| mAP(None-Rare)|
| :--------------: | ---------------- | ---------  | -------- |-------- |
|    HO-RCNN[8]    | CaffeNet         | 10.41      | 8.94     | 10.85  |
|  InteractNet[9]  | ResNet-50-FPN    | -          | -        | -      |
|     GPNN[37]     | ResNet-50        | -          | -        | -      |
|     iCAN[10]     | ResNet-50        | 16.26      | 11.33    | 17.73  |
|   Li et al.[42]  | ResNet-50        | 19.38      | 15.38    | 20.57  |
|     RPNN[30]     | ResNet-50        | -          | -        | -      |
|    PMFNet[14]    | ResNet-50-FPN    | 20.34      | 17.47    | 21.20  |
|    VSGNet[12]    | ResNet-152       | -          | -        | -      |
|    PFNet[32]     | ResNet-50        | 24.01      | 21.09    | 24.89  |
|     Ours         | ResNet-50        | 23.10      | 20.48    | 24.01  |
|     **Ours**     | **ResNet-152**   | **23.96** |**21.37**  | **24.73** |


## Quick start
### Installation
 1. Install pytorch >= v1.3.0.

 2. Clone this repo.and we'll call the directory that you cloned as ```ROOT```.

 3. Install dependencies(preferable to run in a python2 virtual environment):
  ```
    pip2 install -r requirements.txt
  ```
  
For HICO_DET evaluation we will use python3 environment, to install those packages:
  ```
pip3 install -r requirements3.txt
  ```
  
Run only compute_map.sh in a python 3 enviornment. For all other use python 2 environment.

 4. Our datasets and annotations and some necessary files provided by the [VSGNet](https://github.com/ASMIftekhar/VSGNet), you can download the data from [here](https://drive.google.com/drive/folders/1J8mN63bNIrTdBQzq7Lpjp4qxMXgYI-yF?usp=sharing). Then you will get two folders in the directory "All_data" and "infos", this will take close to 10GB space. About keypoints datas, can be generated using the keypoint detection model from [Detectron2](https://github.com/facebookresearch/detectron2), and placed in a folder starting with "KP".
 
  The All_data folder should like this:
```
All_data
├─Annotations_hico
│  ├─test_annotations
│  └─train_annotations
├─Annotations_vcoco
│  ├─test_annotations
│  ├─train_annotations
│  └─val_annotations
├─bad_detections_hico
│  ├─bad_detections_test
│  └─bad_detections_train
├─Data_hico
│  ├─test2015
│  └─train2015
├─Data_vcoco
│  ├─train2014
│  └─val2014
├─hico_infos
├─KP_Detections_hico
│  ├─test
│  └─train
├─KP_Detections_vcoco
│  ├─train
│  └─val
├─Object_Detections_hico
│  ├─test
│  └─train
├─Object_Detections_vcoco
│  ├─train
│  └─val
└─v-coco
```

 5. Training & Testing
 - Training in V-COCO
 ```
    cd ROOT/scripts_vcoco/
    CUDA_VISIBLE_DEVICES=0 python2 main.py -fw new_test -ba 8 -l 0.001 -e 60 -sa 20 
 ```
**Flags description**:

**-fw:** Name of the folder in which the result will be stored.

**-ba:** Batch size.

**-l:** Learning rate.

**-e:** Number of epochs.

**-sa:** After how many epochs the model would be saved,remember by default for every epoch the best model will be saved.

 - Training in HICO_DET
 ```
    cd ROOT/scripts_hico/
    CUDA_VISIBLE_DEVICES=0 python2 main.py -fw new_test -ba 8 -l 0.001 -e 80 -sa 20 
 ```
 - Evaluation in V-COCO

 ```
    cd ROOT/scripts_vcoco/
 ```
 To store the best result in v-coco format run:
 ```
    CUDA_VISIBLE_DEVICES=0 python2 main.py -fw new_test -ba 8 -r t -i t
 ```
To see the results in original v-coco scheme:
 ```
python2 calculate_map_vcoco.py -fw new_test -sa 40 -t test
 ```
 - Evaluation in HICO_DET
 ```
    cd ROOT/scripts_hico/
 ```
 To store the best result in HICO_DET format run:
 ```
    CUDA_VISIBLE_DEVICES=0 python2 main.py -fw new_test -ba 8 -r t -i t
 ```
To see the results in original HICO_DET scheme:
 ```
    cd ROOT/scripts_hico/HICO_eval/
    bash compute_map.sh new_test 20
 ```
**Note**: 20 indicates the number of cpu cores to be used for evaluation.
 ***
 ## Citation
 If you find this code useful, please cite our work with the following bibtex:
 ```
 @article{SHAO2022103529,
title = {Multi-stream feature refinement network for human object interaction detection},
journal = {Journal of Visual Communication and Image Representation},
volume = {86},
pages = {103529},
year = {2022},
issn = {1047-3203},
doi = {https://doi.org/10.1016/j.jvcir.2022.103529},
url = {https://www.sciencedirect.com/science/article/pii/S1047320322000712},
author = {Zhanpeng Shao and Zhongyan Hu and Jianyu Yang and Youfu Li},
 ```

