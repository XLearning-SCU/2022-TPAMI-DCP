# Dual Contrastive Prediction for Incomplete Multi-view Representation Learning

## What's New

- [2024-7] 💡Now we support **arbitrary number** of views for multi-view clustering and classification tasks. See `run_clustering_multiview.py` and `model_multiview.py` for more details.
```python
python run_clustering_multiview.py --missing_rate 0.5
python run_supervised_multiview.py --missing_rate 0.5
```

## Intro
This repo contains the code and data of our IEEE TPAMI'2022 paper Dual Contrastive Prediction for Incomplete Multi-view Representation Learning. Precise numerical results of different missing rates could be accessed from [Results_missing_rate.xlsx](https://github.com/XLearning-SCU/2022-TPAMI-DCP/blob/main/Results_missing_rate.xlsx).

> [Dual Contrastive Prediction for Incomplete Multi-view Representation Learning](http://pengxi.me/wp-content/uploads/2023/02/DCP-2023_compressed.pdf)
>
> [COMPLETER: Incomplete Multi-view Clustering via Contrastive Prediction](http://pengxi.me/wp-content/uploads/2021/03/2021CVPR-completer.pdf)

![framework](figure//framework.png)

## Requirements

pytorch>=1.2.0 

numpy>=1.19.1

scikit-learn>=0.23.2

munkres>=1.1.4

## Configuration

The hyper-parameters, the training options are defined in the configure folder.
- configure_clustering.py: bi-view data clustering
- configure_clustering_multiview.py: 3-view data clustering
- configure_supervised.py: bi-view data classification (including human action recognition)
- configure_supervised_multiview.py: 3-view data classification

Note that for multi-view setting, we place both complete graph and cove view setting (i.e., ```type='CG' or 'CV' ```).

## Datasets

The Caltech101-20, LandUse-21, Scene-15, UWA, and DHA datasets are placed in "data" folder. The NoisyMNIST dataset could be downloaded from [cloud](https://drive.google.com/file/d/1b__tkQMHRrYtcCNi_LxnVVTwB-TWdj93/view?usp=sharing).

## Usage

The code includes:

- an example implementation of the model. The network structure and training/evaluation pipeline are in 
```model.py``` and ```model.multiview.py: ```

- clustering tasks for different missing rates.
```bash
python run_clustering.py --dataset 0 --devices 0 --print_num 100 --test_time 5 --missing_rate 0.5
python run_clustering_multiview.py 
```
- classification tasks for different missing rates.
```bash
python run_supervised.py --dataset 0 --devices 0 --print_num 100 --test_time 5 --missing_rate 0.5
python run_supervised_multiview.py
```
- human action recognition tasks
```bash
python run_HAR.py 
```

You can get the following output by runing ```python run_HAR.py```:

```bash
Epoch : 100/2000 ===> Reconstruction loss = 5.1242===> Reconstruction loss = 0.0489 ===> Map loss = 0.0001 ===> Map loss = 0.0001 ===> Loss_icl = -7.4860e+01 ===> Loss_ccl = 1.2800e+02 ===> All loss = 5.3657e+01
RGB   Accuracy on the test set is 0.6653
Depth Accuracy on the test set is 0.3926
RGB+D Accuracy on the test set is 0.8430
onlyRGB Accuracy on the test set is 0.6860
onlyDepth Accuracy on the test set is 0.3636
Epoch : 2000/2000 ===> Reconstruction loss = 4.3108===> Reconstruction loss = 0.0163 ===> Map loss = 0.0001 ===> Map loss = 0.0004 ===> Loss_icl = -7.7413e+01 ===> Loss_ccl = 1.2800e+02 ===> All loss = 5.1020e+01
RGB   Accuracy on the test set is 0.7769
Depth Accuracy on the test set is 0.8306
RGB+D Accuracy on the test set is 0.8926
onlyRGB Accuracy on the test set is 0.7727
onlyDepth Accuracy on the test set is 0.8182
```

## Reference

If you find our work useful in your research, please consider citing:

```latex
@ARTICLE{9852291,
  author={Lin, Yijie and Gou, Yuanbiao and Liu, Xiaotian and Bai, Jinfeng and Lv, Jiancheng and Peng, Xi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Dual Contrastive Prediction for Incomplete Multi-View Representation Learning}, 
  year={2022},
  doi={10.1109/TPAMI.2022.3197238}
}
@inproceedings{lin2021completer,
   title={COMPLETER: Incomplete Multi-view Clustering via Contrastive Prediction},
   author={Lin, Yijie and Gou, Yuanbiao and Liu, Zitao and Li, Boyun and Lv, Jiancheng and Peng, Xi},
   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
   month={June},
   year={2021}
}
```

