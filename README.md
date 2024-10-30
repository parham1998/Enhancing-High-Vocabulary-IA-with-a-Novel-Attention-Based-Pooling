# Enhancing-High-Vocabulary-IA-with-a-Novel-Attention-Based-Pooling
Official Pytorch Implementation of: "[Enhancing High-Vocabulary Image Annotation with a Novel Attention-Based Pooling](https://doi.org/10.1007/s00371-024-03618-6)"

## Datasets
Three well-known datasets are mostly used in AIA tasks. In addition, we have utilized a dataset with a significantly larger number of images and a vocabulary list consisting of 500 words, which has a very high level of complexity. The table below provides details about these datasets. It is also possible to download them from the links provided. (After downloading each dataset, replace its 'images' folder with the corresponding 'images' folder in the 'datasets' folder).

| *Dataset* | *Num of images* | *Num of training images* | *Num of testing images*  | *Num of vocabularies*  | *Labels per image*  | *Image per label* |
| :------------: | :-------------: | :-------------: | :-------------: | :------------: | :-------------: | :-------------: |
| [Corel 5k](https://www.kaggle.com/datasets/parhamsalar/corel5k) | 5,000 | 4,500 | 500 | 260 | 3.4 | 58.6 |
| [ESP Game](https://www.kaggle.com/datasets/parhamsalar/espgame) | 20,770 | 18,689 | 2081 | 268 | 4.7 | 362.7 |
| [IAPR TC-12](https://www.kaggle.com/datasets/parhamsalar/iaprtc12) | 19,627 | 17,665 | 1962 | 291 | 5.7 | 347.7 |
| [VG-500](https://visualgenome.org/) | 92,904 | 82,904 | 10,000 | 500 | 13.6 | 2256.6 |

We employed the [SSGRL](https://github.com/HCPLab-SYSU/SSGRL) settings when working with the VG 500 dataset, which involves selecting images from the 500 most common categories and then dividing the data into training and testing subsets. We also attempted to identify the names of labels (vocabulary) for the mentioned dataset. Please let us know if there are any errors.

## model
![model](https://user-images.githubusercontent.com/85555218/230767368-82d92d2b-9374-4198-bd98-f548ce1bc788.jpg)

## Attention Maps
![Attention](https://github.com/parham1998/Enhancing-High-Vocabulary-IA-with-a-Novel-Attention-Based-Pooling/assets/85555218/44d24fe9-886d-4f65-bcf3-4175b55c8f0c)

## Train and Evaluation
To train the model in Spyder IDE use the code below:
```python
run main.py --data {select training dataset} --loss-function {select loss function}
```
Please note that:
1) You should put **Corel-5k**, **ESP-Game**, **IAPR-TC-12**, or **VG-500** in {select training dataset}.

2) You should put the **proposedLoss** in {select loss function}.

3) When using the **VG-500** dataset, change the "image-size" to 576, change the "gamma_neg" in **proposedLoss** to 2, and set batch size to 128.

To evaluate the model in Spyder IDE use the code below:
```python
run main.py --data {select training dataset} --loss-function {select loss function} --evaluate
```

## Results
Proposed method:
| data | precision | recall | f1-score | N+ | mAP |
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
Corel 5k | 0.453 | 0.611 | **0.520** | **202** | - |
IAPR TC-12 | 0.515 | 0.584 | **0.547** | **287** | - |
ESP Game | 0.442 | 0.500 | **0.470** | **262** | - |
VG-500 | 0.409 | 0.502 | **0.451** | **477** | 42.515 |

## Citation
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows:
```
@article{salar2024enhancing,
  title={Enhancing high-vocabulary image annotation with a novel attention-based pooling},
  author={Salar, Ali and Ahmadi, Ali},
  journal={The Visual Computer},
  pages={1--15},
  year={2024},
  publisher={Springer}
}
```

## Contact
I would be happy to answer any questions you may have - Ali Salar (parham1998resume@gmail.com)
