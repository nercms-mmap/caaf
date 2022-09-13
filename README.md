# Confidence-Aware Active Feedback for Interactive Instance Search

<!--正式发表后修改原文链接-->
This repo is the official implementation of ["Confidence-Aware Active Feedback for Interactive Instance Search"](https://arxiv.org/abs/2110.12255) by Yue Zhang, Chao Liang and Longxiang Jiang.

## Abstract
Online relevance feedback (RF) is widely utilized in instance search (INS) tasks to further refine imperfect ranking results, but it often has low interaction efficiency. The active learning (AL) technique addresses this problem by selecting valuable feedback candidates. However, mainstream AL methods require an initial labeled set for a cold start and are often computationally complex to solve. Therefore, they cannot fully satisfy the requirements for online RF in interactive INS tasks. To address this issue, we propose a confidence-aware active feedback method (CAAF) that is specifically designed for online RF in interactive INS tasks. Inspired by the explicit difficulty modeling scheme in self-paced learning, CAAF utilizes a pairwise manifold ranking loss to evaluate the ranking confidence of each unlabeled sample. The ranking confidence improves not only the interaction efficiency by indicating valuable feedback candidates but also the ranking quality by modulating the diffusion weights in manifold ranking. In addition, we design two acceleration strategies, an approximate optimization scheme and a top-K search scheme, to reduce the computational complexity of CAAF. Extensive experiments on both image INS tasks and video INS tasks searching for buildings, landscapes, persons, and human behaviors demonstrate the effectiveness of the proposed method. Notably, in the real-world, large-scale video INS task of NIST TRECVID 2021, CAAF uses 25% fewer feedback samples to achieve a performance that is nearly equivalent to the champion solution. Moreover, with the same number of feedback samples, CAAF's mAP is 51.9%, significantly surpassing the champion solution by 5.9%.

## Approach

<div align="center">
  <img width="100%" src="./assets/framework.png">
</div>

## Example

We provide the preprocessed data of Oxford5k, Holidays and CUHK03 on [GoogleDrive](https://drive.google.com/drive/folders/1jUczrybe9i5NeRJjpWxNBb1Yo5ry6dDH?usp=sharing) and [Baiduyun](https://pan.baidu.com/s/1eL1Pp3FhNzUxSQzbCY0cZw?pwd=d2t6), please put them (`*.mat`) into the `./data/` folder.

You can test CAAF on Oxford5k with default settings using the following code:
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
cd code
python main.py
```

You can also test on Holidays and CUHK03 with other parameters by modifying the following lines in `main.py`:
```
dataset = 'oxford5k' #['oxford5k','holidays','cuhk03']
params = init_params(dataset=dataset,t=5,q=5,k=300,alpha=1e-2,method="CAAF")
```

## Results

|  T  | Holidays | Oxford5k | CUHK03 |
|:---:|:--------:|:--------:|:------:|
|  0  |   67.14  |   43.19  |  53.90 |
|  1  |   79.83  |   50.75  |  72.17 |
|  2  |   84.02  |   55.78  |  85.55 |
|  3  |   86.21  |   59.46  |  89.97 |
|  4  |   87.46  |   61.51  |  91.33 |

## Citation

If you find this repository useful, please consider giving ⭐ or citing:

<!--正式发表后修改bibtex-->

```
@article{DBLP:journals/corr/abs-2110-12255,
  author    = {Yue Zhang and
               Chao Liang and
               Longxiang Jiang},
  title     = {Confidence-Aware Active Feedback for Efficient Instance Search},
  journal   = {CoRR},
  volume    = {abs/2110.12255},
  year      = {2021}
}
```
