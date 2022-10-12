# Inducing Neural Collapse in Imbalanced Learning: Do We Really Need a Learnable Classifier at the End of Deep Neural Network?
Code for our paper [Inducing Neural Collapse in Imbalanced Learning: Do We Really Need a Learnable Classifier at the End of Deep Neural Network?](https://arxiv.org/pdf/2203.09081) (NeurIPS 2022).

by [Yibo Yang](https://iboing.github.io/), [Shixiang Chen](https://sites.google.com/view/shixiangchen), [Xiangtai Li](https://lxtgh.github.io/), Liang Xie, [Zhouchen Lin](https://zhouchenlin.github.io/), and Dacheng Tao.

If you find our work interesting and helpful to your project, please consider citing our paper.
```
@article{yang2022we,
  title={Do We Really Need a Learnable Classifier at the End of Deep Neural Network?},
  author={Yang, Yibo and Xie, Liang and Chen, Shixiang and Li, Xiangtai and Lin, Zhouchen and Tao, Dacheng},
  journal={arXiv preprint arXiv:2203.09081},
  year={2022}
}
```

The long-tailed classification code is based on [MiSLAS](https://github.com/dvlab-research/MiSLAS).


### Usage

The following command is for training.

``
python train.py --cfg {path_to_config}
``

The configs can be found in  `./config/DATASET/FILENAME.yaml`. The `FILENAME` that ends with `ETF_DR` uses our ETF classifier and the Dot-Regression loss. Otherwise, it is the baseline practice using a learnable classifier with the CE loss.

Note that the experiment on ImageNet is performed on multiple GPUs, while the other experiments (CIFAR-10, CIFAR-100, SVHN, and STL-10) are performed on one GPU.


### Contact

For any question, please contact Yibo ([ibo@pku.edu.cn](mailto:ibo@pku.edu.cn)).
