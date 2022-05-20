# Dynamic Graph CNN for Learning on Point Clouds
We propose a new neural network module dubbed EdgeConv suitable for CNN-based high-level tasks on point clouds including classification and segmentation. EdgeConv is differentiable and can be plugged into existing architectures.

[[Project]](https://liuziwei7.github.io/projects/DGCNN) [[Paper]](https://arxiv.org/abs/1801.07829) [[Press]](http://news.mit.edu/2019/deep-learning-point-clouds-1021)

## Overview
`DGCNN` is the author's re-implementation of Dynamic Graph CNN, which achieves state-of-the-art performance on point-cloud-related high-level tasks including category classification, semantic segmentation and part segmentation.

<img src='./tensorflow/misc/demo_teaser.png' width=800>

Further information please contact [Yue Wang](https://www.csail.mit.edu/person/yue-wang) and [Yongbin Sun](https://autoid.mit.edu/people-2).

## Author's Implementations

The classification experiments in our paper are done with the pytorch implementation.

* [tensorflow-dgcnn](./tensorflow)
* [pytorch-dgcnn](./pytorch)

## Other Implementations
* [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.EdgeConv)
* [pytorch-dgcnn](https://github.com/AnTao97/dgcnn.pytorch) (This implementation on S3DIS achieves significant better results than our tensorflow implementation)

## Generalization under Corruptions

The performance is evaluated on [ModelNet-C](https://github.com/jiawei-ren/ModelNet-C) with mCE (lower is better) and clean OA (higher is better).

| Method          | Reference                                                  | Standalone |  mCE  | Clean OA |
| --------------- | ---------------------------------------------------------- | :--------: | :---: | :------: |
| PointNet        | [Qi et al.](https://arxiv.org/abs/1612.00593)              |     Yes    | 1.422 |   0.907  |
| DGCNN           | [Wang et al.](https://arxiv.org/abs/1801.07829)            |     Yes    | 1.000 |   0.926  |


## Real-World Applications
* DGCNN has been successfully applied to [ParticalNet in Large Hadron Collider (LHC)](https://arxiv.org/abs/1902.08570).


## Citation
Please cite this paper if you want to use it in your work,

	@article{dgcnn,
	  title={Dynamic Graph CNN for Learning on Point Clouds},
	  author={Wang, Yue and Sun, Yongbin and Liu, Ziwei and Sarma, Sanjay E. and Bronstein, Michael M. and Solomon, Justin M.},
	  journal={ACM Transactions on Graphics (TOG)},
	  year={2019}
	}

## License
MIT License

## Acknowledgement
The structure of this codebase is borrowed from [PointNet](https://github.com/charlesq34/pointnet).
