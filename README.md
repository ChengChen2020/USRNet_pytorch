# USRNet_pytorch

A non-official implementation for [Deep unfolding network for image super-resolution](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Deep_Unfolding_Network_for_Image_Super-Resolution_CVPR_2020_paper.pdf).

## Data Preparation

- Training datasets
  - [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/): Proposed in NTIRE17 (800 train and 100 validation)
  - [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar): 2650 2K images from Flickr for training
- Testing datasets
  - [Set5](https://github.com/cszn/DPSR/tree/master/testsets/Set5/GT):
  - [BSD68](https://github.com/cszn/DPSR/tree/master/testsets/BSD68/GT): 
  - Real images: TODO

## Train

```python
python usrnet_train.py > train_log.txt
```

## Test

```python
python usrnet_test.py --save_LEH > test_log.txt # if saving Low-Resolution (L), Estimated (E) and High-Resolution (H) test images.
```

## Code Structure

The official implementation is at https://github.com/cszn/USRNet.

https://github.com/tkkcc/prior/blob/879a0b6c117c810776d8cc6b63720bf29f7d0cc4/util/gen_kernel.py

https://github.com/assafshocher/BlindSR_dataset_generator

## Reference

```BibTex
@inproceedings{zhang2020deep,
  title={Deep unfolding network for image super-resolution},
  author={Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3217--3226},
  year={2020},
  url={https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Deep_Unfolding_Network_for_Image_Super-Resolution_CVPR_2020_paper.pdf}
}
```
