# USRNet_pytorch

A non-official implementation for [Deep unfolding network for image super-resolution](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Deep_Unfolding_Network_for_Image_Super-Resolution_CVPR_2020_paper.pdf).

## Data Preparation

- Training datasets
  - [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/): Proposed in NTIRE17 (800 train and 100 validation)
  - [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar): 2650 2K images from Flickr for training
- Testing datasets
  - [Set5](https://github.com/cszn/DPSR/tree/master/testsets/Set5/GT): A dataset consisting of 5 images (“baby”, “bird”, “butterfly”, “head”, “woman”) commonly used for testing performance of Image Super-Resolution models.
  - [BSD68](https://github.com/cszn/DPSR/tree/master/testsets/BSD68/GT): Berkeley Segmentation Dataset.
  - [Real images](https://github.com/cszn/USRNet/tree/master/testsets/set_real): `chip.png` and `comic.png`

## Train

```python
python usrnet_train.py > train_log.txt
```
Intermediate validation results and model weights will be saved at `train_log`.

## Test

```python
# Test on BSD68
python usrnet_test.py --save_LEH > test_log.txt # if saving Low-Resolution (L), Estimated (E) and High-Resolution (H) test images.
# Test on real images
python usrnet_real.py
```
Results will be saved at `test_log`.

## Code Structure

The official implementation is at https://github.com/cszn/USRNet.

- `utils_kernel.py`
  - Functions for synthesizing motion and gaussian blur kernels.
  - **Copied**) from https://github.com/tkkcc/prior/blob/879a0b6c117c810776d8cc6b63720bf29f7d0cc4/util/gen_kernel.py and https://github.com/assafshocher/BlindSR_dataset_generator.
- `utils_image.py`
  - Utiliy functions such as kernel visualization, image augmentation, tensor manipulation and PSNR calculation.
  - **Copied** from the official repository with minor modification.
- `usrnet_real.py`
  - Real image super-resolution
  - **Copied** from the official repository.
- `usrnet_data.py`
  - Data processing part.
  - **Modified** from the official repository.
    - Make adjustments to the interfaces, image augmentation strategies, etc.
    - Fix the outdated function calls.
    - Add more comments and unit tests.
- `usrnet_model.py`
  - Model definition part.
  - **Modified** from the official repository.
    - Simplify the interfaces.
    - Adjust definitions that are redundant or not in use.
    - Add more comments and unit tests for each module.
- All others are **original** code.
  - `usrnet_train.py`
  - `usrnet_test.py`
  - `vis_psnr.py`
  - `vis_kernels.py`

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
