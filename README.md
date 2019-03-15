# describing-videos-by-exploiting-temporal-structure

This project tries to implement *SA-LSTM* proposed on **[Describing Videos by Exploiting Temporal Structure](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Yao_Describing_Videos_by_ICCV_2015_paper.pdf)**[1] published in **ICCV 2015**.



# Requirements

* Ubuntu 16.04
* CUDA 9.0
* cuDNN 7.3.1
* Python 2.7.12
  * PyTorch 1.0
  * Other python libraries specified in requirements.txt



# How to use

## Step 1. Setup python virtual environment

```
$ virtualenv .env
$ source .env/bin/activate
(.env) $ pip install --upgrade pip
(.env) $ pip install -r requirements.txt
```


## Step 2. Prepare Data

1. Extract feature vectors of datasets, and locate them at `~/<dataset>/features/<network>.hdf5`
   
   > e.g. InceptionV4 feature vectors of MSVD dataset will be located at `~/data/MSVD/features/InceptionV4.hdf5`.

2. Split datasets into a train / val / test set by running following commands.
   
   ```
   (.env) $ python -m split.MSVD
   (.env) $ python -m split.MSR-VTT
   ```
   

## Step 3. Prepare Evaluation Codes

Clone evaluation codes from [the official coco-evaluation repo](https://github.com/tylin/coco-caption).

   ```
   (.env) $ git clone https://github.com/tylin/coco-caption.git
   (.env) $ mv coco-caption/pycocoevalcap .
   (.env) $ rm -rf coco-caption
   ```

## Step 4. Train

Run
   ```
   (.env) $ CUDA_VISIBLE_DEVICES=0 python train.py
   ```

You can change some hyperparameters by modifying `config.py`.


## Step 5. Inference

1. Set the checkpoint path in `run.py` with a variable named `ckpt_fpath`.
2. Run
   ```
   (.env) $ CUDA_VISIBLE_DEVICES=0 python run.py
   ```


# Performances

* MSVD

  | Model | Features | BLEU4 | METEOR | CIDEr | ROUGE_L |
  | :---: | :---: | :---: | :---: | :---: | :---: |
  | SA-LSTM[1] | GoogLeNet[2], 3D conv. (HOG+HOF+MBH) | 41.92 | 29.6 | 51.67 | - |
  | SA-LSTM[3] | InceptionV4[4] | 45.3 | 31.9 | **76.2** | 64.2 |
  | Ours | InceptionV4 | **46.14** | **32.60** | 71.09 | **68.13** |


* MSR-VTT

  | Model | Features | BLEU4 | METEOR | CIDEr | ROUGE_L |
  | :---: | :---: | :---: | :---: | :---: | :---: |
  | SA-LSTM[3] | InceptionV4 (RGB) | **36.3** | 25.5 | **39.9** | **58.3** |
  | Ours | InceptionV4 (RGB) | 36.10 | **25.69** | 39.75 | 57.66 |


# References

[1] Yao, Li, et al. "Describing videos by exploiting temporal structure." Proceedings of the IEEE international conference on computer vision. 2015.

[2] Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

[3] Wang, Bairui, et al. "Reconstruction Network for Video Captioning." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

[4] Szegedy, Christian, et al. "Inception-v4, inception-resnet and the impact of residual connections on learning." AAAI. Vol. 4. 2017.
