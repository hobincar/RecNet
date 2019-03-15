# RecNet

This project tries to implement *RecNet* proposed on **[Reconstruction Network for Video Captioning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Reconstruction_Network_for_CVPR_2018_paper.pdf)**[1] published in **CVPR 2018**.



# Environment

* Ubuntu 16.04
* CUDA 9.0
* cuDNN 7.3.1


# Requirements
* Java 8
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
  | RecNet (global) | InceptionV4 [2] | 51.1 | 34.0 | 69.4 | 79.7 |
  | RecNet (local) | InceptionV4 | **52.3** | **34.1** | **69.8** | **80.3** |
  | Ours (global) | InceptionV4 | - | - | - | - |
  | Ours (local) | InceptionV4 | - | - | - | - |


* MSR-VTT

  | Model | Features | BLEU4 | METEOR | CIDEr | ROUGE_L |
  | :---: | :---: | :---: | :---: | :---: | :---: |
  | RecNet (global) | InceptionV4 | 38.3 | 26.2 | 59.1 | 41.7 |
  | RecNet (local) | InceptionV4 | **39.1** | **26.6** | **59.3** | **42.7** |
  | Ours (global) | InceptionV4 | - | - | - | - |
  | Ours (local) | InceptionV4 | - | - | - | - |


# References

[1] Wang, Bairui, et al. "Reconstruction Network for Video Captioning." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

[2] Szegedy, Christian, et al. "Inception-v4, inception-resnet and the impact of residual connections on learning." AAAI. Vol. 4. 2017.
