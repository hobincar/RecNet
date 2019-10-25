# RecNet

This project tries to implement *RecNet* proposed in **[Reconstruction Network for Video Captioning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Reconstruction_Network_for_CVPR_2018_paper.pdf) [1], *CVPR 2018***.



# Environment

* Ubuntu 16.04
* CUDA 9.0
* cuDNN 7.3.1
* Nvidia Geforce GTX Titan Xp 12GB


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

1. Extract Inception-v4 [2] features from datasets, and locate them at `<PROJECT ROOT>/<DATASET>/features/<DATASET>_InceptionV4.hdf5`. I extracted features from [here](https://github.com/hobincar/pytorch-video-feature-extractor). <br/>
   [[MSVD InceptionV4 Feature]](https://drive.google.com/open?id=18aZ8AdFeJ8h2wPR3YMnZNHnw7ebtfGih)
   [[MSVD ResNet Feature]](https://drive.google.com/open?id=1efQ2aBRhDuLz2SVZjVcMBwZq5f_ZKtLI)
   [[MSVD 3D-ResNext Feature]](https://drive.google.com/open?id=1XvJf-5yyOv-DicNqp-Z9nM9EHwZ3KB5E)<br/>
   [[MSR-VTT InceptionV4 Feature]](https://drive.google.com/open?id=1pFh4u-KwSnCFRl6UJgg7yeaLo2GbxkVT)

2. Split the dataset along with the official splits by running following:

   ```
   (.env) $ python -m splits.MSVD
   (.env) $ python -m splits.MSR-VTT
   ```
   

## Step 3. Prepare Evaluation Codes

Clone evaluation codes from [the official coco-evaluation repo](https://github.com/tylin/coco-caption).

   ```
   (.env) $ git clone https://github.com/tylin/coco-caption.git
   (.env) $ mv coco-caption/pycocoevalcap .
   (.env) $ rm -rf coco-caption
   ```

## Step 4. Train

* Stage 1 (Encoder-Decoder)

   ```
   (.env) $ python train.py -c configs.train_stage1
   ```

* Stage 2 (Encoder-Decoder-Reconstructor

   Set the `pretrained_decoder_fpath` of `TrainConfig` in `configs/train_stage2.py` as the checkpoint path saved at stage 1, then run

   ```
   (.env) $ python train.py -c configs.stage2
   ```
   
You can change some hyperparameters by modifying `configs/train_stage1.py` and `configs/train_stage2.py`.


## Step 5. Inference

1. Set the checkpoint path by changing `ckpt_fpath` of `RunConfig` in `configs/run.py`.
2. Run
   ```
   (.env) $ python run.py
   ```


# Performances

\* *NOTE: As you can see, the performance of RecNet does not outperform SA-LSTM. Better hyperparameters should be found out.*

* MSVD

  | Model | BLEU4 | CIDEr | METEOR | ROUGE_L | pretrained |
  | :---: | :---: | :---: | :---: | :---: | :---: |
  | SA-LSTM | 45.3 | 76.2 | 31.9 | 64.2 | - |
  | RecNet (global) | 51.1 | 79.7 | 34.0 | 69.4 | - |
  | RecNet (local) | **52.3** | **80.3** | **34.1** | **69.8** | - |
  |  |  |  |  |  |  |
  | (Ours) SA-LSTM | 50.9	| 79.6 |	33.4 |	69.6 | [link](https://drive.google.com/open?id=1Sk24rvyhh13Wiq3PUuASscNZiIUumcsk) |
  | (Ours) RecNet (global) | 50.2 |	78.4 |	33.5 |	69.8 | [link](https://drive.google.com/open?id=1VC3WiXyQmuMqeja02GySuWEdWmzySY9Y) |
  | (Ours) RecNet (local) | 50.1 |	78.6 |	33.1 |	69.4 | [link](https://drive.google.com/open?id=1RY_pVPQILAw5i5rlb4wQnTG3ckOXM99M) |


* MSR-VTT

  | Model | BLEU4 | CIDEr | METEOR | ROUGE_L | pretrained |
  | :---: | :---: | :---: | :---: | :---: | :---: |
  | SA-LSTM | 36.3 | 39.9 | 25.5 | 58.3 | - |
  | RecNet (global) | 38.3 | 41.7 | 26.2 | 59.1 | - |
  | RecNet (local) | **39.1** | **42.7** | **26.6** | **59.3** | - |
  |  |  |  |  |  |  |
  | (Ours) SA-LSTM | 38.0	| 40.2 |	25.6 |	58.1 | [link](https://drive.google.com/open?id=1tXwm13xyv-GM0khTFC9ESHVdRbO2ysEY) |
  | (Ours) RecNet (global) | 37.5 |	40.0	| 25.5	| 57.9 | [link](https://drive.google.com/open?id=1WpVDoxg3KwnHp6zyIGlxrm-kQtq0rW1O) |
  | (Ours) RecNet (local) | 37.0 |	40.0 |	25.5 |	57.9 | [link](https://drive.google.com/open?id=1XQPDXmPLo4coZ02onk_GUaHcqokAujlr) |


# References

[1] Wang, Bairui, et al. "Reconstruction Network for Video Captioning." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

[2] Szegedy, Christian, et al. "Inception-v4, inception-resnet and the impact of residual connections on learning." AAAI. Vol. 4. 2017.
