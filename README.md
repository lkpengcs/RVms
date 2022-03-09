# Student Become Decathlon Master in Retinal Vessel Segmentation via Dual-teacher Multi-target Domain Adaptation

Pytorch implementation for our multi-target domain adaptation method with application to retinal vessel segmentation. We use style transfer, style augmentation and dual-teacher knowledge distillation to train a domain-generic student on multiple domains.

![Network](https://github.com/lkpengcs/RVms/blob/main/figs/1.png)

![Module](https://github.com/lkpengcs/RVms/blob/main/figs/3.png)

## Paper

Please cite our [paper](https://arxiv.org/abs/2203.03631) if you find the code or dataset useful for your research.

## Dataset

You can download our dataset **mmRV** via this [link](https://drive.google.com/drive/folders/1QxGKT9t38SWApXa_webQpC1udRzW23G1?usp=sharing).

![Results](https://github.com/lkpengcs/RVms/blob/main/figs/2.png)

The detailed information is listed in the following table.

|   Modality   | Original Dataset | Train:Test |  Size   |
| :----------: | :--------------: | :--------: | :-----: |
|     OCTA     |     OCTA-500     |   395:50   | 384×384 |
|     OCTA     |       ROSE       |    30:9    | 304×304 |
|     OCT      |     OCTA-500     |   395:50   | 384×384 |
| Fundus Image |      DRIVE       |   34:36    | 512×512 |
| Fundus Image |       HRF        |   30:15    | 512×512 |
|  UWF Fundus  |    PRIME-FP20    |   40:20    | 512×512 |

## Usage

### Prerequisite

- Python 3.8+
- Pytorch 1.8+
- tensorboardX

*The code is coming soon!*

