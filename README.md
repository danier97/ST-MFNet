# ST-MFNet: A Spatio-Temporal Multi-Flow Network for Frame Interpolation

### Duolikun Danier, Fan Zhang, David Bull
### Accepted in CVPR 2022.

[Project](https://danielism97.github.io/ST-MFNet) | [Paper](https://arxiv.org/abs/2111.15483) | [Video](https://drive.google.com/file/d/1zpE3rCQNJi4e8ADNWKbJA5wTvPllKZSj/view)


## Dependencies and Installation
The following packages were used to evaluate the model.

- python==3.8.8
- pytorch==1.7.1
- torchvision==0.8.2
- cudatoolkit==10.1.243
- opencv-python==4.5.1.48
- numpy==1.19.2
- pillow==8.1.2
- cupy==9.0.0

Installation with anaconda:

```
conda create -n stmfnet python=3.8.8
conda activate stmfnet
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
conda install -c conda-forge cupy
pip install opencv-python==4.5.1.48
```

## Model
<img src="https://danielism97.github.io/ST-MFNet/overall.svg" alt="Paper" width="60%">

## Preparing test data

- Download UCF101 quintuplets from [here](https://sites.google.com/view/xiangyuxu/qvi_nips19).
- Download DAVIS sequences from [here](https://sites.google.com/view/xiangyuxu/qvi_nips19).
- Download SNU-FILM dataset from [here](https://myungsub.github.io/CAIN/).
- Download VFITex dataset from [here](https://uob-my.sharepoint.com/:f:/g/personal/mt20523_bristol_ac_uk/EsRvHziGSA9BpQ7J02GO9PoBpVzoXFlrHHjwHCYYAsDIOQ?e=gYxehR).

The dataset folder names should be lower-case and structured as follows.
```
└──── <data directory>/
    ├──── ucf101/
    |   ├──── 0/
    |   ├──── 1/
    |   ├──── ...
    |   └──── 99/
    ├──── davis90/
    |   ├──── bear/
    |   ├──── bike-packing/
    |   ├──── ...
    |   └──── walking/
    ├──── snufilm/
    |   ├──── test-easy/
    |   ├──── test-medium/
    |   ├──── test-hard/
    |   ├──── test-extreme/
    |   └──── data/
    └──── vfitex/
        ├──── beach02_4K_mitch/
        ├──── bluewater_4K_pexels/
        ├──── ...
        └──── waterfall_4K_pexels/

```

## Downloading the pre-trained model
Download the pre-trained ST-MFNet from [here](https://drive.google.com/file/d/1s5JJdt5X69AO2E2uuaes17aPwlWIQagG/view?usp=sharing).

## Evaluation
```
python evaluate.py \
--net STMFNet \
--data_dir <data directory> \
--checkpoint <path to pre-trained model (.pth file)> \
--out_dir eval_results \
--dataset <dataset name>
```
where `<dataset name>` should be the same as the class names defined in `data/testsets.py`, e.g. `Snufilm_extreme_quintuplet`.

## Example results
<img src="https://danielism97.github.io/ST-MFNet/qualitative.png" alt="Paper" width="100%">

## Citation
```
@misc{danier2021spatiotemporal,
     title={ST-MFNet: A Spatio-Temporal Multi-Flow Network for Frame Interpolation}, 
     author={Duolikun Danier and Fan Zhang and David Bull},
     year={2021},
     eprint={2111.15483},
     archivePrefix={arXiv},
     primaryClass={cs.CV}
}
```

## Acknowledgement
Lots of code in this repository are adapted/taken from the following repositories:

- [AdaCoF-pytorch](https://github.com/HyeongminLEE/AdaCoF-pytorch)
- [softmax-splatting](https://github.com/sniklaus/softmax-splatting) 
- [pytorch-pwc](https://github.com/sniklaus/pytorch-pwc)

We would like to thank the authors for sharing their code.
