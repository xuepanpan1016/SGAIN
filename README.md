# A global-to-local synergistic diffusion transformer with adaptive feature interaction for structured image generation

## Requirements

- python 3.12.3
- Pytorch 2.2.2
- vGPU-32GB * 1 or stronger GPUs

## Installation

Clone this repo.

```javascript
cd GAIN-Diffusion
conda env create -f environment.yml
conda activate GAIN
```
If the cuda version doesn't match your GPUs, see [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for a suitable one

### Datasets Preparation

1. If you would like to request our T2IThangka dataset, please contact  [wenjin_zhm@126.com](mailto:wenjin_zhm@126.com).
2. Download the [flower_dataset](https://drive.google.com/file/d/1cL0F5Q3AYLfwWY7OrUaV1YmTx4zJXgNG/view) image data and extract them to dataset/flower/dataset.

## Training on data
```javascript
python train.py
```
## Sampling
```javascript
python sample_simple.py
```
## Evaluating

```javascript
torchrun --nnodes=1 --nproc_per_node=1 --master_port=9902 --data-path='path to dataset' sample_ddp.py
```

### Evaluate models
- FID
`python fid.py --gpu 0 --path1 /root/(path to test set images) --path2 /root/(path to generated images)`
- CLIPScore
`bash eval_clipscore.sh`
- LPIPS
`bash eval_lpips.sh`

### Reference
- [DATA EXTRAPOLATION FOR TEXT-TO-IMAGE GENER-ATION ON SMALL DATASETS](https://arxiv.org/abs/2410.01638),[[code]](https://github.com/senmaoy/RAT-Diffusion).

