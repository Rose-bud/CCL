# Correspondence Cognitive Learning for Multi-Modal Object Re-Identification

Official implementation of **Correspondence Cognitive Learning (CCL)** for
multi-modal object re-identification.

CCL contains two correspondence-aware components:

- **Correspondence-Guided Semantic Refinement (CGSR)** refines visual features
  with text semantics according to correspondence difficulty estimated from
  the previous epoch.
- **Cognitive-Driven Dynamic Optimization (CDDO)** uses a self-paced objective
  to emphasize reliable image-text pairs and progressively learn from harder
  pairs.

The implementation supports RGBNT201, RGBNT100, and MSVR310. It is developed
from the public DeMo codebase; the original DeMo modules retained by CCL remain
under their original license and attribution.

## Preparation

Create a Python 3.8 environment and install the CUDA-matched PyTorch build
first:

```bash
conda create -n ccl python=3.8.18 -y
conda activate ccl
python -m pip install --upgrade pip
python -m pip install \
  torch==1.12.0+cu113 \
  torchvision==0.13.0+cu113 \
  torchaudio==0.12.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
```

Verify that PyTorch can access CUDA, then install the remaining environment
snapshot:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip install -r requirements.txt
```

We use the same datasets as [IDEA](https://github.com/924973292/IDEA). Please download the datasets following the instructions provided in the IDEA repository and download the pretrained CLIP weights, then set `DATASETS.ROOT_DIR`
and `MODEL.PRETRAIN_PATH` in the selected configuration file. The existing
configuration filenames are retained for compatibility:

- `configs/RGBNT201/DeMo.yml`
- `configs/RGBNT100/DeMo.yml`
- `configs/MSVR310/DeMo.yml`

## Training

Run one of the dataset scripts:

```bash
bash RGBNT201.sh
bash RGBNT100.sh
bash MSVR310.sh
```

or invoke the formal training entry directly:

```bash
python train_net.py --config_file configs/RGBNT201/DeMo.yml
```

Additional YACS options can be appended to either command. For example:

```bash
bash RGBNT201.sh MODEL.DEVICE_ID 0 OUTPUT_DIR ./outputs/RGBNT201
```

## Evaluation

Pass a trained checkpoint explicitly:

```bash
python test_net.py \
  --config_file configs/RGBNT201/DeMo.yml \
  --weight ./outputs/RGBNT201/CCL_best.pth
```

Alternatively, set `TEST.WEIGHT` in the dataset configuration.

## Citation

```bibtex
@inproceedings{su2026ccl,
  title={Correspondence Cognitive Learning for Multi-Modal Object Re-Identification},
  author={Su, Chao and Li, Shuying and Pu, Ruitao and Peng, Dezhong and Ren, Zhenwen and Sun, Yuan},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}
```
