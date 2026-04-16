# DAF: Detecting Diffusion-Generated Images via Dynamic Assembly Forests

**Mengxin Fu, Yuezun Li**

![](/home/lab/pycharmprograms/DAF/Overview.png)

## Introduction

This repository provides the official implementation of our paper:

> **DAF: Detecting Diffusion-Generated Images via Dynamic Assembly Forests**
> 
> 📄 [Paper Link (arXiv)]([[2604.09106] Detecting Diffusion-generated Images via Dynamic Assembly Forests](https://arxiv.org/abs/2604.09106))

### 🔹 Overview

We propose **Dynamic Assembly Forest (DAF)**, a lightweight model based on the deep forest paradigm for detecting diffusion-generated images.

Unlike conventional deep neural network (DNN) based methods, DAF:

- builds an adaptive ensemble structure for representation learning
- requires significantly fewer parameters and computation
- can be deployed without GPU

---

## Quick Start

### Environment

```
conda create -n daf python=3.9
conda activate daf
pip install -r requirements.txt
```

---

### Installation

```
git clone https://github.com/OUC-VAS/DAF.git
cd DAF
```

## ⚠️ Important: DeepForest Modification

This project requires modifications to the `deepforest` package.

Before running the code, please replace the following files in your local `deepforest` installation with the versions provided in this repository:

### Steps

1. Navigate to the package directory and replace the following files:

```
deepforest/
├── cascade.py   ← replace with our daf_cascade.py
├── _layer.py   ← replace with our daf_layer.py
```

3. You can find the modified files in this repository:

```
./build_DAF/
```

---

## Datasets

### Download

This project uses publicly available datasets:

- 👉 [DiffusionForensics(pre-processed version)]([GitHub - Chuchad/FIRE · GitHub](https://github.com/Chuchad/FIRE))
- 👉 [GenImage]([GitHub - GenImage-Dataset/GenImage · GitHub](https://github.com/GenImage-Dataset/GenImage))
- 👉 [Chameleon]([GitHub - shilinyan99/AIDE: 「ICLR 2025」 A Sanity Check for AI-generated Image Detection · GitHub](https://github.com/shilinyan99/AIDE))

> Please refer to the original repositories for detailed download instructions and make sure to follow their respective licenses and usage policies.

### Structure

To use this project, the dataset should be organized in the following structure:

```
dataset/
│── train/
│   ├── 0/   # real images
│   └── 1/   # diffusion-generated images
│── test/
│   ├── 0/   # real images
│   └── 1/   # diffusion-generated images
```

- **train/**: used for model training, containing both real and diffusion-generated images.
- **test/**: used for final evaluation.

> Each subset should follow the same structure, where `0/` and `1/` correspond to authentic and generated images, respectively.

> If you are using custom datasets, please convert them into this format before training or evaluation.

## Feature Extraction

Before training, the images need to be preprocessed using the provided feature extraction code.

### Step

1. Update the file paths in the code to match your local environment (e.g., dataset paths, output directories).
2. Run the following command to extract features:

```
python feature_extract.py
```

### Output

The extracted features will be saved to:

```
features/
│── train/
│   ├── 0/   
│   └── 1/   
│── test/
│   ├── 0/   
│   └── 1/   
```

These features are used as input to train the DAF model.

---

## Training

  After feature extraction, you can train the DAF model using the following command:

```
python train.py
```

### Notes

- Before training, update the file paths in the code:
  - Set the input path to the extracted training features.
  - Set the model saving path to your desired directory.
- Make sure the feature extraction step has been completed before training.

---

## Testing

To evaluate the trained model, run:

```
python test.py
```

### Notes

- Before testing, update the file paths in the code:
  - Set the model loading path to the saved model path.
  - Set the input path to the extracted test features.
- Ensure the saved model path exists and matches the training configuration.

---

## Pretrained Models

We provide pretrained models within this repository for reproducibility and evaluation.

### Structure

The pretrained models are organized as follows:

```
daf_pretrained/
│── imagenet_pretrained_daf/
│── lsun_pretrained_daf/
```

### Usage

Before testing, please update the model loading path in `test.py`:

```
model.load('save_model_path')
```

Then run:

```
python test.py
```

## 📖 Citation

If you find this work useful, please cite:

```
@misc{daf2026,
      title={Detecting Diffusion-generated Images via Dynamic Assembly ForestsDetecting Diffusion-generated Images via Dynamic Assembly Forests}, 
      author={Mengxin Fu and Yuezun Li},
      year={2026},
      eprint={2604.09106},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.09106}, 
}
```
