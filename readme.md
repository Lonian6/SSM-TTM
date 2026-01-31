# Training-Efficient Text-to-Music Generation with State-Space Modeling

Official implementation of the paper:

**Training-Efficient Text-to-Music Generation with State-Space Modeling**

---

## 🚨 Code Release Status

* [x] Training and inference code released (2026/02)
* [ ] Data preprocessing pipeline (coming soon)

---

## 1. Environment Setup

This project depends on **Mamba-SSM** and **causal-conv-1d**. Please install them before setting up the Python environment.

### Create Conda Environment

```
conda create -n ssm_ttm python=3.12
conda activate ssm_ttm
pip install -r requirements.txt
```

> ⚠️ Make sure your CUDA / PyTorch versions are compatible with Mamba-SSM.

---

## 2. Training

To train the text-to-music model, run the following command:

```
python main_pl.py \
  --project_name [experiment_name] \
  --root_path [dataset_root_path] \
  --layer_num [number_of_layers]
```

### Arguments

* `project_name`: Name of the experiment (used for logging and checkpoints)
* `root_path`: Root directory of the preprocessed dataset
* `layer_num`: Number of SSM-based layers in the model

---

## 3. Inference

### Download Pretrained Checkpoints

Please download the pretrained checkpoint and configuration files from Google Drive:

👉 [https://drive.google.com/drive/folders/1O3uIUAMx12Y2VsI-Y5zdtwdYCr2PfgiU](https://drive.google.com/drive/folders/1O3uIUAMx12Y2VsI-Y5zdtwdYCr2PfgiU)

### Edit Text Prompts

You can edit the input text captions at **line 77** in `pl_inference.py`.

### Run Inference

```
python pl_inference.py \
  --model_path [model_ckpt_path] \
  --config_path [model_config_path] \
  --save_dir_name [output_directory_name]
```

The generated outputs will be saved to:

```
./outputs/[output_directory_name]
```

---

## 4. Audio Decoding

This repository **only generates DAC audio tokens**. To convert DAC tokens into waveforms, please use the official DAC decoder:

* DAC repository: [https://github.com/descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec)

Make sure the DAC configuration matches the one used during training.

---

## Notes

* Currently, only DAC-token-level generation is supported
* Data preprocessing scripts will be released in a future update

---

## Citation

If you find this work useful, please consider citing our paper (BibTeX will be released upon publication).