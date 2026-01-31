# Training-Efficient Text-to-Music Generation with State-Space Modeling

Official implementation of  
**"Training-Efficient Text-to-Music Generation with State-Space Modeling"**.

🚨 **Code Release Status**  

- [x] Training and inference code update (2026/02)
- [ ] Implementation of data preprocessing
<!-- - [ ]  -->

# 1. Env setting

Please download the causal-conv-1d and mamba-ssm first.

Then follow the

```
conda create -n ssm_ttm python=3.12
conda activate ssm_ttm
pip install -r requirements.txt
```

# 2. Training

```
python main_pl.py \
--project_name [exp name] \
--root_path [dataset root path] \
--layer_num [layers number]
```

# 3. Inference

Please download the ckpt and config from the [Google Cloude](https://drive.google.com/drive/folders/1O3uIUAMx12Y2VsI-Y5zdtwdYCr2PfgiU?usp=sharing).

You can edit the `captions` at `line 77` in `pl_inference.py`.

```
python pl_inference.py \
--model_path [model_ckpt_path] \
--config_path [model_config_path] \
--save_dir_name [save_dir_name]
```

The output will be saved in `./outputs/[save_dir_name]`.

Note that we only support to generate the DAC audio tokens, so you may need to decode into audio by [DAC](https://github.com/descriptinc/descript-audio-codec).