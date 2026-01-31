# Data Preprocessing for Jamendo Dataset

This document describes the complete data preprocessing pipeline used in this project. The goal is to transform the raw **MTG-Jamendo** audio dataset into model-ready audio tokens with aligned captions.

The pipeline consists of the following steps:

1. Downloading Jamendo audio
2. Audio segmentation
3. Source separation (vocal removal)
4. Audio tokenization using DAC

---

## 1. Download Jamendo Audio

Please refer to the official **MTG-Jamendo** repository for dataset access and download instructions:

* GitHub: [https://github.com/MTG/mtg-jamendo-dataset](https://github.com/MTG/mtg-jamendo-dataset)
* Website: [https://mtg.github.io/mtg-jamendo-dataset/](https://mtg.github.io/mtg-jamendo-dataset/)

In this project, we use **all audio files listed in `raw_30s_cleantags.tsv`** as the full Jamendo dataset.

### Expected Directory Structure

After downloading and extracting the audio, the dataset should be organized as follows:

```
Jamendo-full/
├── 00/
│   ├── 1100.mp3
│   ├── 3100.mp3
│   └── ...
├── 01/
├── .../
└── 99/
```

Each subfolder (`00`–`99`) contains MP3 files indexed by `song_id`.

---

## 2. Audio Segmentation

Each full-length song is segmented into **30-second clips**. All segments belonging to the same song are stored under a directory named by `song_id`. Each clip is named using a sequential `segment_id`.

### Output Directory Structure

```
Jamendo-segments/
├── 00/
│   ├── 1100/
│   │   ├── segment_0.mp3
│   │   ├── segment_1.mp3
│   │   ├── ...
│   │   └── segment_9.mp3
│   ├── 3100/
│   │   ├── segment_0.mp3
│   │   ├── segment_1.mp3
│   │   ├── ...
│   │   └── segment_5.mp3
│   └── ...
├── 01/
├── .../
└── 99/
```

**Notes**:

* Segment length is fixed to 30 seconds
* The number of segments per song depends on its duration

---

## 3. Source Separation (Vocal Removal)

To focus on instrumental music generation, we remove vocals from each audio segment using **Demucs**.

### Tool

* Demucs repository: [https://github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs)
* Model used: `htdemucs`

We apply Demucs to each 30-second segment and keep the **no-vocal (instrumental)** track.

### Output Directory Structure

```
Jamendo-segments-novocal/
├── 00/
│   ├── 1100/
│   │   ├── segment_0.mp3
│   │   ├── segment_1.mp3
│   │   ├── ...
│   │   └── segment_9.mp3
│   ├── 3100/
│   │   ├── segment_0.mp3
│   │   ├── segment_1.mp3
│   │   ├── ...
│   │   └── segment_5.mp3
│   └── ...
├── 01/
├── .../
└── 99/
```

**Notes**:

* Only instrumental tracks are kept
* Directory structure mirrors `Jamendo-segments`

---

## 4. Convert Audio to DAC Tokens

After source separation, the instrumental audio clips are converted into **DAC (Discrete Audio Codec) tokens**, which are used as model inputs for training and generation.

### Key Steps

* Load the no-vocal audio clips
* Resample to 44100 Hz if required (depending on DAC configuration)
* Encode audio into discrete token sequences using a pretrained DAC model
* Save token sequences for efficient loading during training


### Output Directory Structure

```
Jamendo-segments-novocal-dactoken/
├── 00/
│   ├── 1100/
│   │   ├── segment_0.npy
│   │   ├── segment_1.npy
│   │   ├── ...
│   │   └── segment_9.npy
│   ├── 3100/
│   │   ├── segment_0.npy
│   │   ├── segment_1.npy
│   │   ├── ...
│   │   └── segment_5.npy
│   └── ...
├── 01/
├── .../
└── 99/
```

---

## Alternative: Custom Caption Pipelines

If you plan to use a customized caption format, you may reuse this audio pipeline while replacing or extending the caption preprocessing stage or checkout this [repo](https://github.com/fundwotsai2001/Text-to-music-dataset-preparation) for impelementation.

---

## TODO

The following implementation scripts will be provided in the future:

* [ ] Download Jamendo audio automatically
* [ ] Audio cropping / segmentation
* [ ] Source separation with Demucs
* [ ] DAC token conversion

---

## Acknowledgements

* MTG-Jamendo Dataset
* Demucs (Facebook Research)
* DAC: Discrete Audio Codec

If you find this preprocessing pipeline useful, please consider citing the relevant datasets and tools.
