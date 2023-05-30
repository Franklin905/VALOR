# Modality-Independent Teachers Meet Weakly-Supervised Audio-Visual Event Parser

This is the official repository of VALOR.

![VALOR](./figures/framework_figure.png)

**Modality-Independent Teachers Meet Weakly-Supervised Audio-Visual Event Parser**
<br/>[Yung-Hsuan Lai](https://github.com/Franklin905), [Yen-Chun Chen](https://github.com/ChenRocks), Yu-Chiang Frank Wang<br/>

---
## TODO

- [ ] Feature extraction code
- [ ] Pseudo labels generation code

---
## Machine environment
- Ubuntu version: 20.04.6 LTS
- CUDA version: 11.4
- Testing GPU: NVIDIA GeForce RTX 3090
---

## Requirements
A [conda](https://conda.io/) environment named `valor` can be created and activated with:

```bash
conda env create -f environment.yaml
conda activate valor
```
---

## Dataset setup
### LLP dataset annotations
Please download LLP dataset annotations from [AVVP-ECCV20](https://github.com/YapengTian/AVVP-ECCV20) and put in data/

### Pre-extracted features
Please download audio features (VGGish), 2D visual features (ResNet152), and 3D visual features (ResNet (2+1)D) from [AVVP-ECCV20](https://github.com/YapengTian/AVVP-ECCV20) and put in data/feats/

### CLIP features & segment-level pseudo labels
Please download visual features and segment-level pseudo labels from CLIP from this Google Drive [link](https://drive.google.com/file/d/113QVJtvLf1Qdbz3P2Z2kG3aCi_x0ipjY/view?usp=sharing), put in data/, and unzip the file with the following command:
```
unzip CLIP.zip
```

### CLAP features & segment-level pseudo labels
Please download audio features and segment-level pseudo labels from CLAP from this Google Drive [link](https://drive.google.com/file/d/17ErPglRS7Yzm93aF_3WTlfZuSyw12QQA/view?usp=sharing), put in data/, and unzip the file with the following command:
```
unzip CLAP.zip
```

### File structure for dataset and code
Please make sure that the file structure is the same as the following.

   <details><summary>File structure</summary>

   ```
   > data/
       ├── AVVP_dataset_full.csv
       ├── AVVP_eval_audio.csv
       ├── AVVP_eval_visual.csv
       ├── AVVP_test_pd.csv
       ├── AVVP_train.csv
       ├── AVVP_val_pd.csv
       ├── feats/
       │     └── r2plus1d_18
       │     └── res152
       │     └── vggish
       ├── CLIP/
       │     └── features
       │     └── segment_pseudo_labels
       └── CLAP/
             └── features
             └── segment_pseudo_labels
   ```

   </details>

---

## Download trained models
Please download the trained models from this Google Drive [link](https://drive.google.com/drive/folders/1BDBZ3Ws75yzemIQrnBhBFfO4XD3zsEGR?usp=sharing) and put the models in their corresponding model directory.

   <details><summary>File structure</summary>

   ```
   > models/
       ├── model_VALOR/
       │        └── checkpoint_best.pt
       ├── model_VALOR+/
       │        └── checkpoint_best.pt
       └── model_VALOR++/
                └── checkpoint_best.pt
   ```

   </details>

---

## Training
We provide some sample scripts for training VALOR, VALOR+, and VALOR++.

### VALOR
```bash
bash scripts/train_valor.sh
```

### VALOR+

```bash
bash scripts/train_valor+.sh
```

### VALOR++

```bash
bash scripts/train_valor++.sh
```

---

## Evaluation
### VALOR
```bash
bash scripts/test_valor.sh
```

### VALOR+

```bash
bash scripts/test_valor+.sh
```

### VALOR++

```bash
bash scripts/test_valor++.sh
```


## Acknowledgement
We build VALOR codebase heavily on the codebase of [AVVP-ECCV20](https://github.com/YapengTian/AVVP-ECCV20) and [MM-Pyramid](https://github.com/JustinYuu/MM_Pyramid). We sincerely thank the authors for open-sourcing! 

## Citation
If you find this code useful for your research, please consider citing:
```bibtex
@article{lai2023modality,
  title={Modality-Independent Teachers Meet Weakly-Supervised Audio-Visual Event Parser},
  author={Yung-Hsuan Lai, Yen-Chun Chen, Yu-Chiang Frank Wang},
  journal={arXiv preprint arXiv:2305.17343},
  year={2023}
}
```

## License
This project is released under the MIT License.
