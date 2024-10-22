# CosHand: Controlling the World by Sleight of Hand
ECCV 2024 (Oral, Best Paper Candidate) <br>
[CosHand: Controlling the World by Sleight of Hand](https://coshand.cs.columbia.edu/) <br>
Sruthi Sudhakar, Ruoshi Liu, Basile Van Hoorick, Carl Vondrick, Richard Zemel <br>
Columbia University <br>

## Installation
### 1. Clone the repository
```
git clone https://github.com/SruthiSudhakar/CosHand.git
cd CosHand

conda create -n coshand python=3.9
conda activate coshand

pip install -r requirements.txt

git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/
```

## Usage
Download the config and checkpoint here:
```
wget https://cv.cs.columbia.edu/zero123/assets/coshandrelease_config.yaml    
wget https://cv.cs.columbia.edu/zero123/assets/coshandrelease.ckpt    
```

## Qualitative Evaluation
change the paths inside this file
```
python run_results_on_final_eval_set.py
```
## Gradio demo
change the paths inside this file
```
python gradio.py
```

## Dataset Structure

structure of dataset folder for data.params.root_dir in cfg below:
```
FullSSv2
│
├── data
│   ├── handmasks
│   │   ├── video_id_1
│   │   │   ├── image_0001.jpg
│   │   │   ├── image_0002.jpg
│   │   │   └── ...
│   │   ├── video_id_2
│   │   ├── video_id_3
│   │   └── ...
│   ├── rawframes
│   │   ├── video_id_1
│   │   │   ├── image_0001.jpg
│   │   │   ├── image_0002.jpg
│   │   │   └── ...
│   │   ├── video_id_2
│   │   ├── video_id_3
│   │   └── ...
```
## Training
Download image-conditioned stable diffusion checkpoint released by Lambda Labs: <br>
wget https://cv.cs.columbia.edu/zero123/assets/sd-image-conditioned-v2.ckpt

```
python main.py \
    -t \
    --base configs/sd-somethingsomething-finetune.yaml \
    --gpus 0,1,2,3, \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 10 \
    --finetune_from /path/to/sd-image-conditioned-v2.ckpt \
    --name name_of_run \
    data.params.root_dir=/path/to/dataset 
```

## Acknowledgement
This repository is based on Zero123 and Stable Diffusion. We would like to thank the authors of these work for publicly releasing their code. 

This project is based on research partially supported by NSF #2202578, NSF #1925157, and the National AI Institute for Artificial and Natural Intelligence (ARNI). SS is supported by the NSF GRFP.

## Citation
```
@misc{sudhakar2024controllingworldsleighthand,
        title={Controlling the World by Sleight of Hand}, 
        author={Sruthi Sudhakar and Ruoshi Liu and Basile Van Hoorick and Carl Vondrick and Richard Zemel},
        year={2024},
        eprint={2408.07147},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2408.07147}, 
}
```
