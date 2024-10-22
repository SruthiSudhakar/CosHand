# CosHand: Controlling the World by Sleight of Hand
ECCV 2024, Oral, Best Paper Candidate
[CosHand: Controlling the World by Sleight of Hand](https://coshand.cs.columbia.edu/)
Sruthi Sudhakar, Ruoshi Liu, Basile Van Hoorick, Carl Vondrick, Richard Zemel
Columbia University

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

## Installation

structure of dataset folder for data.params.root_dir in cfg above:
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
    data.params.root_dir=/path/to/dataset \
    data.params.max_number_of_conditioning_frames=5 \
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
