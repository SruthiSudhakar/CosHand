# CosHand

## Controlling the World by Sleight of Hand
Official Repo for CosHand: Controlling the World by Sleight of Hand

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/SruthiSudhakar/CosHand.git
cd CosHand
conda create -n coshand python==3.9.19
pip install -r requirements.txt

structure of dataset folder for data.params.root_dir in cfg above:
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

## Training

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

## Qualitative Evaluation
change the paths inside this file

python run_results_on_final_eval_set.py

## gradio demo
change the paths inside this file

python gradio.py
