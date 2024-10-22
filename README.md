Installation:
1. git clone 

structure of dataset folder for data.params.root_dir in cfg above:
FullSSv2
    |__data
        |__handmasks
                |__video_id_1
                        |__image_0001.jpg
                        |__image_0002.jpg
                        |__image_0003.jpg
                        ...
                |__video_id_2
                |__video_id_3
                ...
        |__rawframes
                |__video_id_1
                        |__image_0001.jpg
                        |__image_0002.jpg
                        |__image_0003.jpg
                        ...
                |__video_id_2
                |__video_id_3
                ...

training:

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

qualitative evaluation: change the paths inside this file

python run_results_on_final_eval_set.py

gradio demo: change the paths inside this file
python gradio.py