import fire, sys, cv2, os, torch, imageio
import gradio as gr
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from contextlib import nullcontext
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from functools import partial
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from transformers import AutoFeatureExtractor #, CLIPImageProcessor
from torch import autocast
from torchvision import transforms
from groundingdino.util.inference import load_model, predict, annotate
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops
from segment_anything import build_sam, SamPredictor
import matplotlib.pyplot as plt
import pdb

device = 'cuda'
sam_checkpoint = 'sam_vit_h_4b8939.pth'
sam = build_sam(checkpoint=sam_checkpoint).to("cuda")
sam_predictor = SamPredictor(sam)
SAM_CONFIG_PATH = os.path.join("./GroundingDino/GroundingDINO_SwinT_OGC.py")
SAM_WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
SAM_WEIGHTS_PATH = SAM_WEIGHTS_NAME
sam_model = load_model(SAM_CONFIG_PATH, SAM_WEIGHTS_PATH)
TEXT_PROMPT = "hands"
CONFIG = None
_SHOW_DESC = True
_SHOW_INTERMEDIATE = False
# _SHOW_INTERMEDIATE = True
_GPU_INDEX = 0
# _GPU_INDEX = 2

# _TITLE = 'Zero-Shot Control of Camera Viewpoints within a Single Image'
_TITLE = '2024-02-04T14-06-22_jgd_stride_10_cond_1'

# This demo allows you to generate novel viewpoints of an object depicted in an input image using a fine-tuned version of Stable Diffusion.
_DESCRIPTION = '''
Instructions:
1. Add a short clip of a hand moving an object, click "Split into Frames"
2. Choose which frames you'd like as input and output. Tip: works best with small movement changes
3. Click "Get Hand Masks" to generate hand masks of each frame
4. Click "Get Next Frame" to see generated output based on the input frame, input handmask, and desired final handmask
'''

_ARTICLE = 'See uses.md'
def load_image(input_img_path):
    
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(input_img_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def get_image_masks(input_img_path, bbox_threshold, text_threshold):
    image_source, image = load_image(input_img_path) # image_source = 0:255 224x336x3 image. image = -2:2.6 3x224x336
    boxes, logits, phrases = predict(model=sam_model, image=image, caption=TEXT_PROMPT, box_threshold=bbox_threshold, text_threshold=text_threshold) #takes in image 3xhxw
    try:
        assert boxes.shape[0]>0
    except:
        return None 
        # raise Exception("No hand in image")
    sam_predictor.set_image(image_source) #expect 224,224,3 takes in imagesource
    # box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
    mask, _, _ = sam_predictor.predict_torch(point_coords = None,point_labels = None,boxes = transformed_boxes,multimask_output = False)
    final_mask=mask[0]
    if mask.shape[0]!=1:
        for i in range(mask.shape[0]-1):
            final_mask += mask[i+1]
    final_mask=final_mask[0,:,:]
    final_mask = final_mask.cpu().numpy().astype(np.uint8)*255        
    return final_mask

def load_model_from_config(config, ckpt, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)
    model.to(device)
    model.eval()
    return model

def resize_image(path, target_width, target_height):
    img = plt.imread(path)
    if len(img.shape)==2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    # Calculate aspect ratios
    original_height, original_width = img.shape[0:2]
    original_aspect = original_width / original_height
    target_aspect = target_width / target_height

    # Crop to correct aspect ratio
    if original_aspect > target_aspect:
        # Crop width
        new_width = int(original_height * target_aspect)
        start = (original_width - new_width) // 2
        cropped_img = img[:, start: start + new_width, :]
    else:
        # Crop height
        new_height = int(original_width / target_aspect)
        start = (original_height - new_height) // 2
        cropped_img = img[start: start + new_height, :, :]

    # Resize video
    # resized_img = cv2.resize(cropped_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    # # print(lo(resized_img))
    # resized_img = resized_img / 255.0 #(we commented this out because its already in the 0-1 scale somehow)
    # # print(lo(resized_img))
    # return resized_img

    input_im = Image.fromarray((cropped_img *255).astype(np.uint8))
    input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
    return input_im


@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, all_cond_images_and_handmasks):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            input_im_encoded = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            c = model.cc_projection(input_im_encoded)
            
            cond = {}
            cond['c_crossattn'] = [c]
            all_cond_images_and_handmasks_encoded = torch.zeros(1,2*int((CONFIG.model.params.unet_config.params.in_channels-8)/4/2)*4 + 4, 32, 32) #max 5 frames of cond, with 1 image and 1 handmask per frame. 4 channels per image once encoded
            for idx in range(int(2*(CONFIG.model.params.unet_config.params.in_channels-8)/4/2) + 1):
                if all_cond_images_and_handmasks[idx*3:idx*3+3,:,:].to(device).all() != 0:
                    all_cond_images_and_handmasks_encoded[:,idx*4:idx*4+4, :, :] = model.encode_first_stage(all_cond_images_and_handmasks[None,idx*3:idx*3+3,:,:].to(device)).mode().detach()
                else:
                    all_cond_images_and_handmasks_encoded[:,idx*4:idx*4+4, :, :] = torch.zeros(1, 4, 32, 32)
            c_concat = all_cond_images_and_handmasks_encoded.to(device)
            cond['c_concat'] = [c_concat.repeat(n_samples, 1, 1, 1)]
            
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, CONFIG.model.params.unet_config.params.in_channels-4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            x_samples_ddim = model.decode_first_stage(samples_ddim)

            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()
    
def main_run(model, gallery_frames, cond_image1, hm_frames, scale=1.5, ddim_steps=50, target_width=256, target_height=256):
    '''
    :param raw_im (PIL Image).
    '''
    torch.cuda.empty_cache()
    sam.to(device)
    model.to(device)
    torch_transforms = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'h w c-> h w c')),
    ])
    cond_images=[cond_image1]
    all_cond_images_and_handmasks = torch.zeros((3*2*len(cond_images) + 3,256,256)) #last +3 is for the target handmask
    for idx in range(len(cond_images)):
        hand_masks0 = resize_image(hm_frames[idx]['name'], target_width, target_height)
        all_cond_images_and_handmasks[idx*6:idx*6+3,:,:] = torch_transforms(hand_masks0).unsqueeze(0).to(device)


        input_im = resize_image(gallery_frames[int(cond_images[idx])-1][0]['name'], target_width, target_height)
        all_cond_images_and_handmasks[idx*6+3:idx*6+6,:,:] = torch_transforms(input_im).unsqueeze(0).to(device)

        # hand_masks0=(hand_masks0*255.).astype(np.uint8)
        # input_im=(input_im*255.).astype(np.uint8)


    final_hm = resize_image(hm_frames[idx+1]['name'], target_width, target_height)
    all_cond_images_and_handmasks[-3:,:,:] = torch_transforms(final_hm).unsqueeze(0).to(device)
    # final_hm=(final_hm*255.).astype(np.uint8)

    sampler = DDIMSampler(model)
    x_samples_ddim = sample_model(torch_transforms(input_im).unsqueeze(0).to(device), model, sampler, 'fp32', target_height, target_width, ddim_steps, 4, scale, 1.0, all_cond_images_and_handmasks)
    
    output_ims=[]
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

    with imageio.get_writer(f'handmask.mp4', mode='I') as writer:
        for img in [hand_masks0, hand_masks0, hand_masks0, hand_masks0, hand_masks0, final_hm,final_hm,final_hm,final_hm,final_hm]:
            writer.append_data(np.array(img))

    for i in range(len(output_ims)):
        with imageio.get_writer(f'movie_{i}.mp4', mode='I') as writer:
            for img in [input_im, input_im, input_im, input_im, input_im, output_ims[i], output_ims[i], output_ims[i], output_ims[i], output_ims[i]]:
                writer.append_data(np.array(img))
    

    return input_im, 'handmask.mp4', 'movie_0.mp4', 'movie_1.mp4', 'movie_2.mp4', 'movie_3.mp4'

def split_into_frames_and_get_handmasks(video_block, num_frames=15, bbt=0.5, ttt=0.5):
    vidcap = cv2.VideoCapture(video_block)
    take_indices = int(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))/num_frames)

    frames = []
    # handmasks = []
    success,image = vidcap.read()
    count = 0
    while success:
        if count%take_indices==0:
            if image.shape[-1]==3:
                image=image[:,:,::-1]
            else:
                image=image[::-1,:,:]
            frames.append((image,str(int(count/take_indices)+1)))    # save frame as JPEG file  
            # handmasks.append(get_image_masks(image, bbt, ttt)) 
            # plt.imsave('temp.png',handmasks[-1])
        success,image = vidcap.read()
        count += 1
    return frames, gr.update(visible=True), gr.update(visible=False)

def get_handmasks(gallery, cond1, output2, bbox_threshold, text_threshold):
    hm1 = get_image_masks(gallery[int(cond1)-1][0]['name'], bbox_threshold, text_threshold)
    hm_output = get_image_masks(gallery[int(output2)-1][0]['name'], bbox_threshold, text_threshold)
    outputs = [hm1, hm_output]
    errors = []
    for i in range(len(outputs)):
        if outputs[i] is None:
            errors.append(f'{i}')
    if len(errors) > 0:
        raise gr.Error(f"failed to get handmask for {errors} frame(s). try with different advanced parameters")
    return outputs, gr.update(visible=True)

def run_demo(
        ckpt='/proj/vondrick3/sruthi/projeccv24/zero123jgd/CosHand/logs/2024-02-04T14-06-22_jgd_stride_10_cond_1/checkpoints/trainstep_checkpoints/epoch=000010-step=000038999.ckpt',
        config='/proj/vondrick3/sruthi/projeccv24/zero123jgd/CosHand/logs/2024-02-04T14-06-22_jgd_stride_10_cond_1/configs/2024-02-04T14-06-22-project.yaml'):
    
    config = OmegaConf.load(config)
    global CONFIG
    CONFIG = config

    # Instantiate all models beforehand for efficiency.
    model = dict()
    model = load_model_from_config(config, ckpt)

    # Compose demo layout & data flow.
    demo = gr.Blocks(title=_TITLE)

    with demo:
        gr.Markdown('# ' + _TITLE)
        gr.Markdown(_DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=0.9, variant='panel'):

                video_block = gr.Video(label='Input video of acting upon an objext')

                gr.Markdown('*Number of Frames:*')
                frames_slider = gr.Slider(1, 100, value=15, step=5, label='Number of Frames')

                with gr.Row():
                    split_into_frames_btn = gr.Button('Split into Frames', variant='primary')

                gr.Markdown('Split frames', visible=_SHOW_DESC)
                
                frames_outputs = gr.Gallery(label='Split Frames', elem_id="gallery")
                frames_outputs.style(grid=5)


                #handmask stuff
                gr.Markdown('*Hand Mask Parameters:*')
                first_frame = gr.Dropdown([str(x) for x in range(1, 100)], label="1st conditioning Frame")
                final_frame = gr.Dropdown([str(x) for x in range(1, 100)], label="Desired Output Frame")
                
                with gr.Accordion('Advanced Parameters', open=False):
                    text_threshold = gr.Slider(0, 1, value=0.5, step=0.1, label='Text SAM threshold of masking')
                    bbox_threshold = gr.Slider(0, 1, value=0.5, step=0.1, label='BBox SAM threshold of masking')

                with gr.Row(visible=False) as view_handmasks:
                    get_handmasks_button = gr.Button('Get handmasks of the frames', variant='primary')
                
                gr.Markdown('*If handmasks look good, proceed to generating next frame output. Else re-run handmasks with different settings in Advanced Parameters:*')
                handmask_outputs = gr.Gallery(label='Hadnmasked Frames')
                handmask_outputs.style(grid=2)
                    

                #generate samples stuff
                
                with gr.Accordion('Generate Next Frame: Advanced options', open=False):
                    # num_samples = gr.Slider(1, 9, value=4, step=1, label='Number of Samples')
                    scale_slider = gr.Slider(0, 30, value=1.5, step=0.5, label='Diffusion guidance scale')
                    steps_slider = gr.Slider(5, 200, value=50, step=5, label='Number of diffusion inference steps')

                with gr.Row(visible=False) as generate_samples:
                    get_samples_btn = gr.Button('Generate Next Frame', variant='primary')
                
                gr.Markdown('The results will appear on the right.')

            with gr.Column(scale=1.1, variant='panel'):
                with gr.Row():
                    input_frame = gr.Image(label='Input Frame', interactive=False)
                    hm_mp4 = gr.Video(label='Input/Output hand motions', interactive=False)
                with gr.Row():
                    sample1= gr.Video(label=f'sample 1', interactive=False)
                    sample2= gr.Video(label=f'sample 2', interactive=False)
                with gr.Row():
                    sample3= gr.Video(label=f'sample 3', interactive=False)
                    sample4= gr.Video(label=f'sample 4', interactive=False)
                # with gr.Row():
                #     save_text = gr.Textbox(label="Save Text")
                #     save_frames = gr.Button('Save files', variant='primary')
        # gr.Markdown("## Text Examples")
        # gr.Examples(
        #     [["hi", "Adam"], ["hello", "Eve"]],
        #     [video_block, frames_outputs, handmask_outputs],
        #     [hm_mp4, sample1, sample2, sample3, sample4],
        #     main_run,
        #     cache_examples=True,
        # )

        split_into_frames_btn.click(fn=partial(split_into_frames_and_get_handmasks),
                      inputs=[video_block, frames_slider, bbox_threshold, text_threshold],
                      outputs=[frames_outputs, view_handmasks, generate_samples])
        get_handmasks_button.click(fn=partial(get_handmasks),
                      inputs=[frames_outputs, first_frame, final_frame, bbox_threshold, text_threshold],
                      outputs=[handmask_outputs, generate_samples])
        get_samples_btn.click(fn=partial(main_run, model),
                      inputs=[frames_outputs, first_frame, handmask_outputs, scale_slider, steps_slider],
                      outputs=[input_frame, hm_mp4, sample1, sample2, sample3, sample4])
        # save_frames.click(fn=partial(save_videos),
        #               inputs=[save_text],
        #               outputs=[])
    demo.launch(enable_queue=True, share=True)


if __name__ == '__main__':

    fire.Fire(run_demo)
