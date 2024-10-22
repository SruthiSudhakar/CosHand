import os, json, pdb
from tqdm import tqdm
root_dir='/proj/vondrick3/datasets/FullSSv2/data/rawframes'
labels_dir='/proj/vondrick3/datasets/FullSSv2/labels'

labels = json.load(open(labels_dir+'/train.json'))
data = []
count=0
total_labels_size = len(labels)
# notthere=[]
for x in tqdm(labels):
    if os.path.exists(os.path.join(root_dir, x['id'], 'handmasks')):
    #     count+=1
    # else:
    #     notthere.append(x['id'])
# print(count)
        # print('ITS THERE')
        paths_with_masks = os.listdir(os.path.join(root_dir, x['id'], 'handmasks'))
        paths_with_masks.sort()
        if len(paths_with_masks)>10:
            x["frames_list"] = paths_with_masks
            data.append(x)
json.dump(data, open("gathere_train.json", "w"))
# json.dump(notthere, open("train_notthere.json", "w"))