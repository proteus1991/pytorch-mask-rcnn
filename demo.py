import os
import sys
import time
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import coco
import utils
import model as modellib
import visualize

import torch

load_start = time.time()
# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "CarWithGroundEvalImages")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

##########################################3
config = InferenceConfig()
config.display()

# Create model object.
model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
if config.GPU_COUNT:
    model = model.cuda()

# Load weights trained on MS-COCO
model.load_state_dict(torch.load(COCO_MODEL_PATH))

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Load a random image from the images folder

# file_names = next(os.walk(IMAGE_DIR))[2]
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
load_time = time.time() - load_start
print(load_time)
start_time = time.time()

with open('{}/OriginalImages/list.txt'.format(IMAGE_DIR)) as f:
    contents = f.readlines()
    file_names = [i.strip() for i in contents]

for i in range(len(file_names)):
    image = skimage.io.imread(os.path.join(IMAGE_DIR, 'OriginalImages', file_names[i]))

    # Run detection
    results = model.detect([image])

    # Visualize results
    r = results[0]
    save_image = [IMAGE_DIR, file_names[i]]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, save_image, r['scores'])


    # show the image
    # plt.show()

    # Check which masks are for car category
    car_masks = []
    for index in range(len(r['class_ids'])):
        if r['class_ids'][index] == 3 or r['class_ids'][index] == 8:
            car_masks.append(r['masks'][:, :, index])
    if len(car_masks) == 1:
        im = Image.fromarray((car_masks[0]*255.0).astype(np.uint8))
        im.save('{0}/TrimapCreatedFromMaskRCNN/{1}'.format(IMAGE_DIR, file_names[i]))

    else:
        masks_size = []
        for index in range(len(car_masks)):
            masks_size.append(np.sum(car_masks[index]))

        if not len(masks_size):
            with open('{}/OriginalImages/blacklist.txt'.format(IMAGE_DIR)) as f:
                black_contents = f.readlines()
                black_file_names = [i.strip() for i in black_contents]
            if file_names[i] not in black_file_names:
                with open('{}/OriginalImages/blacklist.txt'.format(IMAGE_DIR), 'a') as f:
                    # write the unfitted video names in the file, 'a' means append
                    print(file_names[i], file=f)
                    print(file_names[i])
            continue

        dominant_mask_location = masks_size.index(max(masks_size))
        car_mask = car_masks[dominant_mask_location]
        im = Image.fromarray((car_mask*255.0).astype(np.uint8))
        im.save('{0}/TrimapCreatedFromMaskRCNN/{1}'.format(IMAGE_DIR, file_names[i]))


elapsed_time = time.time() - start_time
print(elapsed_time)


