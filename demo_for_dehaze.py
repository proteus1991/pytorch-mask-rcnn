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
import my_exception as myexp

load_start = time.time()
# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
# TODO model path
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

# Directory of images to run detection on
# TODO
IMAGE_DIR = os.path.join(ROOT_DIR, "HazeImages")
IMAGE_CAT = 'dehaze'

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
print('total loading time:{}'.format(load_time))
start_time = time.time()

with open('{0}/{1}_list.txt'.format(IMAGE_DIR, IMAGE_CAT)) as f:
    contents = f.readlines()
    file_names = [i.strip() for i in contents]

for i in range(len(file_names)):
    try:
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[i]))
        results = model.detect([image])
        print(i)
    except:
        with open('{0}/{1}_blacklist.txt'.format(IMAGE_DIR, IMAGE_CAT), 'a') as f:
            print(file_names[i], file=f)
            print(file_names[i])
        continue

    # try:
    #     results = model.detect([image])
    # except BaseException as var:
    #     print(file_names[i])
    #     continue

    # Visualize results
    r = results[0]
    save_image = [IMAGE_DIR, file_names[i]]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, save_image, r['scores'])

    # show the image
    # plt.show()

elapsed_time = time.time() - start_time
print('Running time: {}'.format(elapsed_time))


