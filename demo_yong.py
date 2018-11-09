import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

from slic import SLICProcessor

# Root directory of the project
ROOT_DIR = os.path.abspath("/home/zhuangzi/PycharmProjects/Mask_RCNN_yong")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco
from keras.utils.vis_utils import plot_model
# matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
images_Predict_superpixel_output_DIR = os.path.join(ROOT_DIR, "images_Predict_superpixel_output")
images_Predict_output_DIR = os.path.join(ROOT_DIR, "images_Predict_output")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)

# plot_model(modellib.MaskRCNN, to_file='mode_MaskRCNN.png')
#plot_model(modellib.MaskRCNN, to_file='model_MaskRCNN_more detail.png', show_shapes=True)  # more detail

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

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
file_names = next(os.walk(IMAGE_DIR))[2]

for image_name in file_names:

    image_path = os.path.join(IMAGE_DIR, image_name) #random.choice
    image = skimage.io.imread(image_path)
    images_original = SLICProcessor.open_image(image_path)  ### superpixels

    # Run detection
    results = model.detect_superpixels([image], images_original, verbose=1)

    # Visualize results
    r = results[0]
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
    #                             class_names, r['scores'])
    visualize.save_predict_instances(images_Predict_superpixel_output_DIR, image_name, image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])

    # # Run detection original
    # results = model.detect([image],  verbose=1)
    # r = results[0]
    # visualize.save_predict_instances(images_Predict_output_DIR, image_name, image, r['rois'], r['masks'], r['class_ids'],
    #                                  class_names, r['scores'])

print("Image Count: {}")
