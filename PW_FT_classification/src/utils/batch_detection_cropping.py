# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Demo for batch detection, cropping and resizing"""

#%% 
# PyTorch imports 
import torch
from tkinter import Tk, filedialog
# Importing the model, dataset, transformations and utility functions from PytorchWildlife
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife.data import datasets as pw_data 
# Importing the utility function for saving cropped images
from src.utils import utils

def batch_detection_cropping(folder_path=None, output_path=None, annotation_file=None):
    # Interactive selection if paths are not provided
    if not folder_path:
        root = Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory(title="Select folder containing images")
        root.destroy()
    if not output_path:
        root = Tk()
        root.withdraw()
        output_path = filedialog.askdirectory(title="Select output folder for cropped images")
        root.destroy()
    if not annotation_file:
        root = Tk()
        root.withdraw()
        annotation_file = filedialog.askopenfilename(title="Select annotation CSV file", filetypes=[("CSV files", "*.csv")])
        root.destroy()
    # Setting the device to use for computations ('cuda' indicates GPU)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Initializing the MegaDetectorV5 model for image detection
    detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True)

    """ Batch-detection demo """
    # Performing batch detection on the images
    results = detection_model.batch_image_detection(folder_path)

    # Saving the detected objects as cropped images
    crop_annotation_path = utils.save_crop_images(results, output_path, annotation_file)
    return crop_annotation_path



# %%
