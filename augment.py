import os
import glob
import shutil
import tempfile
import nibabel as nib

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch

from monai.apps import download_and_extract
from monai.config import print_config
from monai.transforms import Affine, Rand2DElastic

print_config()


class augmentor:

    def __init__(self, imageRepoPath, outputRepoPath):
        self.imageRepoPath = imageRepoPath
        self.outputRepoPath = outputRepoPath
        self.counter=0

    def fetchImagePaths(self):
        files = list()
        for subject_dir in glob.glob(os.path.join(self.imageRepoPath, "*")):
            subjectID = os.path.basename(subject_dir)
            imageFile = os.path.join(self.imageRepoPath, subjectID, "ct.nii.gz")		
            truthFile = os.path.join(self.imageRepoPath, subjectID, "truth.nii.gz")
            files.append({"image":imageFile, "label":truthFile})
        return files

    def agumentFiles(self, files):
        for pair in files[:2]:
            print(pair)
            self.augmentFile(pair)
    
    def augmentFile(self, pair):
        self.scaleAugment(pair)
    def scaleAugment(self, pair):
        image = nib.load(pair["image"])
        label = nib.load(pair["label"])
        image_matrix = image.get_fdata()
        label_matrix = label.get_fdata()
        image_matrix = np.expand_dims(image_matrix, 0)
        label_matrix = np.expand_dims(label_matrix, 0)
        scale_params=[1.2,0.8]

        for p in scale_params:
            affine = Affine(scale_params=(p, p, p), padding_mode="reflection")
            print(p)
            new_img_matrix = affine(image_matrix, mode="nearest")
            new_lbl_matrix = affine(label_matrix, mode="nearest")
            final_img = nib.Nifti1Image(new_img_matrix[0], image.affine, image.header)
            final_lbl = nib.Nifti1Image(new_lbl_matrix[0], label.affine, label.header)
            try:
                os.makedirs(os.path.join(self.outputRepoPath,"{}".format(self.counter)))
            except OSError:
                print ("failed to create the path")
            nib.save(final_img, os.path.join(self.outputRepoPath,"{}".format(self.counter),"ct.nii.gz"))
            nib.save(final_lbl, os.path.join(self.outputRepoPath,"{}".format(self.counter),"truth.nii.gz"))
            self.counter= self.counter+1 

























