'''
Code which rigidly registers patients to one patient, 
so that when they are clustered they are easier to then compare
'''

import os
import nibabel as nib 
import numpy as  np 
import matplotlib.pyplot as plt

def rigidReg(reg_aladin_path, ref_img, float_img, affine_matrix, resampled_img, RigOnly):
    
    """Perform a rigid registration using Aladin registration algorithm

    :param reg_aladin_path: path to reg_aladin executable (.exe)
    :type reg_aladin_path: str
    :param ref_img: path to reference image (.nii.gz/.nii)
    :type ref_img: str
    :param float_img: path to floating image (.nii.gz/.nii)
    :type float_img: str
    :param affine_matrix: path to resulting affine matrix (.txt)
    :type affine_matrix: str
    :param resampled_img: path to floating reference image with affine matrix (.nii.gz/.nii)
    :type resampled_img: str
    :param RigOnly: specify whether this is a rigid only or affine registration (.txt)
    :type resampled_img: bool
    """

    basic_command = reg_aladin_path + ' -ref ' + ref_img + ' -flo ' + float_img + ' -aff ' + affine_matrix + ' -res ' + resampled_img + ' -omp 12 '
    if RigOnly:
        command = basic_command + '-rigOnly'
    else:
        command = basic_command
    os.system(command)


BASE_PATH = 'D:/HNSCC/ARCHIVE/2023_10_08/'
reg_aladin_path = 'C:/Users/poppy/Documents/Nifty/niftyreg_install/bin/reg_aladin.exe'

patients = os.listdir(BASE_PATH)
ref_patient = str(BASE_PATH) + str(patients[0]) + '/NIFTI_IMGS/MASKED_CT.nii.gz'


for patient in patients: 


    float_patient = str(BASE_PATH) + str(patient) + '/NIFTI_IMGS/MASKED_CT.nii.gz'

    transformation = 'D:/HNSCC/ARCHIVE/2023_10_08_ALIGNED/' + str(patient) + '.txt'
    
    resampled_img = 'D:/HNSCC/ARCHIVE/2023_10_08_ALIGNED/' + str(patient) + '.nii.gz'
    if os.path.exists(resampled_img):
        slice = nib.load(resampled_img)
        Pixel_spacing = slice.header['pixdim'][1:4]

        #if not os.path.exists(resampled_img):
        #    if os.path.exists(float_patient):
        #
        #        rigidReg(reg_aladin_path, ref_patient, float_patient, transformation, resampled_img, RigOnly = True)
        #        os.remove(transformation)
        
        # save slice for easy viewing

        slice_path = 'D:/HNSCC/ARCHIVE/2023_10_08_ALIGNED/SLC_' + str(patient) + '.png'

        slice = nib.load(resampled_img)
        slice_img = slice.get_fdata()
        slice_img = np.array(slice_img, dtype = np.float32)
        # clip and rescale images 
        slice_img[np.isnan(slice_img)] = -1000
        slice_img[slice_img <= -1000] = -1000
        slice_img[slice_img >= 1000] = 1000
        minimum = np.amin(slice_img)
        maximum = np.amax(slice_img)
        slice_img = ((slice_img - minimum)/(maximum - minimum))*255

        section_img = slice_img[258,:,:].T
        fig = plt.figure()
        axes = plt.axes()
        axes.imshow(section_img.astype('uint8'), cmap='gray', origin = 'lower') 
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
        sag_aspect = Pixel_spacing[2]/Pixel_spacing[1]
        axes.set_aspect(sag_aspect)
        plt.tight_layout()
        plt.savefig(slice_path)
        plt.close()

