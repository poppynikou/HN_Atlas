import os
import numpy as np 




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
    
def rigidReg_SINGLETREAD(reg_aladin_path, ref_img, float_img, affine_matrix, resampled_img, RigOnly):

    """Perform a rigid registration using Aladin registration algorithm with a single thread

    :param reg_aladin_path: path to reg_aladin executable (.exe)
    :type reg_aladin_path: str
    :param ref_img: path to reference image (.nii.gz/.nii)
    :type ref_img: str
    :param float_img: path to floating image (.nii.gz/.nii)
    :type float_img: str
    :param affine_matrix: path to resulting affine matrix (.txt)
    :type affine_matrix: str
    :param resampled_img: path to floating reference image with affine matrix (.txt)
    :type resampled_img: str
    :param RigOnly: specify whether this is a rigid only or affine registration
    :type resampled_img: bool
    """

    
    basic_command = reg_aladin_path + ' -ref ' + ref_img + ' -flo ' + float_img + ' -aff ' + affine_matrix + ' -res ' + resampled_img + ' -omp 1 '
    if RigOnly:
        command = basic_command + '-rigOnly'
    else:
        command = basic_command
    os.system(command)


def deformableReg(reg_f3d_path, ref_img, float_img, resampled_img, transformation):
    
    """Perform a deformable registration between a reference and floating image. 
    The optimization is perfromed with a normalised linear cross corelation as similarity measure with a five voxel gaussian window.
    A linear energy regularization with a weighting of 0.1 is used. 
    A control point spacing of 10 voxels is set in all x,y,z directions and 5 levels are used. 
    Resampled images are padded with nan.

        :param reg_f3d_path: path to reg_f3d executable (.exe)
        :type reg_f3d_path: str
        :param ref_img: path to reference image (.nii.gz/.nii)
        :type ref_img: str
        :param float_img: path to floating image to resample(.nii.gz/.nii)
        :type float_img: str
        :param transformation: path to transformation (.nii.gz/.nii)
        :type transformation: str
        :param resampled_img: path to resampled image (.nii.gz/.nii)
        :type resampled_img: str
        """

    command_basic = reg_f3d_path + ' -ref ' + ref_img + ' -flo ' + float_img + ' -res ' + resampled_img + ' -cpp ' + transformation + ' -omp 12 '
    command_params = ' -sx -10 -sy -10 -sz -10 -be 0 --lncc -5 -ln 5 -vel -le 0.1 -pad nan'
    
    command = command_basic + command_params
    os.system(command)


def avgAff(reg_average_path, average_affine_path, affine_matrixes):

    """Calculate the average affine of a group of affine matrixes

    :param reg_average_path: path to reg_average executable (.exe)
    :type reg_average_path: str
    :param average_affine_path: path to where to store the average affine matrix (.txt)
    :type average_affine_path: str
    :param affine_matrixes: a list of affine matrixes to average []
    :type affine_matrixes: list
    
    """

    command = reg_average_path + ' ' + average_affine_path + ' -avg ' 
    for i in np.arange(0,len(affine_matrixes)):

        command = command + affine_matrixes[i] + ' '

    os.system(command)
 
def invAff(reg_transform_path, ref_img, affine, inv_affine):

    """Calculate the inverse of an affine transformation 

    :param reg_transform_path: path to reg_transform executable (.exe)
    :type reg_transform_path: str
    :param ref_img: path to reference image used in registration where affine matrix was obtained (.nii.gz/.nii)
    :type ref_img: str
    :param affine: path to affine matrix to invert (.txt)
    :type affine: str
    :param inv_affine: path to inverse affine matrix (.txt)
    :type inv_affine: str
    """

    command = reg_transform_path + ' -ref ' + ref_img + ' -invAff ' + affine + ' ' +  inv_affine
    os.system(command)

def invDef(reg_transform_path, ref_img, Def_transformation, inv_Def_transformation):

    """Calculate the inverse of a deformation field

    :param reg_transform_path: path to reg_transform executable (.exe)
    :type reg_transform_path: str
    :param ref_img: path to reference image used in registration where the deformation field was obtained (.nii.gz/.nii)
    :type ref_img: str
    :param Def_transformation: path to deformation field to invert (.nii.gz/.nii)
    :type Def_transformation: str
    :param inv_Def_transformation: path to inverse deformation field (.nii.gz/.nii)
    :type inv_Def_transformation: str
    """
     
    command = reg_transform_path + ' -ref ' + ref_img + ' -invNrr ' + Def_transformation +' ' + ref_img  + ' ' +  inv_Def_transformation
    os.system(command)


def resampleImg(reg_resample_path, ref_img, float_img, transformation, resampled_img):

    """Resample an Image with cubic interpolation and pad with nan

        :param reg_resample_path: path to reg_resample executable (.exe)
        :type reg_resample_path: str
        :param ref_img: path to reference image (.nii.gz/.nii)
        :type ref_img: str
        :param float_img: path to floating image to resample(.nii.gz/.nii)
        :type float_img: str
        :param transformation: path to transformation to resample with (.nii.gz/.nii/.txt)
        :type transformation: str
        :param resampled_img: path to resampled image (.nii.gz/.nii)
        :type resampled_img: str
        """


    command = reg_resample_path + ' -ref ' + ref_img +  ' -flo ' + float_img + ' -trans ' + transformation + ' -res ' + resampled_img + ' -inter 3 -pad nan'
    os.system(command)
    
def resampleBINImg(reg_resample_path, ref_img, float_img, transformation, resampled_img):

    """Resample a binary Image with linear interpolation and pad with 0

        :param reg_resample_path: path to reg_resample executable (.exe)
        :type reg_resample_path: str
        :param ref_img: path to reference image (.nii.gz/.nii)
        :type ref_img: str
        :param float_img: path to floating image to resample (.nii.gz/.nii)
        :type float_img: str
        :param transformation: path to transformation to resample with (.nii.gz/.nii/.txt)
        :type transformation: str
        :param resampled_img: path to resampled image (.nii.gz/.nii)
        :type resampled_img: str
        """
    
    command = reg_resample_path + ' -ref ' + ref_img +  ' -flo ' + float_img + ' -trans ' + transformation + ' -res ' + resampled_img + ' -inter 1 -pad 0'
    os.system(command)


def UpdSform(reg_transform_path, img_to_be_updated_path, affine_matrix_path, updated_img_path):
    
    """Update the Sform of an image

    :param reg_transform_path: path to reg_transform executable (.exe)
    :type reg_transform_path: str
    :param img_to_be_updated_path: path to image which the Sform needs updating (.nii.gz/.nii)
    :type img_to_be_updated_path: str
    :param affine_matrix_path: path to affine matrix to update the Sform with (.txt)
    :type affine_matrix_path: str
    :param updated_img_path: path to image which has been updated (.nii.gz/.nii)
    :type updated_img_path: str
    """

    command = reg_transform_path +' -updSform ' + img_to_be_updated_path + ' ' + affine_matrix_path + ' ' + updated_img_path
    os.system(command)

def RigidToDeformation(reg_transform_path, ref_img, input_transformation, output_transformation):

    """Calculation deformation fiel from rigid transformation 

        :param reg_transform_path: path to reg_transform executable (.exe)
        :type reg_transform_path: str
        :param ref_img: path to reference image (.nii.gz/.nii)
        :type ref_img: str
        :param input_transformation: path to rigid transfromation (.txt)
        :type input_transformation: str
        :param output_transformation: path to deformation field (.nii.gz/.nii)
        :type output_transformation: str

        """

    command = reg_transform_path + ' -ref ' + ref_img + ' -def ' + input_transformation + ' ' + output_transformation 
    os.system(command)

def CppToDeformation(reg_transform_path, ref_img, input_transformation, output_transformation):

    """Calculation deformation fiel from control point grid 

        :param reg_transform_path: path to reg_transform executable (.exe)
        :type reg_transform_path: str
        :param ref_img: path to reference image (.nii.gz/.nii)
        :type ref_img: str
        :param input_transformation: path to control point grid transfromation (.nii.gz/.nii)
        :type input_transformation: str
        :param output_transformation: path to deformation field (.nii.gz/.nii)
        :type output_transformation: str

        """

    command = reg_transform_path + ' -ref ' + ref_img + ' -def ' + input_transformation + ' ' + output_transformation
    os.system(command)

def ComposeTransformations(reg_transform_path, ref_img, transformation1, transformation2, output_transformation):

    """Composition of two transformations  output_transformation(x) = transformation2(transformation1(x)).

        :param reg_transform_path: path to reg_transform executable (.exe)
        :type reg_transform_path: str
        :param ref_img: path to reference image (.nii.gz/.nii)
        :type ref_img: str
        :param transformation1: path to deformation field (.nii.gz/.nii)
        :type transformation1: str
        :param transformation2: path to deformation field (.nii.gz/.nii)
        :type transformation2: str
        :param output_transformation: path to composed transfromation (.nii.gz/.nii)
        :type output_transformation: str

        """
    
    command = reg_transform_path + ' -ref ' + ref_img + ' -comp ' + transformation1 + ' ' + transformation2 + ' '  + output_transformation
    os.system(command)

    
