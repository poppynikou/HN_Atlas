import os 
import numpy as np 
import nibabel as nib 

"""
Script takes in nifti images, spits out a T4 vertabra segmentation and uses segmentation to mask the image 
"""

def get_image_objects(Img_path):
    
    """Get the key information from the nifti image. 
    Function reads in the nifti image and returns the image voxel value, the affine and the header. 

    :param Img_path: path to the image to read in 
    :type Img_path: str
    :return img_obj: voxel values stored in the image 
    :rtype img_obj: array
    :return img_affine: affine matrix associated with the image
    :rtype img_affine: Matrix
    :return img_header: header information associated with the image
    :rtype img_header: struct
    """
    
    img = nib.load(Img_path)
    img_obj = img.get_fdata()
    img_affine = img.affine
    img_header = img.header

    return img_obj, img_affine, img_header

def get_segmentation(input_img, output_segmentation):
    os.system('TotalSegmentator -i ' + str(input_img) + ' -o ' + str(output_segmentation) + ' --fast')

def get_cropping_slice(vertabrae_segmentation):

    img_data, _, _ = get_image_objects(vertabrae_segmentation)
    del _ 

    binary_img = np.sum(np.sum(img_data, axis=0), axis =0)
    index = np.where(binary_img!=0)[0][0]
    cropping_slice = int(index)

    return cropping_slice

def mask_anatomy(input_img, vertabrae_segmentation, masked_img):

    
    # read in data 
    img_data, img_affine, img_header = get_image_objects(input_img)
    img_data_copy = np.array(img_data.copy(), dtype = np.float32)
    (_,y,_) = np.shape(img_data_copy)
    del(_)

    min_z = get_cropping_slice(vertabrae_segmentation)

    img_data_copy[:,:,0:min_z] = np.NaN  

    # overide the CT 
    newNiftiObj = nib.Nifti1Image(img_data_copy, img_affine, img_header)
    newNiftiObj.set_data_dtype('float32')
    nib.save(newNiftiObj, masked_img)


path_to_patient_folders = 'D:/HNSCC/ARCHIVE/2023_10_08'
patient_folders = [file for file in os.listdir(path_to_patient_folders) if file[0:5]=='HNSCC']

for patient_folder in patient_folders:
    if os.path.exists(path_to_patient_folders + '/' + patient_folder + '/NIFTI_IMGS/CT.nii.gz'):
        input_img = path_to_patient_folders + '/' + patient_folder + '/NIFTI_IMGS/CT.nii.gz'
        output_path = path_to_patient_folders + '/' + patient_folder + '/NIFTI_TOTALSEG/'
        if len(os.listdir(output_path)) ==0:
            get_segmentation(input_img, output_path)
            try:
                vertabrae_segmentation = path_to_patient_folders + '/' + patient_folder + '/NIFTI_TOTALSEG/vertebrae_T4.nii.gz'
                masked_img = path_to_patient_folders + '/' + patient_folder + '/NIFTI_IMGS/MASKED_CT.nii.gz'
                mask_anatomy(input_img, vertabrae_segmentation, masked_img)

            except:
                pass