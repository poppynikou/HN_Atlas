## script which includes the classes which does the postprocessing of the groupwise registrations
# which creates the atlas 
from natsort import natsorted
import nibabel as nib 
import numpy as np 
import os 
from copy import deepcopy
from utils import * 
import pandas as pd 

class Data():

    """Data Class. Contains methods to set the global paths for all other classes, to be used iteratively in the groupwise algorithm."""

    def __init__(self, base_path, batch, no_patients):
        """Initialise the class. Set all the paths to niftireg executables

        :param base_path: path to folder in which the batch folders are kept 
        :type base_path: str
        :param batch: the batch number which you would like to process
        :type batch: int
        :param no_patients: the number of patients in that batch 
        :type no_patients: str
        :param niftireg_path: folder in which the niftireg executables are stored
        :type niftireg_path: str
        """

        # define path to the batch 
        self.batch_path = base_path + '/BATCH_' + str(batch)
        # no of patients to process
        self.no_patients = no_patients

        self.reg_transform ='reg_transform'
        self.reg_average = 'reg_average'
        self.reg_aladin = 'reg_aladin'
        self.reg_resample = 'reg_resample'
        self.reg_f3d = 'reg_f3d'

    def get_image_objects(self,Img_path):
    
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




class Groupwise(Data):

    def __init__(self, base_path, batch, no_patients, patient_nos):
    
        Data.__init__(self, base_path, batch, no_patients)
        self.patient_nos = patient_nos
        return

    def set_initial_ref_patient(self, patient_no):

        # define reference img path 
        self.ref_patient = patient_no

    def set_itteration_no(self,itteration):

        """Set all the paths for a particular iteration 

        :param base_path: itteration number 
        :type base_path: int
        """

        # iteration number
        self.itteration = itteration

        # current folder
        self.current_iteration_path = self.batch_path + '/Iteration_' + str(self.itteration)  
        if not os.path.exists(self.current_iteration_path):
            os.mkdir(self.current_iteration_path)
        # previous folder
        self.prev_iteration_path = self.batch_path + '/Iteration_' + str(self.itteration-1)  

        
        if itteration != 0:
            # define path to body masks 
            self.img_masks = natsorted([self.prev_iteration_path + '/' + file for file in os.listdir(self.prev_iteration_path) if (file.startswith('BODY_') & file.endswith('nii.gz'))])
        
            if itteration < 3:
                if itteration != 1:
                    # define reference img path 
                    self.ref_img = self.prev_iteration_path + '/average_CT.nii.gz'
                elif itteration == 1:
                    # define reference img path 
                    self.ref_img = self.prev_iteration_path + '/CT_'+str(self.ref_patient)+'.nii.gz'
                # define average image path 
                self.average_img = self.current_iteration_path + '/average_CT.nii.gz'
                # define average transformation path 
                self.average_transformation = self.current_iteration_path + '/average_transformation.txt'
                # define inverse average transformation path
                self.inv_average_transformation = self.current_iteration_path + '/inv_average_transformation.txt'
                # define paths to imgs
                self.imgs_paths = natsorted([self.prev_iteration_path + '/' + file for file in os.listdir(self.prev_iteration_path) if (file.startswith('CT_') & file.endswith('nii.gz'))])
            else:
                #define reference img path 
                self.ref_img = self.prev_iteration_path + '/multichannel_average_CT.nii.gz'
                #define average image path 
                self.average_img = self.current_iteration_path + '/multichannel_average_CT.nii.gz'
                # define average transformation path 
                self.average_transformation = self.current_iteration_path + '/average_transformation.nii.gz'
                # define inverse average transformation path
                self.inv_average_transformation = self.current_iteration_path + '/inv_average_transformation.nii.gz'
                # define paths to imgs
                self.imgs_paths = natsorted([self.prev_iteration_path + '/' + file for file in os.listdir(self.prev_iteration_path) if (file.startswith('multichannel_CT_') & file.endswith('nii.gz'))])
        else:
            # define paths to imgs
            self.imgs_paths = natsorted([self.current_iteration_path + '/' + file for file in os.listdir(self.current_iteration_path) if (file.startswith('CT_') & file.endswith('nii.gz'))])
            # define reference img path 
            self.ref_img = self.current_iteration_path + '/CT_'+str(self.ref_patient)+'.nii.gz'
    
    def set_results(self):
        
        if self.itteration != 0:

            if self.itteration < 3:
                # define paths to transformations 
                self.transformation_paths = natsorted([self.current_iteration_path + '/' + file for file in os.listdir(self.current_iteration_path) if (file.startswith('transformation_') & file.endswith('txt')) & ('backward' not in str(file))])
            else:
                # define paths to transformations 
                self.transformation_paths = natsorted([self.current_iteration_path + '/' + file for file in os.listdir(self.current_iteration_path) if (file.startswith('transformation_') & file.endswith('nii.gz')) & ('backward' not in str(file))])

    def InitialAlignment(self):

        #loops over the images
        for index, img in enumerate(self.imgs_paths):

            float_img = img 
            affine_matrix = self.current_iteration_path + '/transformation_' + str(self.patient_nos[index]) + '.txt'
            resampled_img = self.current_iteration_path + '/InitialAlignment_' + str(self.patient_nos[index]) + '.nii.gz'
            # performs rigid registration
            rigidReg(self.reg_aladin, self.ref_img, float_img, affine_matrix, resampled_img, RigOnly=True)
            #deletes resampled image 
            os.remove(resampled_img)
            UpdSform(self.reg_transform, float_img, affine_matrix, float_img)
            os.remove(affine_matrix)

    def AffineAlignment(self):

        """Performs an affine alignment between a reference image and a group of floating images one at a time.
        If iteration = 1 this will be rigid only, if iteration = 2 it will include sheering and stretching too.
        """

        if self.itteration == 1:
            RigOnly = True
        elif self.itteration == 2:
            RigOnly = False 

        #loops over the images
        for index, img in enumerate(self.imgs_paths):

            float_img = img 
            affine_matrix = self.current_iteration_path + '/transformation_' + str(self.patient_nos[index]) + '.txt'
            resampled_img = self.current_iteration_path + '/resampled_img_' + str(self.patient_nos[index]) + '.nii.gz'
            # performs rigid registration
            rigidReg(self.reg_aladin, self.ref_img, float_img, affine_matrix, resampled_img, RigOnly)
            #deletes resampled image 
            os.remove(resampled_img)
    
    def Generate_Bash_Scripts(self):

        #function which generates bash script to send to the cluster 

        return

    
    
    def calc_average_transformation(self):

        """Calls correct functions to calculate an average transformation for current iteration
        """

        if self.itteration <3:
            self.calc_average_transformation_affine()
        else:
            self.calc_average_transformation_def()


    def calc_average_transformation_affine(self):

        """Calls niftireg wrapper function to calculate the average affine transformation for current iteration
        """
    
        avgAff(self.reg_average, self.average_transformation, self.transformation_paths)


    def calc_average_transformation_def(self):
        """Calculates the average deformable transformation for current iteration
        """

        # reads in a transformation to get the dimensions and shape of transformations
        _, _, header = self.get_image_objects(self.transformation_paths[0])
        del _
        average_shape = header['dim']
        transformation_storage = np.empty(shape = (self.no_patients, average_shape[1], average_shape[2], average_shape[3], 3), dtype = np.float16)

        # reads in all deformation fields 
        for index in range(0,self.no_patients,1):
            
            # import transformation for that particular patient 
            transformation_path = self.transformation_paths[index] 

            # imports the data 
            transformation_obj, _, _ = self.get_image_objects(transformation_path)
            transformation_obj = np.array(transformation_obj, dtype = np.float16)
            transformation_obj = np.squeeze(transformation_obj)
    
            #store
            transformation_storage[index, :, :, :, :] = transformation_obj

        del transformation_path
        del transformation_obj
        del _
        # averages the transformations
        Average = np.nanmean(transformation_storage, axis = 0, dtype = np.float16)
        del transformation_storage
        # puts it in the correct shape 
        Average = np.reshape(Average, newshape = (average_shape[1], average_shape[2], average_shape[3], 1, 3))
        Average = np.array(Average, dtype = np.float32)

        # read in the first reference image to use the header
        _, ref_img_affine, _ = self.get_image_objects(self.ref_img)
        del _
        
        # save the transformation 
        niftiobject1 = nib.Nifti1Image(Average, ref_img_affine)
        niftiobject1.set_data_dtype('float32')
        nib.save(niftiobject1, self.average_transformation)


    def calc_inv_average_transformation(self):
        """Calls correct niftireg wrapper functions to calculate an inverse transformation for current iteration
        """

        if self.itteration <3:
            invAff(self.reg_transform, self.ref_img, self.average_transformation, self.inv_average_transformation)
        else:
            invDef(self.reg_transform, self.ref_img, self.average_transformation, self.inv_average_transformation)

        
    def calc_composition(self):

        """Loops over correct niftireg wrapper functions to calculate an inverse transformation for current iteration
        """

        for index in range(0,self.no_patients,1):
            
            # import transformation for that particular patient 
            transformation_path = self.transformation_paths[index] 
            if self.itteration <3:
                composed_transformation = self.current_iteration_path + '/comp_transformation_' + str(self.patient_nos[index]) + '.txt'
            else:
                composed_transformation = self.current_iteration_path + '/comp_transformation_' + str(self.patient_nos[index]) + '.nii.gz'
            ComposeTransformations(self.reg_transform, self.ref_img, self.inv_average_transformation, transformation_path, composed_transformation)

    
    def resample_imgs(self):

        """Loops over correct niftireg wrapper function to resample images and body masks 
        """
        # the prefixes which the images are stored by 
        mask_prefix = 'BODY'
        if self.itteration <3:
            img_prefix = 'CT'
        else:
            img_prefix = 'multichannel_CT'
        
        # find transformations in the folder 
        Transformations = natsorted([self.current_iteration_path + '/' + file for file in os.listdir(self.current_iteration_path) if (file.startswith('comp_transformation_') & ('backward' not in str(file)))])

        # loop over all patients and resample images and masks 
        for index in range(0,self.no_patients,1):

            float_img = self.imgs_paths[index]
            transformation = Transformations[index]
            resampled_img = self.current_iteration_path + '/' + str(img_prefix) + '_' + str(self.patient_nos[index]) +'.nii.gz'
            resampleImg(self.reg_resample, self.ref_img, float_img, transformation, resampled_img)


            float_img = self.img_masks[index]
            transformation = Transformations[index]
            resampled_img = self.current_iteration_path + '/' + str(mask_prefix) + '_' + str(self.patient_nos[index]) +'.nii.gz'
            resampleImg(self.reg_resample, self.ref_img, float_img, transformation, resampled_img)



    def calc_average_image(self):

        """Loops resampled images and calculates the average 
        """

        mask_prefix = 'BODY'
        if self.itteration <3:
            img_prefix = 'CT'
        else:
            img_prefix = 'multichannel_CT'

        # list resamples images and masks 
        Resampled_imgs = natsorted([self.current_iteration_path  + '/' + file for file in os.listdir(self.current_iteration_path ) if file.startswith(img_prefix)])
        Resampled_masks = natsorted([self.current_iteration_path  + '/' + file for file in os.listdir(self.current_iteration_path ) if file.startswith(mask_prefix)])

        #reads in the image 
        img_obj, _, _ = self.get_image_objects(Resampled_imgs[0])
        img_shape = np.shape(np.array(img_obj))
        del img_obj
        del _

        # create an array of nans for storage
        Nan_map = np.empty(shape=(img_shape[0],img_shape[1],img_shape[2],self.no_patients))
        del img_shape

        for index in np.arange(0,self.no_patients):

            # mask each image of the body
            # and replace the rest of the image with nans
            # then average over all patients using reg_average


            # define img and mask path 
            img_path = Resampled_imgs[index]
            mask_path = Resampled_masks[index]

            # read in the body mask 
            mask, _, _ = self.get_image_objects(mask_path)
            mask_obj = mask.copy().astype('bool')

            # imports the data 
            img = nib.load(img_path)
            img_obj = img.get_fdata().copy()
            img_obj = np.array(img_obj, dtype = np.float32)

            # replaces all the pCT which isnt body with a nan so its not included in the averaging
            img_obj[~mask_obj] = float('NaN')

            #store
            Nan_map[:,:,:,index] = img_obj

        # calculate the average image, ignoring the nans
        Nan_map = Nan_map.astype('float32')
        Masked_Average = np.nanmean(Nan_map, axis = 3)
        Average = Masked_Average.copy()
        del Nan_map

        # save 
        img_hdr = img.header
        img_affine = img.affine
        Avg_Niftiobj = nib.Nifti1Image(Average, img_affine, img_hdr)
        Avg_Niftiobj.set_data_dtype(np.float32)
        nib.save(Avg_Niftiobj, self.average_img)

    def create_multichannel_imgs(self, threshold_region = []):
        
        """
        Function which takes in the CT and then create a multichannel CT
        The first channel is the CT and the second channel is a bone segmentation
        Bone segmentation is created by thresholding - thresholding region given as input to function
        :param threshold_region: The Haunsfield unit threshold in which the images should be clipped
        :type base_path: list

        """
        Resampled_imgs = natsorted([self.current_iteration_path  + '/' + file for file in os.listdir(self.current_iteration_path ) if file.startswith('CT')])
        Resampled_masks = natsorted([self.current_iteration_path  + '/' + file for file in os.listdir(self.current_iteration_path ) if file.startswith('BODY')])


        for index, resampled_img in enumerate(Resampled_imgs):

            # load in the info 
            CT_obj, CT_affine, CT_header = self.get_image_objects(resampled_img)
            CT_obj = np.array(CT_obj, dtype = np.float32)
            CT_obj_copy = CT_obj.copy()

            masked_CT_obj, CT_affine, CT_header = self.get_image_objects(Resampled_masks[index])
            masked_CT_obj = np.array(masked_CT_obj, dtype = np.float32)
            
            # set everything which isnt bone to air 
            CT_obj_copy[CT_obj_copy<threshold_region[0]] = -1000
            CT_obj_copy[CT_obj_copy>threshold_region[1]] = -1000

            stacked = np.stack((masked_CT_obj,CT_obj_copy), axis = -1)

            # save the multichannel image 
            new_nifti_obj = nib.Nifti1Image(stacked, CT_affine, CT_header)
            new_nifti_obj.set_data_dtype('float32')
            output_img = self.current_iteration_path  + '/multichannel_CT_' + str(index+1) +'.nii.gz'
            nib.save(new_nifti_obj, output_img)
        
class PreProcessing(Data):
    
    
    """PreProcessing Class which inherits from Data Class. Contains methods which preprocess the images for the groupwise registration."""

    def __init__(self, base_path, batch, no_patients):

        Data.__init__(self, base_path, batch, no_patients)
        self.current_iteration_path = self.batch_path + '/Iteration_0'
        return
    
    def get_patient_nos(self):

        patient_nos = natsorted([file.split('CT_')[1].split('.nii.gz')[0] for file in os.listdir(self.current_iteration_path) if (file.startswith('CT_') & file.endswith('nii.gz'))])

        return patient_nos

    def mask_CT(self, csv_path, patient_nos):
        
        """Function which takes in a CT and saves a masked CT.
        The image is masked to remove the couch, and the excess air in front of the patient, and anything underneath the C11 vertabrae of the patient.
        The voxels within these locations are set to NaN, so that they are not considered in algorithms down the line. 
        The slices for the preprocessing are selected manually, and are stored in a csv file called 'preprocessing.csv'.

        :param csv_path: path to csv file containing image slices to mask CTs with 
        :type csv_path: str
        """
        slices = pd.read_csv(csv_path)

        for patient_no in patient_nos:

            img_path = self.current_iteration_path + '/CT_' + str(patient_no) + '.nii.gz'

            min_y = int(slices['Min_y'].loc[slices['HN_Patient'] == 'HN_' + str(int(patient_no))])# excess air in front of patient
            max_y = int(slices['Max_y'].loc[slices['HN_Patient'] == 'HN_' + str(int(patient_no))])# radiotherapy couch
            #min_z = int(slices['Min_z'].loc[slices['HN_Patient'] == 'HN_' + str(int(patient_no))])# anything under C4
            
            # read in data 
            img_data, img_affine, img_header = self.get_image_objects(img_path)
            img_data_copy = np.array(img_data.copy(), dtype = np.float32)
            (_,y,_) = np.shape(img_data_copy)
            del(_)

            # mask the CT 
            img_data_copy[:,0:min_y,:] = np.NaN 
            img_data_copy[:,max_y:y,:] = np.NaN 
            #img_data_copy[:,:,0:min_z] = np.NaN  

            # overide the CT 
            newNiftiObj = nib.Nifti1Image(img_data_copy, img_affine, img_header)
            newNiftiObj.set_data_dtype('float32')
            nib.save(newNiftiObj, img_path)


    def mask_GTV(self, patient_nos):

        """Function which takes in a CT and masks the tumour from the scan.
        This is done so that the patient to patient alignments are not attempting to align pieces of anatomy from one scan to another which doesnt exist.
        GTV binary segmentations must be stored in the same folder, and have name 'GTV_X where X is the patient number.'
        """

        new_imgs = natsorted([self.current_iteration_path + '/' + file for file in os.listdir(self.current_iteration_path) if (file.startswith('CT_') & file.endswith('nii.gz'))])
        new_masks = natsorted([self.current_iteration_path + '/' + file for file in os.listdir(self.current_iteration_path) if (file.startswith('GTV_') & file.endswith('nii.gz'))])
        
        for index, patient_no in enumerate(patient_nos):

            if os.path.exists(new_masks[index]):

                # load in the info 
                CT_obj, CT_affine, CT_header = self.get_image_objects(new_imgs[index])
                CT_obj = np.array(CT_obj, np.float32)
                CT_obj_copy = CT_obj.copy()
                
                tumour_mask, _, _ = self.get_image_objects(new_masks[index])
                tumour_mask = np.array(tumour_mask, dtype = np.bool8)

                CT_obj_copy[tumour_mask] = np.NaN

                # save the multichannel image 
                new_nifti_obj = nib.Nifti1Image(CT_obj_copy, CT_affine, CT_header)
                new_nifti_obj.set_data_dtype(np.float32)
                nib.save(new_nifti_obj, new_imgs[index])
            else:
                print('No GTV mask found for Patient ' + str(patient_no))

    def create_body_mask(self, patient_nos):

        """Function which takes in a masked CT and creates a body mask.
        The body mask is stored in the folder as 'BODY_X', where X is the patient number.
        """
        new_imgs = natsorted([self.current_iteration_path + '/' + file for file in os.listdir(self.current_iteration_path) if (file.startswith('CT_') & file.endswith('nii.gz'))])
        
        for index, patient_no in enumerate(patient_nos):

            #read in patient masked CT scan 
            img_CT, img_affine, img_hdr = self.get_image_objects(new_imgs[index])
            img_CT_copy = deepcopy(img_CT)
            img_CT_copy = np.array(img_CT_copy, dtype = np.float64)

            # any NaN voxels are replaced with air 
            img_CT_copy[np.isnan(img_CT_copy)] = -1000
            # any voxels of small HU are replaced with air 
            img_CT_copy[img_CT_copy < -200] = -1000

            shape = np.shape(img_CT)
            body_mask = np.zeros(shape = shape, dtype=np.float32)

            # finds the first and last voxels in each z and y direction which are not air 
            # i.e. it finds the starting locations of the body 
            for z_slice in np.arange(0, shape[2]):
            
                for y_slice in np.arange(0, shape[1]):

                    x_img_array = np.squeeze(img_CT_copy[:,y_slice,z_slice])
                    img_boolean = ~(x_img_array == -1000)

                    # these locations will not exist for the first few slices above the skull 
                    # need to include a condition for this 
                    if len(np.nonzero(img_boolean)[0]) == 0:

                        body_mask[:,y_slice,z_slice] = 0

                    elif len(np.nonzero(img_boolean)[0]) == shape[0]:

                        body_mask[:,y_slice,z_slice] = 0

                    else:
                        
                        first_location_x = np.nonzero(img_boolean)[0][0]
                        last_location_x=  np.nonzero(img_boolean)[0][-1]

                        body_mask[first_location_x:last_location_x, y_slice, z_slice]=1

            mask_path = self.current_iteration_path + '/BODY_' + str(patient_no) + '.nii.gz'
            newNiftiObj = nib.Nifti1Image(body_mask, img_affine, img_hdr)
            nib.save(newNiftiObj, mask_path)

