import os 
from utils import *

"""
Code for converting the original DICOM files and LIMBUSAI generated RT structs into binary nifti files
The code assumes that you have each of the patient data stored in a particular folder
The code will save each structure as an individual binary file 

"""

# no of patients to convert
no_patients = 3
# path to the dicom data 
base_path = 'C:/Users/poppy/Documents/Test_LIMBUS/Contours/'


folders = os.listdir(base_path)

for index in range(0,no_patients):
    patient_path = base_path + '/' + folders[index]

    try:

        dicom_Reader = DicomReaderWriter()
        dicom_Reader.walk_through_folders(patient_path)
        
        all_rois = dicom_Reader.return_rois(print_rois=False) 
        
        filename = patient_path + '/CT_'+str(index)+'.nii.gz'
        dicom_Reader.get_images()
        dicom_sitk_handle = dicom_Reader.dicom_handle
        sitk.WriteImage(dicom_sitk_handle, filename) 

        
        for roi in all_rois:
            
            try:
                #print(contour_name)

                dicom_Reader.set_contour_names_and_associations(contour_names=[roi], associations=[ROIAssociationClass(roi, [roi])])
                #path = dicom_Reader.where_is_ROI()
                #print('-----')
                #print(path)
                dicom_Reader.get_mask()

                #load mask
                mask_sitk_handle = dicom_Reader.annotation_handle

                # saves as boolean mask 
                roi_name =  patient_path + '/BIN_'+ str(roi) + '_'+str(index)+'.nii.gz'
                sitk.WriteImage(mask_sitk_handle, roi_name)
            except:
                pass
                
    except:
        pass
    

