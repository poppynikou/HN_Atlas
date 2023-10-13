import os 
from DicomRTTool.ReaderWriter import DicomReaderWriter
from DicomRTTool.ReaderWriter import ROIAssociationClass
import SimpleITK as sitk 

"""
Code for converting the original DICOM files and LIMBUSAI generated RT structs into binary nifti files
The code assumes that you have each of the patient data stored in a particular folder
The code will save each structure as an individual binary file 
"""

path_to_patient_folders = 'D:/HNSCC/ARCHIVE/2023_10_08'
patient_folders = [file for file in os.listdir(path_to_patient_folders) if file[0:5]=='HNSCC']

for patient_folder in patient_folders:

    patient_path = path_to_patient_folders + '/' + patient_folder + '/DICOM/'

    try:

        dicom_Reader = DicomReaderWriter()
        dicom_Reader.walk_through_folders(patient_path)
        
        all_rois = dicom_Reader.return_rois(print_rois=False) 
        

        nifti_img_path = path_to_patient_folders + '/' + patient_folder + '/NIFTI_IMGS/CT.nii.gz'
        dicom_Reader.get_images()
        dicom_sitk_handle = dicom_Reader.dicom_handle
        sitk.WriteImage(dicom_sitk_handle, nifti_img_path) 

        
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
                roi_name =  path_to_patient_folders + '/' + patient_folder  + '/NIFTI_LIMBUS/BIN_'+ str(roi) + '.nii.gz'
                sitk.WriteImage(mask_sitk_handle, roi_name)
            except:
                pass
                
    except:
        pass
    

