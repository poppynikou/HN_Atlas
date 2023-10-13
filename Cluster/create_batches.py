# create batches of patients for the cluster 

import os
import shutil

no_of_batches = 1
patients_per_batch = [4]

path_to_patient_data = 'D:/HNSCC/ARCHIVE/2023_10_08'

patient_folders = os.listdir(path_to_patient_data)
new_path = 'D:/HNSCC/BATCH_test/Itteration_0/'

for batch in range(0, no_of_batches):

    for index, patient_folder in enumerate(patient_folders):

        nifti_img = path_to_patient_data + '/' + patient_folder + '/NIFTI_IMGS/MASKED_CT.nii.gz'
        body_mask = path_to_patient_data + '/' + patient_folder + '/NIFTI_LIMBUS/BIN_body.nii.gz'

        
        if os.path.exists(nifti_img):

            if len(os.listdir(new_path)) < patients_per_batch[batch]*2:
            
                new_IMG_path = new_path + '/CT_' + str(index) + '.nii.gz'
                shutil.copy(nifti_img, new_IMG_path)


                new_BODY_path = new_path + '/BODY_' + str(index) + '.nii.gz'
                shutil.copy(body_mask, new_BODY_path)
