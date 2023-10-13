import shutil 
import os

# reorganise the files 
path_to_patient_imgs = 'X:/manifest-1685648397515/RADCURE'

patient_imgs = [file for file in os.listdir(path_to_patient_imgs) if file[0:7]=='RADCURE']

for patient_img in patient_imgs:

    patient_inner_folder = os.listdir(path_to_patient_imgs + '/' + patient_img)[0]
    if not os.path.exists(path_to_patient_imgs + '/' + patient_img + '/DICOM'):
        os.mkdir(path_to_patient_imgs + '/' + patient_img + '/DICOM')
    if not os.path.exists(path_to_patient_imgs + '/' + patient_img + '/NIFTI_IMGS'):
        os.mkdir(path_to_patient_imgs + '/' + patient_img + '/NIFTI_IMGS/')
    if not os.path.exists(path_to_patient_imgs + '/' + patient_img + '/NIFTI_LIMBUS'):
        os.mkdir(path_to_patient_imgs + '/' + patient_img + '/NIFTI_LIMBUS/')
    if not os.path.exists(path_to_patient_imgs + '/' + patient_img + '/NIFTI_TOTALSEG'):
        os.mkdir(path_to_patient_imgs + '/' + patient_img + '/NIFTI_TOTALSEG/')

    if len(os.listdir(path_to_patient_imgs + '/' + patient_img + '/DICOM')) == 0:

        patient_inner_inner_folders = os.listdir(path_to_patient_imgs + '/' + patient_img  + '/' + patient_inner_folder)

        for patient_inner_inner_folder in patient_inner_inner_folders:

            dcm_files = os.listdir(path_to_patient_imgs + '/' + patient_img  + '/' + patient_inner_folder + '/' + patient_inner_inner_folder)

            for dcm_file in dcm_files:

                file_to_move = path_to_patient_imgs + '/' + patient_img  + '/' + patient_inner_folder + '/' + patient_inner_inner_folder + '/' + dcm_file
                new_file_location = path_to_patient_imgs + '/' + patient_img + '/DICOM/' + dcm_file
                
                shutil.move(file_to_move, new_file_location)

'''

# reorganise the files 
path_to_patient_imgs = 'X:/manifest-1685648397515/RADCURE'

patient_imgs = [file for file in os.listdir(path_to_patient_imgs) if file[0:7]=='RADCURE']

for patient_img in patient_imgs:

    patient_inner_folders = os.listdir(path_to_patient_imgs + '/' + patient_img)

    for patient_inner_folder in patient_inner_folders:

        if patient_inner_folder == 'DICOM':
            pass
        elif patient_inner_folder == 'NIFTI_IMGS':
            pass
        elif patient_inner_folder == 'NIFTI_LIMBUS':
            pass
        elif patient_inner_folder == 'NIFTI_TOTALSEG':
            pass
        else:
            os.system('rm -r ' + path_to_patient_imgs + '/' + patient_img + '/' + patient_inner_folder)


'''