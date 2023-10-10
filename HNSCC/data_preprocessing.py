import os 
import shutil 

# reorganise the files 
path_to_limbus_contours = 'D:/LIMBUS'
path_to_patient_imgs = 'D:/HNSCC/ARCHIVE/2023_10_08'

'''

patient_imgs = [file for file in os.listdir(path_to_patient_imgs) if file[0:5]=='HNSCC']
limbus_contours = [file for file in os.listdir(path_to_limbus_contours)]


for patient_img in patient_imgs:

    for limbus_contour in limbus_contours:
        if patient_img[0:13] in limbus_contour:
            print(patient_img[0:13])
            print(limbus_contour)

            original_limbus_contour_path = path_to_limbus_contours + '/' + limbus_contour
            new_limbus_contour_path = path_to_patient_imgs + '/' + patient_img + '/' + limbus_contour
            print(original_limbus_contour_path)
            print(new_limbus_contour_path)

            
            shutil.move(original_limbus_contour_path, new_limbus_contour_path)



# put everything into a dicom folder 
patient_folders = [folder for folder in os.listdir(path_to_patient_imgs) if folder[0:5]=='HNSCC']

for patient_folder in patient_folders:

    dicom_objects = [file for file in os.listdir(path_to_patient_imgs + '/' + patient_folder)]

        #os.path.exists(path_to_patient_imgs + '/' + patient_folder + '/DICOM'):
    os.mkdir(path_to_patient_imgs + '/' + patient_folder + '/DICOM')
    os.mkdir(path_to_patient_imgs + '/' + patient_folder + '/NIFTI_IMGS/')
    os.mkdir(path_to_patient_imgs + '/' + patient_folder + '/NIFTI_LIMBUS/')
    os.mkdir(path_to_patient_imgs + '/' + patient_folder + '/NIFTI_TOTALSEG/')


'''
# put everything into a dicom folder 
patient_folders = [folder for folder in os.listdir(path_to_patient_imgs) if folder[0:5]=='HNSCC']

for patient_folder in patient_folders:

    dicom_objects = [file for file in os.listdir(path_to_patient_imgs + '/' + patient_folder) if file[-3:] == 'dcm']

    for dicom_obj in dicom_objects:


        old_path = path_to_patient_imgs + '/' + patient_folder + '/' + dicom_obj
        new_path = path_to_patient_imgs + '/' + patient_folder + '/DICOM/' + dicom_obj
        shutil.move(old_path, new_path)

