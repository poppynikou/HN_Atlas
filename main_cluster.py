from datetime import datetime
startTime = datetime.now()
from Classes import * 
from utils import *
import sys

#path to niftireg executables 
niftireg_path = sys.argv[1]
# csv path for reproprocessing
current_directory = sys.argv[2]
# base path to where the batches of patient images are stored
base_path = sys.argv[3]

#number of patients 
no_patients = 3
# batch number - the batch which you are working on 
batch = 1
#initial reference patient 
ref_patient = 18

# csv path for reproprocessing
preprocessing_data = current_directory +'/preprocessing.csv'

HNData = Data(base_path, batch, no_patients,niftireg_path)

Preprocess = PreProcessing(base_path, batch, no_patients)
patient_nos = Preprocess.get_patient_nos()
Preprocess.mask_CT(preprocessing_data,patient_nos)
Preprocess.mask_GTV(patient_nos)
Preprocess.create_body_mask(patient_nos)

Groupwise_ = Groupwise(base_path, batch, no_patients, patient_nos, niftireg_path)
Groupwise_.set_initial_ref_patient(ref_patient)
Groupwise_.set_itteration_no(itteration=0)
Groupwise_.InitialAlignment()

for itteration in range(1,3):
    
    Groupwise_.set_itteration_no(itteration)
    Groupwise_.AffineAlignment()
    Groupwise_.set_results()
    Groupwise_.calc_average_transformation()
    Groupwise_.calc_inv_average_transformation()
    Groupwise_.calc_composition()
    Groupwise_.resample_imgs()
    Groupwise_.calc_average_image()

Groupwise_.set_itteration_no(itteration = 3)
Groupwise_.create_multichannel_imgs([200,3000])
Groupwise_.Generate_Bash_Scripts()

endTime = datetime.now()
time_elapsed = endTime - startTime
print(f"Done: Started at {startTime}, finished at {endTime} (Took {time_elapsed})")