# Scheduler directives
#$ -S /bin/bash
#$ -l h_rt=03:00:00
#$ -l tmem=3G
#$ -l h_vmem=3G
#$ -j y
#$ -cwd
#$ -N Batch_1

#export lib path
export LD_LIBRARY_PATH=/share/apps/gcc-8.3/lib64:$LD_LIBRARY_PATH

#path to niftireg executables 
export PATH=/SAN/medic/RTIC-MotionModel/software/niftyReg/install/bin:${PATH}
export LD_LIBRARY_PATH=/SAN/medic/RTIC-MotionModel/software/niftyReg/install/bin:${LD_LIBRARY_PATH}

# path to executable of python installed in my environment 
python_poppy="/home/pnikou/.conda/envs/hn_atlas/bin/python3.9"

#root data folder on the cluster
path_to_data="/home/pnikou/Documents/Data"

#root data folder on the cluster
path_to_excel_file="/home/pnikou/"

#path to save log file
path_to_log_file="/home/pnikou/"

# command to run the code 
$python_poppy main.py $path_to_data $path_to_excel_file -u $path_to_log_file