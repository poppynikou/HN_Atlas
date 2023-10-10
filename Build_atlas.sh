# Scheduler directives
#$ -S /bin/bash
#$ -l h_rt=03:00:00
#$ -l tmem=3G
#$ -l h_vmem=3G
#$ -j y
#$ -cwd
#$ -N Regs
#$ -pe smp 5
#$ -R y 
#$ -t 1-131

#export lib path
export LD_LIBRARY_PATH=/share/apps/gcc-8.3/lib64:$LD_LIBRARY_PATH

#source file location
source /SAN/medic/RTIC-MotionModel/software/niftyReg/niftyReg.source

#f3d command
F3D_CMD="/SAN/medic/RTIC-MotionModel/software/niftyReg/install/bin/reg_f3d"

#root data folder on the cluster
path_to_data="/home/pnikou/Documents"

source /share/apps/source_files/python/python-3.9.5.source

python main.py