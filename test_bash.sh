# Scheduler directives
#$ -S /bin/bash
#$ -l h_rt=00:01:00
#$ -l tmem=0.5G
#$ -l h_vmem=0.5G
#$ -j y
#$ -cwd
#$ -N Test_executables


PATH="/SAN/medic/RTIC-MotionModel/software/niftyReg/install/bin/reg_f3d"