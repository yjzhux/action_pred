'''
Run jobs in batches using slurm
Usage: python slurm.py python python train_lstm.py
'''

from itertools import product
from subprocess import STDOUT,PIPE,Popen,call
import math
import sys
import csv
import os

# Experiment setup and user input
output_DIR = './slurm/resnet18/'
if not os.path.exists(output_DIR):
    os.makedirs(output_DIR)

parameter_lists={
                  "--phase": ['--phase rgb', '--phase flow', '--phase pose', '--phase rgb_flow'],
                  "--epochs": ['--epochs 150'],
                  "--frame_length": ['--frame_length 5'],
                  "--log_folder": ['--log_folder logs/YX'],
                  "--batch_size": ['--batch_size 1'],
                  "--pose_flag": ['--pose_flag yes', '--pose_flag no']
                }
task_NUM = 200

# A "config file" in this sense is a short string of input parameters
def print_list_as_config_file(filename, params):
   with open(filename,"w") as config_file:
      output_string=""
      for item in params:
         output_string=output_string+" "+str(item[1])+" "
      config_file.write(output_string)

def main(module_name, prog_name):
   #This converts a dictionary of lists to a list of dictionaries. Each dictionary is a representation of a single job.
   job_id = 0 
   param_dicts=[zip(parameter_lists.keys(), params) for params in product(*parameter_lists.values())]
   for parameter_dict in param_dicts:
      print_list_as_config_file(output_DIR+"config_"+str(job_id)+".csv",parameter_dict)
      job_id = job_id + 1
   job_total = job_id
   task_NUM = job_total-1

   queue_file_text='#!/bin/bash'+"\n"
   queue_file_text=queue_file_text+"#SBATCH --job-name="+prog_name+"\n"
   queue_file_text=queue_file_text+"#SBATCH --output="+output_DIR+prog_name+"_%A_%a.out\n"
   queue_file_text=queue_file_text+"#SBATCH --exclude=patel-gpu,gpu04,gpu[06-07],cat[04-08],elk[01-04]\n"
   queue_file_text=queue_file_text+"#SBATCH --cpus-per-task=4\n"
   queue_file_text=queue_file_text+"#SBATCH --mem=32g\n"
   queue_file_text=queue_file_text+"#SBATCH -a 0-"+str(task_NUM)+"\n\n\n"
   queue_file_text=queue_file_text+"task=`awk 'FNR==1 {print}' "+output_DIR+"config_$SLURM_ARRAY_TASK_ID.csv`"+"\n"
   queue_file_text=queue_file_text+"echo Task parameters \n"
   queue_file_text=queue_file_text+"echo $task"+"\n\n"
   queue_file_text=queue_file_text+module_name+" "+prog_name+" "+"$task"+"\n"
   open("queue_slurm.bash","w").write(queue_file_text)
   result_string=Popen(["sbatch", "queue_slurm.bash"],stderr=STDOUT,stdout=PIPE).stdout.read().decode("utf-8")
   sys.stdout.write(result_string)

if __name__ =="__main__":
   main(sys.argv[1], sys.argv[2])