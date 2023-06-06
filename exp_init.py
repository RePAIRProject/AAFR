#!/home/ali.alagrami/.conda/envs/my1env/bin/python

#SBATCH -n 1 # 5 cores
import sys
import os 
sys.path.append(os.getcwd()) 
from runner import experiment

def main():
    my_expirments = experiment("conf3.yaml")
    my_expirments.run()
    print("finished !")

if __name__ == '__main__':
    main()
