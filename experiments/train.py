# Run a baseline model in BasicTS framework.
import os
import sys
from argparse import ArgumentParser
import subprocess

sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

import basicts

torch.set_num_threads(4) # aviod high cpu avg usage

def parse_args():
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
    parser.add_argument('-c', '--cfg', default='baselines/MSFMP01/TMAE_METRLA.py', help='training config')
    parser.add_argument('-g', '--gpus', default='0', help='visible gpus')
    return parser.parse_args()

def main():
    args = parse_args()
    basicts.launch_training(args.cfg, args.gpus, node_rank=0)
    # import os
    # current_dir = o s.path.dirname(os.path.abspath(__file__))
    # stop_path = os.path.join(current_dir, "stop.py")
    # print("start run stop.py")
    # subprocess.run([sys.executable, stop_path], check=True)
    # print("successfully run stop.py")


if __name__ == "__main__":
    main()

