import torch
from config import Config
from dcevae.data_loader import make_bucketed_loader
from dcevae.model import DCEVAE
from dcevae.train import train_dcevae
from dcevae.test import test_dcevae
from utils import parse_args

def main():
  args = parse_args()
