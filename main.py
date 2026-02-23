import torch
from config import Config
from src.data_loader import make_bucketed_loader
from src.model import DCEVAE
from src.train import train_dcevae
from src.test import test_dcevae
from src.utils import parse_args

def main():
  args = parse_args()
