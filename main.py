import tensorflow as tf
from utils import train
import config

# Load cfg file
cfg = config.Config()

# Call training loop
train.train_and_evaluate(output_dir=cfg.output, cfg=cfg)
