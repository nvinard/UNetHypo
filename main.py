import tensorflow as tf
from func import train
import config

#tf.config.list_physical_devices('GPU')

# Load cfg file
cfg = config.Config()

# Call training loop
train.train_and_evaluate("/tudelft.net/staff-umbrella/ConvNetHypo/{}".format(cfg.output), cfg)
