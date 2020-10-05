import numpy as np

# Load field noise
field_noise1 = np.load("field_noise/field_noise1.npy")
#field_noise2 = np.load("field_noise/field_noise2.npy")
#field_noise3 = np.load("field_noise/field_noise3.npy")
#field_noise4 = np.load("field_noise/field_noise4.npy")
#field_noise5 = np.load("field_noise/field_noise5.npy")
#field_noise = np.concatenate((field_noise1, field_noise2, field_noise3, field_noise4, field_noise5), axis=1)

# Load gaussiand noise
gauss_noise = np.load("gaussNoise/gaussianNoise.npy")

class Config(object):
    def __init__(self):
        self.output = "Texas"
        self.field_noise = field_noise
        self.gauss_noise = gauss_noise
        self.PATH_TO_DATA = "TFRs"
        self.MAIN_CHECKPOINTS = "training_checkpoints"
        self.FROM_CHECKPOINT = False
        self.CHECK_FOLDER = ""
        self.BATCH_SIZE = 20
        self.N_TRACES = 97
        self.N_TIMESAMPLES = 1401
        self.MODEL_TYPE = 'texas'
        self.LABEL = 'Unet_Hypo'
        self.EVAL_STEPS = 20
        self.TRAIN = "train"
        self.TEST = "test"
        self.LEARNING_RATE =  0.01
