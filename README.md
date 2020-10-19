# UNetHypo
In this repository we save the codes used to train a U-Net with tensorflow to localize microseismic events as a 3-D Gaussian distribution given seismic waveforms as input.
To start the training the function main.py needs to be executed. In the config.py file the paths to the data have to be given together with some additional information.
The utils folder contains code to generate the U-Net model, the data augmentations, the data pipeline and the script to train the model.
The train and test data are stored in TFRecords. No data is provided here, because of the large size of the files. 
