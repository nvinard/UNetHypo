# UNetHypo
In this repository we save the codes used to train a U-Net with tensorflow to localize microseismic events as a 3-D Gaussian distribution given seismic waveforms as input.
The model is trained in Tensorflow by calling the function main.py
The utils folder contains code to generate the U-Net model, the data augmentations, the data pipeline and the script to train the model.
The train and test data are stored in TFRecords. No data is provided here, because of the large size of the files. 
