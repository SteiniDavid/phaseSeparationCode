# phaseSeparationCode

This is my code for processing the final time steps of Tom's active matter simulations. The general pipline is that I first take the .gsd files, strip them down to the last time step, run them through ovito (using ovitos), then match the photos to a value indicating if they are phase separated or not. Once that side of things is done the data is converted into three tfrecord file's: train set, test set, and validation set. These are then going to be fed to train a classification algorithm. 
