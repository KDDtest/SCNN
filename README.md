# Structured Component-based Neural Network
This is an implementation of SCNN.

# Requirements
Python 3.7  
numpy >= 1.18.5  
pandas >= 1.0.3  
torch >= 1.12.0  
pytorchts == 0.6.0  
h5py

 
## Model Training
```
python -u main.py -mode train -short_term 8 -long_term 144 -n_local_input 2 -cuda 0
```
### Arguments 
short_term: length of short term.  
long_term: length of long term.  
dataset: dataset name.  
version: version number.  
hidden_channels: number of hidden channels.  
n_pred: number of output steps.  
n_his: number of input steps.  
n_local_input: kernel size of causual convolution.  
n_layers: number of hidden layers.  
cuda: cuda device id.  

## Model Evaluation
```
python -u main.py -mode eval -cuda 0
```
