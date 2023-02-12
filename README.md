# Structured Component-based Neural Network
This is an implementation of SCNN. The skeleton of the code refers to [ST-Norm](https://github.com/JLDeng/ST-Norm).

# Requirements
Python 3.7  
Numpy >= 1.17.4  
Pandas >= 1.0.3  
Pytorch >= 1.4.0  

h5py

 
## Model Training
```
python main.py -mode train -short_term 8 -long_term 144 -n_local_input 2 -cuda 0
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
python main.py -mode eval 0
```
