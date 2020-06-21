## ANN

A module to simplify the establishment of neural networks


## Installation

Run the following to install:
```python
pip install Advanced-Neural-Network
```


## Usage
```python
from ANN import *

####Initialise network####
my_network = neural_network()

####Set INPUT_LAYER####
#Add Input neurons
my_network.Add_Input_Neuron("speed_neuron","Input Cell")
my_network.Add_Input_Neuron("pos_neuron","Input Cell")

####Set HIDDEN_LAYER####
#Add Hidden neurons
my_network.Add_Hidden_Neuron("neuron1_layer1","Hidden Cell","Sigmoid")
my_network.Add_Hidden_Neuron("neuron1_layer2","Hidden Cell","Linear",alpha=1)
my_network.Add_Hidden_Neuron("neuron2_layer2","Hidden Cell","Sigmoid",biais=0.7)

####Set OUTPUT_LAYER####
my_network.Add_Output_Neuron("output1","Output Cell","Sigmoid")

####Set Bridge####
bridge_list = [
    ["speed_neuron","neuron1_layer1"],#Bridge from speed_neuron to neuron1_layer1
    ["pos_neuron","neuron1_layer1"],
    ["pos_neuron","neuron2_layer2"],
    ["neuron1_layer1","neuron1_layer2"],
    ["neuron1_layer1","neuron2_layer2"],
    ["neuron1_layer2","output1"],
    ["neuron2_layer2","output1"]
]
my_network.Add_Bridge(bridge_list)

#####TRAIN NEURONAL NETWORK#####
#return 1 when speed_neuron and pos_neuron is one
inputs = [
    [0,0],
    [0,1],
    [1,1],
    [1,0],
    [1,1]
    ]
expected = [
    [0],
    [0],
    [1],
    [0],
    [1]
    ]
#set learning_rate
learning_rate = 0.01
#set number of epoch
nb_epoch = 2000

#start training
my_network.train(inputs,expected,learning_rate,nb_epoch,display=True)

#predict output value
print(my_network.predict([0,1]))
print(my_network.predict([1,1]))
```


```bash
$ pip install -e .[dev]
```