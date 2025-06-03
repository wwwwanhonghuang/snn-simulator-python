This repository provide an implementation of spiking neural network simulator in pure python.
The example usage is provided in the `example.ipynb` in the root of this repository.

Currently follow functions or modules are implemented:
1. Neuro Models
+ **LIF neuron**: A classical leaky-and-fire neuron model.
+ **Fixed spike train neuron**: A kind of neuron that emit spikes according to pre-specified spike train.
+ **Hodgkin-Huxley neuron**: A classical Hodgkin-Huxley neuron model.
+ **Izhikevich neuron**: A classical Izhikevich neuron model.
+ **Possion point process models**

2. Plot Utilities

3. Synapse Models
+ **Exponential synapse models**: Single-exponential synapse and double-exponential synapse models are implemented.
+ **Delay connection model**

4. Connections 
+ **All-to-all connection**: connect two neuron populations in an all-to-all way.
+ **One-to-one connection**: i-th neuron of pre-synapse neuron population is connect to i-th post-synapse neuron population.
+ **Possibility connection**: neuron are connected randomly with a specified probability.
+ **Custom connection**: connect pre-post neurons in a custom way, with specifying the connection edges.

5. Network Building
A network builder is provided to build a network instance.

6. Monitor
A simulator monitor class is provided to fine-grain record necessary information in the information, and can be accessed in the simulation. 

Currently, the training of network is by recording presudo-gradient utilize the monitor, and 
access the these gradient in the simulation to perform gradient-desending-based optimization.