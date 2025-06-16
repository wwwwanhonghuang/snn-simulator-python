# Spiking Neural Network Simulator (Pure Python)

This repository provides a pure Python implementation of a spiking neural network (SNN) simulator.

An example of how to use the simulator is included in [`example.ipynb`](./example.ipynb) in the root directory.

---

## ✅ Implemented Modules and Functions

### 1. Neuron Models
- **LIF Neuron**: Classical leaky integrate-and-fire model.
- **Fixed Spike Train Neuron**: Emits spikes based on a predefined spike train.
- **Hodgkin-Huxley Neuron**: Classical Hodgkin-Huxley model.
- **Izhikevich Neuron**: Classical Izhikevich model.
- **Poisson Process Neuron**: Emits spikes according to a Poisson process.

### 2. Plotting Utilities
- Basic tools to visualize spike trains, membrane potentials, etc.

### 3. Synapse Models
- **Exponential Synapse Models**: Supports single and double exponential decay.
- **Delay Connection Model**: Introduces synaptic transmission delay.

### 4. Connection Schemes
- **All-to-All**: Fully connects two neuron populations.
- **One-to-One**: Connects the *i*-th neuron in the pre-synaptic group to the *i*-th in the post-synaptic group.
- **Probabilistic Connection**: Connects neurons randomly with a specified probability.
- **Custom Connection**: Allows user-defined connection patterns via edge lists.

### 5. Network Building
- A network builder is provided to easily instantiate and configure SNNs.

### 6. Monitoring
- Fine-grained monitor class for tracking states and gradients during simulation.
- Enables pseudo-gradient recording for training via gradient descent.

---

## ⚠️ Limitations

- This framework is **resource-consuming** and relatively slow due to (1) the inefficiency recording and retreiving run-time states, (2) no computational optimizations, e.g., exploit parallelism.
- Suited for educational or prototyping purposes rather than high-performance applications.

### 🛠 Future Improvements
- Refactor the recording mechanism for better efficiency.
- Implement simulation parallelization.
- Add distributed computing support, including:
  - Network partitioning (include partitioning optimization algorithms)
  - Code generation and optimization
  - Partition-to-device mapping

---

## 📌 Note
For high-performance applications, consider using other frameworks like spikingjelly, SpiNNaker, SpikBindsNET, Brian2, or NEST. However, this simulator serves as a learning tool for understanding the internals of SNNs.
