# Adaptive PID Regulation by Neural Networks

**⚠️ This project is still in development ⚠️**

## Contents
1. [Abstract](#abstract)
2. [Scripts](#scripts)
    * [Learning](#learning)
    * [Comparisons](#comparisons)
    * [Simulations](#simulations)
3. [Acknowledgements](#acknowledgements)

## Abstract
This bachelor's thesis focuses on the use of neural networks for the automatic tuning of PID controllers. The main goal is to develop and implement a system combining LSTM and RBF networks for adaptive control. The work includes a theoretical analysis of the issue, the design of a custom solution, and its experimental verification on two types of systems – a cart and a thermal system. The results demonstrate that the proposed approach can, in some cases, outperform conventional PID controller tuning methods, particularly when working with nonlinear systems. Additionally, limitations and practical challenges of the implementation are discussed, including issues with numerical stability and computational complexity.

## Scripts
Each script is located in a different folder. The scripts are divided into three main categories: `learning`, `comparisons`, and `simulations`.

### `learning`
These scripts are used to train the neural networks.

* `sys_rbf_${system_name}.py` - These scripts train a neural network to approximate the system dynamics of the system with the name `${system_name}`. The neural network is a Radial Basis Function (RBF) network.
* `pid_lstm_sys_rbf_${system_name}.py` - These scripts train a neural network to tune the PID controller parameters. The neural network is a Long Short-Term Memory (LSTM) network. The system dynamics are approximated by an RBF network.

```py
python3 -m learning.<script name>
python3 -m learning.sys_rbf_trolley
python3 -m learning.pid_lstm_sys_rbf_trolley
```


### `comparisons`
These scripts are used to compare the performance of the PID controller with the neural network tuned PID controller.

* `pid_lstm_trolley.py` - This script compares the performance of the PID controller with the neural network tuned PID controller for the trolley system.

```py
python3 -m comparisons.<script name>
python3 -m comparisons.pid_lstm_trolley
```

### `simulations`
These scripts are used for measuring metrics of the system behavior:
- overshoot
- settling time
- response time
- nyquist plot
- bode plot

```py
python3 -m simulations.<system_name>
python3 -m simulations.trolley
```


## Acknowledgements
This project is my Bachelor's thesis. I would like to thank my supervisor, Ing. Cyril Oswald, Ph.D., for his guidance and support. You can find more about his work [here](https://orcid.org/0000-0001-5268-2785).
