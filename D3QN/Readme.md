
## D3QN
The main differences in the implementation of this algorithm compared to DQN can be summarized in two points:
In this algorithm, the neural network is derived from Dueling DQN. In Dueling DQN, the parameters of the two neural networks Q(s,a) are initialized with random final weights. Unlike traditional DQN, each network in Dueling DQN is split at a certain point into two separate streams—one for estimating the state-value function V(s) and the other for estimating the advantage function A(s,a). Additionally, we set epsilon to ϵ = 1. We use Stochastic Gradient Descent (SGD) to update the main network and minimize loss.
Here, a D3QN_Agent is used, whose main difference from the traditional agent lies in the part related to the update_model function
D3QN results are listed below :

<p align="center">
  <img src="https://github.com/user-attachments/assets/000d8ee0-d653-4375-b109-4efe7922a5bf" width ="400">
</p>
