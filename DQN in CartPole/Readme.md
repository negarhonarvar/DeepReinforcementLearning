## DQN
I provided an example of an agent exploring its environment and gradually converging to best solution in 2000 episodes.the hyperparametes are set according to an article in Research gate about 'Table of best hyperparameter for Cartpole-v1 Hyperparameter QRDQN with standard deviation penalisation PPO' 
<p align="center">
   <img src="https://github.com/negarhonarvar/DeepReinforcementLearning/assets/79962938/f2e606d6-793f-4436-9082-6b69207b2ba1" width ="400">
</p>
and the nn is implemented based on "Deep Q-learning (DQN) Tutorial with CartPole-v0" article on medium.com . the rest of the code is a modified version of course's head TA Mr. Mehdi Shahbazi .

## Applying boltzman policy 
When adopting an epsilon-greedy policy, all actions have an equal probability of being selected during exploration. As a result, we may choose actions that are very poor or irrelevant, leading to bad and misleading experiences, ultimately resulting in poor training.

To address the issues with the epsilon-greedy policy, the Boltzmann exploration policy was introduced. This policy is based on considering the value of each (state, action) pair according to the knowledge acquired from the environment during random selection, allowing for more intelligent choices. Consequently, in this policy, better actions in each state have a higher chance of being selected during exploration, and random selection is no longer uniform. The balance between exploration and exploitation is managed using a temperature parameter, such that with an increase in temperature, the Boltzmann policy becomes more inclined to select actions with lower value.

To implement this policy in the DQN algorithm, the parts related to the epsilon-greedy policy are removed, and the following sections are added:
<p align="center">
   <img src="https://github.com/user-attachments/assets/47c29302-188f-4dd2-93df-5004462b27b8">
</p>

https://github.com/user-attachments/assets/483719a6-9560-41ff-9930-1852fb684b30


Implementation of softmax (fixed temperature):
We set the temperature equal to 1.
In situations where the epsilon-greedy policy leads to the beginning of convergence at step 1600, the Boltzmann policy starts converging to the optimal solution at step 300, with much less variance.

https://github.com/user-attachments/assets/40d9c3a8-c68d-469d-8c88-15f609c72ce5

Necessary Implementations for Boltzman:
Modifications in applying the softmax function.
Implementation of a function to adjust the temperature.
In this implementation, in addition to the temperature hyperparameter, we also need the temperature decay hyperparameter and the minimum temperature hyperparameter. Therefor we set tau = 1.0 , tau_min = 0.01 and tau_decay = 0.995

https://github.com/user-attachments/assets/a75517e5-22c8-44d1-a567-e8075e65f943


