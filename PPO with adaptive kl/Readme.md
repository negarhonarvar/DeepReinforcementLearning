# PPO with Adaptive KL Penalty
One way to formulate the objective is to assume the stated constraint as a multiplier of the objective function and subtract it from 𝐿(𝜃). The value below corresponds to the calibrated advantage, which can be calculated based on either the old policy or the current policy, and its calibrated amount is based on the probability rate in both policies. β controls the penalty weight. This parameter penalizes the objective function if the new policy differs from the old policy. By borrowing a page from the trust region, we can dynamically adjust β. 𝑑 in the equation below represents the KL divergence between the old and new policies. If this value exceeds a target threshold, we decrease 𝛽. Similarly, if this value falls below another target threshold, we expand the trust region. Thus, the dynamic adjustment of the trust region by establishing a lower confidence bound ensures that we never experience performance collapse. Additionally, the initial value of β does not create significant sensitivity, as the algorithm can quickly adjust it.
<p align="center">
 <img src="https://github.com/user-attachments/assets/c4ddb067-ecbf-4237-bd77-b03697aebac8" width ="400">
</p>

<p align="center">
 <img src="https://github.com/user-attachments/assets/0511e23c-3540-42c5-8f11-84c71c9d5679" width="400">
</p>

The algorithm that we follow in this method is described below:
<p align="center">
 <img src="https://github.com/user-attachments/assets/bfe83e0a-25bc-4c29-8c3c-b7edc76062e9" width="400">
</p>
Based on the above structure and the details provided in the Adaptive KL PPO paper, we implement it as follows:

### Neural Network: 
The neural network is similar to what is described in the paper for Mujoco environments and is based on Actor-Critic and TD(0). However, to introduce non-linearity in the intermediate layers, ReLU is used, and a hyperbolic tangent is used in the final layer. Additionally, to generalize this model, one layer of the Actor neural network has been reduced compared to the introduced neural network.
In this environment, we assume that advantage and value follow a normal distribution. Therefore, in the forward function, the normal distribution values corresponding to the value function are calculated. 
### Hyperparameters: 
These values are set as follows based on the suggested values in the paper related to this algorithm.
#### Memory:
We have a simple implementation of memory here, where in each epoch, a sample of size 32 is collected and used for training. At the end of every 𝑘 epochs, the memory needs to be completely cleared because the trajectories collected with the previous policy are no longer useful. Additionally, here we consider the batch size as 32 instead of the recommended value of 64 from the paper.
#### Training:
In the train function, for each episode sampled from the environment, we use the select_action function in the PPO class to choose the best action in the current state according to the current probability distribution. For each step, we collect the reward received from the environment along with the state and whether the episode has ended in memory. After a thousand steps and upon the completion of a trajectory, this trajectory is stored in memory, and the learning process is carried out using the learn function. By adjusting the parameters of the target network, which estimates the normal distribution of actions, we expect improved action selection in the next trajectory.
#### Action Selection:
During action selection based on the state and the old policy network, we attempt to choose the action that yields the highest advantage in the current state, using the probabilities given by the distribution. The details regarding this selection are then stored in memory to be used during the learning process.
#### Learn Function:
In this function, for each epoch, a sample is drawn from memory, and we attempt to improve the neural network using this sample. The error is calculated for each sample based on the relationship mentioned in the paper, and the network parameters are updated for each sample.
<p align="center">
  <img src="https://github.com/user-attachments/assets/0cecf4fb-68ae-41ca-8b43-6bc3dd0697e0" width="400">
</p>

## Performance of this algorithm with diffrent target_kl 
<p align = "center">
 <p>KL = 0.1</p>
  <img src="https://github.com/user-attachments/assets/de7e1065-b710-4675-9886-694b95abf393" width="400">
  <img src="https://github.com/user-attachments/assets/ebe0be75-6420-4b17-af5d-e5358318ac61" width="400">
<p>KL = 0.05</p>
  <img src="https://github.com/user-attachments/assets/39e7df20-fda8-4d37-bbbe-cadf714c652d" width="400">
  <img src="https://github.com/user-attachments/assets/1a0b0723-9188-4034-9402-2130d1ea453d" width="400">
<p>KL = 0.005</p>
  <img src="https://github.com/user-attachments/assets/67581685-87eb-4694-b70f-865069b292b2" width="400">
  <img src="https://github.com/user-attachments/assets/a2532fb0-775a-4c88-9b83-7fc68ce5dc98" width="400">
<p>Final Results</p>
  <img src="https://github.com/user-attachments/assets/b121ff41-6f49-4eb1-9c38-77b9fe1ddb24" width="400">
  <img src="https://github.com/user-attachments/assets/4529e6e7-ab3b-40aa-9702-4f68825dbb85" width="400">
</p>



https://github.com/user-attachments/assets/15dc86a7-4e38-4dbc-8014-224ffaf53c2b

