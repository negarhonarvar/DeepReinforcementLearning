# DeepReinforcementLearning
Deep Reinforcement Learning Course Assignments by DR. Armin Salimi Badr
the codes in this repository utilize qymnasium library environments.
# CartPole V1
The goal of this environment is to balance a pole by applying forces in the left and right directions on the cart. It has a discrete action space:

    0: Push cart to the left
    1: Push cart to the right

Upon taking an action, either left or right, an agent observes a 4-dimensional state consisting of:

    Cart Position
    Cart Velocity
    Pole Angle
    Pole Angular Velocity

A reward of +1 is granted to the agent at each step while the pole is kept upright. The maximum reward an agent can earn in a single episode is 500.

The episode ends under the following conditions:

    Termination: Pole Angle is greater than ±12°
    Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    Truncation: Episode length exceeds 500 steps


# Lunar Lander
This environment is part of the Box2D environments which contains general information about the environment and  is a classic rocket trajectory optimization problem. According to Pontryagin’s maximum principle, it is optimal to fire the engine at full throttle or turn it off. This is the reason why this environment has discrete actions: engine on or off. for more information on Lunar Lander environment visit gymnasium. we will examine performance of 3 common variations of DQN in this environment.


## Enhanced DQN
To implement this algorithm, the following two changes need to be made to the D3QN algorithm:
For the D3QN_Agent class, two parameters—learning_rate and discount_factor—are defined.
In the hard_update function, when the weights of the target network are updated, the discount_factor and learning_rate parameters are also updated according to the relationships outlined in the paper.

<img src="https://github.com/user-attachments/assets/87135e17-ef1f-47ee-882f-b9be40e19871" width = "400">

# SWIMMER ENVIRONMENT
The Swimmer environment in MuJoCo is a reinforcement learning environment where the goal is to control a multi-jointed swimmer to move forward quickly in a two-dimensional fluid environment. The swimmer is essentially a small robotic entity with a simple body structure, consisting of a head and multiple tail-like segments. As described on the reference site:

"One rotor joint connects three or more segments ('links') and exactly two rotors ('rotors') to form a linear chain of articulation joints."

Objective:
The primary goal of the actor in the Swimmer environment is to move to the right along the horizontal axis of a two-dimensional plane by activating its rotors. When a trajectory ends, the agent always starts its new trajectory from a fixed starting point. The reward function typically rewards the agent based on the horizontal distance covered by the swimmer in each time step, and it is accompanied by penalties to prevent excessive use of activation forces. Therefore, the agent must learn how to propel itself forward effectively and efficiently.for further information on this environment, visit gymnasium.
## PROXIMAL POLICY OPTIMIZATION
We will implement two agents for two common version of PPO algorithm but before proceeding to that we shall take a quick look at what Proximal policy optimization is.
Gradient-based policy methods tended to suffer from divergence, which we saw could be addressed by calculating a matrix of second-order derivatives and its inverse as follows:

<img src="https://github.com/user-attachments/assets/8e71548b-00ad-4caf-a777-9fd8c2224a46" width="400">

<img src="https://github.com/user-attachments/assets/a4d8d654-3e68-41c5-b259-ff386c0c7e4f" width="400">

However, this method is very costly and practically infeasible in large environments with high complexity. To address this issue, the Proximal Policy Optimization (PPO) algorithm was introduced. The main approach in PPO to solve this problem involves using first-order derivatives combined with the application of several soft constraints. Sometimes, we may continue the learning process with a poor policy decision; therefore, we proceed with first-order derivatives similar to stochastic gradient descent. However, adding soft constraints to the objective function ensures that optimization occurs within a trust region, thereby reducing the likelihood of making poor decisions.
It is worth mentioning that in this method, we use the advantage function instead of the Q(s,a) function because it leads to less variance in the approximation.
## PPO with Adaptive KL Penalty
One way to formulate the objective is to assume the stated constraint as a multiplier of the objective function and subtract it from 𝐿(𝜃). The value below corresponds to the calibrated advantage, which can be calculated based on either the old policy or the current policy, and its calibrated amount is based on the probability rate in both policies. β controls the penalty weight. This parameter penalizes the objective function if the new policy differs from the old policy. By borrowing a page from the trust region, we can dynamically adjust β. 𝑑 in the equation below represents the KL divergence between the old and new policies. If this value exceeds a target threshold, we decrease 𝛽. Similarly, if this value falls below another target threshold, we expand the trust region. Thus, the dynamic adjustment of the trust region by establishing a lower confidence bound ensures that we never experience performance collapse. Additionally, the initial value of β does not create significant sensitivity, as the algorithm can quickly adjust it.

<img src="https://github.com/user-attachments/assets/c4ddb067-ecbf-4237-bd77-b03697aebac8" width ="400">

<img src="https://github.com/user-attachments/assets/0511e23c-3540-42c5-8f11-84c71c9d5679" width="400">

The algorithm that we follow in this method is described below:

<img src="https://github.com/user-attachments/assets/bfe83e0a-25bc-4c29-8c3c-b7edc76062e9" width="400">

Based on the above structure and the details provided in the Adaptive KL PPO paper, we implement it as follows:

Neural Network: 
The neural network is similar to what is described in the paper for Mujoco environments and is based on Actor-Critic and TD(0). However, to introduce non-linearity in the intermediate layers, ReLU is used, and a hyperbolic tangent is used in the final layer. Additionally, to generalize this model, one layer of the Actor neural network has been reduced compared to the introduced neural network.
In this environment, we assume that advantage and value follow a normal distribution. Therefore, in the forward function, the normal distribution values corresponding to the value function are calculated.
Hyperparameters: 
These values are set as follows based on the suggested values in the paper related to this algorithm.
Memory:
We have a simple implementation of memory here, where in each epoch, a sample of size 32 is collected and used for training. At the end of every 𝑘 epochs, the memory needs to be completely cleared because the trajectories collected with the previous policy are no longer useful. Additionally, here we consider the batch size as 32 instead of the recommended value of 64 from the paper.
Training:
In the train function, for each episode sampled from the environment, we use the select_action function in the PPO class to choose the best action in the current state according to the current probability distribution. For each step, we collect the reward received from the environment along with the state and whether the episode has ended in memory. After a thousand steps and upon the completion of a trajectory, this trajectory is stored in memory, and the learning process is carried out using the learn function. By adjusting the parameters of the target network, which estimates the normal distribution of actions, we expect improved action selection in the next trajectory.
Action Selection:
During action selection based on the state and the old policy network, we attempt to choose the action that yields the highest advantage in the current state, using the probabilities given by the distribution. The details regarding this selection are then stored in memory to be used during the learning process.
Learn Function:
In this function, for each epoch, a sample is drawn from memory, and we attempt to improve the neural network using this sample. The error is calculated for each sample based on the relationship mentioned in the paper, and the network parameters are updated for each sample.

<img src="https://github.com/user-attachments/assets/0cecf4fb-68ae-41ca-8b43-6bc3dd0697e0" width="400">

performance of this algorithm with diffrent target_kl values are available in form of charts in related directory.
## PPO with Clipped Objective
All sections of this algorithm are similar to the previous method, but there are differences in the learning phase. This learning arises from the difference in how the loss is calculated, which has a different impact on updating the network parameters. Additionally, it has fewer hyperparameters.

Learning:
The computations above are based on the calculations mentioned for this section in the paper, which are as follows:

<img src="https://github.com/user-attachments/assets/2bd97bbd-cbc5-4df1-87ce-8fd000989a85" width="400">


performance of this algorithm with diffrent eps_clip values are available in form of charts in related directory.

<img src="https://github.com/user-attachments/assets/ea0ea8a7-3457-4c1f-82bc-7d92011924b9" width="400">

<img src="https://github.com/user-attachments/assets/97f65453-bff4-436f-852d-ec1047fc9dc2" width="400">

