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


