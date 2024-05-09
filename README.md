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
in my code i provided an example of an agent exploring its environment and gradually converging to best solution in 2000 episodes.the hyperparametes are set according to an article in Research gate about 'Table of best hyperparameter for Cartpole-v1 Hyperparameter QRDQN with standard deviation penalisation PPO' ![SharedScreenshot](https://github.com/negarhonarvar/DeepReinforcementLearning/assets/79962938/f2e606d6-793f-4436-9082-6b69207b2ba1)

and the nn is implemented based on "Deep Q-learning (DQN) Tutorial with CartPole-v0" article on medium.com . the rest of the code is a modified version of course's head TA Mr. Mehdi Shahbazi .



https://github.com/negarhonarvar/DeepReinforcementLearning/assets/79962938/bb188e6e-13fc-4080-88e2-964296934c04

