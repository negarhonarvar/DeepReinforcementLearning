# DeepReinforcementLearning
Deep Reinforcement Learning Course Assignments by DR. Armin Salimi Badr
the codes in this repository utilize qymnasium library environments.

## Prerequisites 📋
To successfully run the codes in this repository, you need to install:

    Gymnasium v1.0.0
   

# CartPole V1
### Objective
The goal of this environment is to balance a pole by applying forces in the left and right directions on the cart. It has a discrete action space:

    0: Push cart to the left
    1: Push cart to the right
### Observation Space
Upon taking an action, either left or right, an agent observes a 4-dimensional state consisting of:

    Cart Position
    Cart Velocity
    Pole Angle
    Pole Angular Velocity

A reward of +1 is granted to the agent at each step while the pole is kept upright. The maximum reward an agent can earn in a single episode is 500.

### Termination
The episode ends under the following conditions:

    Termination: Pole Angle is greater than ±12°
    Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    Truncation: Episode length exceeds 500 steps


# Lunar Lander
This environment is part of the Box2D environments which contains general information about the environment and  is a classic rocket trajectory optimization problem.

### Objective
According to Pontryagin’s maximum principle, it is optimal to fire the engine at full throttle or turn it off. This is the reason why this environment has discrete actions: engine on or off.

### Action Space 
There are four discrete actions available

    0: do nothing
    1: fire left orientation engine
    2: fire main engine
    3: fire right orientation engine

### Observation Space
The state is an 8-dimensional vector: the coordinates of the lander in x & y, its linear velocities in x & y, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.

# Swimmer Environment
The Swimmer environment in MuJoCo is a reinforcement learning environment where the goal is to control a multi-jointed swimmer to move forward quickly in a two-dimensional fluid environment. The swimmer is essentially a small robotic entity with a simple body structure, consisting of a head and multiple tail-like segments. As described on the reference site:

"One rotor joint connects three or more segments ('links') and exactly two rotors ('rotors') to form a linear chain of articulation joints."    

### Objective:
The primary goal of the actor in the Swimmer environment is to move to the right along the horizontal axis of a two-dimensional plane by activating its rotors. When a trajectory ends, the agent always starts its new trajectory from a fixed starting point. The reward function typically rewards the agent based on the horizontal distance covered by the swimmer in each time step, and it is accompanied by penalties to prevent excessive use of activation forces. Therefore, the agent must learn how to propel itself forward effectively and efficiently.for further information on this environment, visit gymnasium.

### Problem parameters:

    n: number of body parts
    mi: mass of part i (i ∈ {1…n})
    li: length of part i (i ∈ {1…n})
    k: viscous-friction coefficient

### Action Space

    Num = 0 : Torque applied on the first rotor (-1,1)
    Num = 1 : Torque applied on the second rotor (-1,1)
    
### Observation Space

    qpos (3 elements by default): Position values of the robot’s body parts.
    qvel (5 elements): The velocities of these individual body parts (their derivatives).




