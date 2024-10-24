# SARSA
In the SARSA algorithm, there is no longer a neural network; instead, a concept called a Q-Table is used. The Q-Table is applied by examining each experience in the format of current state, current action, reward, next state, and next action. Given the continuous nature of the environment, it is necessary to first discretize the environment into a discrete one based on a set of hyperparameters with an appropriate approximation. Then, we create a table, the size of which is determined based on the hyperparameters like the number of bins or buckets, the states of the observation space, and the action space. This table will then be filled with values according to the equation Q(s,a) <- Q(s,a) + alpha(yi - Q(s,a)) (TD), and it will be the basis for our decision-making to select the optimal action in each state.

The other parts of the code, such as the epsilon-greedy policy and its application in decision-making, training, and testing, as well as seeding, are similar to the DQN algorithm. However, in this case, the hyperparameter values play a significant role, and their final values are as follows:
- Epsilon = 1
- Epsilon decay = 0.999
- Epsilon min = 0.01
- Learning_rate = 0.25
- Discount_factor = 0.995

Additionally, the hyperparameters related to the discretization of the environment, which were mentioned in the previous section, are also included.


<img src="https://github.com/user-attachments/assets/b07c19fe-e6c7-4a72-9acf-d882f4226f74" width ="400">


https://github.com/negarhonarvar/DeepReinforcementLearning/assets/79962938/bb188e6e-13fc-4080-88e2-964296934c04
