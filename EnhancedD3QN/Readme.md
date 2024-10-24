# Enhanced DQN
To implement this algorithm, the following two changes need to be made to the D3QN algorithm:
For the D3QN_Agent class, two parameters—learning_rate and discount_factor—are defined.
In the hard_update function, when the weights of the target network are updated, the discount_factor and learning_rate parameters are also updated according to the relationships outlined in the paper.
<p align="center">
   <img src="https://github.com/user-attachments/assets/87135e17-ef1f-47ee-882f-b9be40e19871" width = "400">
</p>

## Results
Comparing to DQN and D3QN, this algorithm has fastest convergance rate and estimation pression.
<p align="center">
   <img src="https://github.com/user-attachments/assets/08b5177f-6ecf-4173-a09c-fb2137753663" width = "400">
</p>

