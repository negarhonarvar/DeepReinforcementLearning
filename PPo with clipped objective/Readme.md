## PPO with Clipped Objective
All sections of this algorithm are similar to the Adaptive KL method, but there are differences in the learning phase. This learning arises from the difference in how the loss is calculated, which has a different impact on updating the network parameters. Additionally, it has fewer hyperparameters.

Learning:
The computations above are based on the calculations mentioned for this section in the paper, which are as follows:
<p align = "center">
  <img src="https://github.com/user-attachments/assets/2bd97bbd-cbc5-4df1-87ce-8fd000989a85" width="400">
</p>

performance of this algorithm with diffrent eps_clip values are available in form of charts in related directory.

<p align = "center">
  <p> EPS = 0.1</p>
   <img src="https://github.com/user-attachments/assets/63cd526c-eb5d-47c2-a119-e40bf5f41b48" width="400">
    <img src="https://github.com/user-attachments/assets/0fbb68dc-368b-4572-a450-600ec6da9e1d" width="400">
 <p> EPS = 0.4</p>
   <img src="https://github.com/user-attachments/assets/5d658d38-41a1-41f5-9703-b46b04d5fb2f" width="400">
  <img src="https://github.com/user-attachments/assets/ba5d2b95-029a-4e86-b0bc-2a9e430f5ece" width="400">
</p>


https://github.com/user-attachments/assets/df08375b-dcc6-479c-9be9-bebb368fd232

