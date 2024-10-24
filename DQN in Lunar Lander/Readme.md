# DQN
The components of this algorithm have been implemented similarly to what we did in DQN for CartPole, but with some differences:

 - The neural network is built with a greater number of neurons, but it has one fewer layer and a simpler structure compared to the previous neural network.
 - Instead of the Model_TrainTest class, we have rewritten its two main components—namely, the loop where the training process occurs and
   the testing—as two separate functions in the main class. Here, once the training process is complete (which can happen in two cases: either the average score over an interval is greater than or equal to 200, or the maximum allowed episodes are reached), the neural network weights are saved, and a score chart is plotted at the end of each episode.
<p align = "center">
    <img src="https://github.com/user-attachments/assets/8ba5175f-262d-431d-9916-e7d362de3196" width = "400">
</p>

## Results
DQN has better Convergance rate and estimation accuracy comparing to D3QN , but not as accurate and fast as enhanced D3QN.

<p align = "center">
    <img src="https://github.com/user-attachments/assets/36631dea-ebfd-4a23-932e-de7bde6291d5" width = "400">
</p>



https://github.com/user-attachments/assets/636dab2d-a901-4b34-8645-154c7ef5a718

