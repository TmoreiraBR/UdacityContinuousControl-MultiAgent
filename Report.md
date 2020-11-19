<<<<<<< HEAD
[//]: # (Image References)

[image2]: https://github.com/TmoreiraBR/UnityMLAgents1stProject/blob/main/TrainedResults.jpg  "Test Agent"

# Report for Project 1: Navigation

### Introduction

For solving this project a Deep Q-Learning Algorithim, with two neural networks (local and target) and  was utilized.

The Algorithim, based on [[1]](#1), utilizes the "local" neural network as a function approximation for the action-value function:

<img src="https://render.githubusercontent.com/render/math?math=\hat{q}_{\pi}(s,a,\theta)">,

where the function approximation is parametrized by the neural network weights <img src="https://render.githubusercontent.com/render/math?math=\theta">.

Assuming an already trained neural network, the optimal policy can be obtained by sampling actions that maximize the action value function:

<img src="https://render.githubusercontent.com/render/math?math=arg max_a \hat{q}_{\pi^*}(s,a,\theta)">.

Since the parametrized action-value function is continuous, on-policy learning approch, such as updating a Q-table, could lead to reinforcing correlated state-actions values (i.e. actions that lead to known states), which is bad for generalization.

To avoid this, the Algorithim utilizes Experience Replay, where batches of <img src="https://render.githubusercontent.com/render/math?math=<s, a, r', s', a'>"> are sampled randomly, breaking the correlation (in this report ' denotes a forward time-step).

Also, to avoid unstable learning a second network, called "target" network is utilized during the update step. This target network has its weights frozen during the update, and after some number of steps (hyperparameter) its weights are updated to be the same as the "local" network:

<img src="https://render.githubusercontent.com/render/math?math=\Delta \theta = \alpha (sum(r',  \gamma max_a \hat{q}(s,a,\theta_{frozen})) - \hat{q}(s,a,\theta)) \nabla_{\theta} \hat{q}(s,a,\theta)">,

where <img src="https://render.githubusercontent.com/render/math?math=\alpha"> is the learning-rate and <img src="https://render.githubusercontent.com/render/math?math=\gamma"> the discount factor.

## Algorithim

Detailed Algorithim pseudocode, edited from [[1]](#1)

**Algorithm 1: deep Q-learning with experience replay**
* Initialize replay memory **D** to capacity **N**
* Initialize parametrized action-value function <img src="https://render.githubusercontent.com/render/math?math=\hat{q}(s,a,\theta)"> (local neural network) with random weights <img src="https://render.githubusercontent.com/render/math?math=\theta"> 
* Initialize parametrized target action-value function <img src="https://render.githubusercontent.com/render/math?math=\hat{q}(s,a,\theta_{frozen})">.  with weights <img src="https://render.githubusercontent.com/render/math?math=\theta_{frozen}"> 
* **For** episode = 1,M **do**
  * Start environment and sample initial state <img src="https://render.githubusercontent.com/render/math?math=s">
  * **For** t = 1,T **do**
    * With probability <img src="https://render.githubusercontent.com/render/math?math=\epsilon">  select a random action <img src="https://render.githubusercontent.com/render/math?math=\a"> 
    * otherwise choose action from current policy <img src="https://render.githubusercontent.com/render/math?math=\a = arg max_a \hat{q_{\pi}}(s,a,\theta)">
    * Execute action <img src="https://render.githubusercontent.com/render/math?math=\a"> in Unity environment and observe reward <img src="https://render.githubusercontent.com/render/math?math=\r"> and next state <img src="https://render.githubusercontent.com/render/math?math=\s'">
    * Set <img src="https://render.githubusercontent.com/render/math?math=\s' \leftarrow s">
    * Store transition tuple <img src="https://render.githubusercontent.com/render/math?math=<s, a, r', s'>"> in **D**
    * Sample random minibatch of transitions <img src="https://render.githubusercontent.com/render/math?math=<s, a, r', s'>"> from **D**
    * Set target q-value as <img src="https://render.githubusercontent.com/render/math?math=Q_{target} = sum(r',  \gamma max_a \hat{q}(s,a,\theta_{frozen}))">
    * Perform local network weights update with <img src="https://render.githubusercontent.com/render/math?math=\Delta \theta = \alpha (Q_{target} - \hat{q}(s,a,\theta)) \nabla_{\theta} \hat{q}(s,a,\theta)">
    * Every C steps update target neural network weights: <img src="https://render.githubusercontent.com/render/math?math=\theta_{frozen} \leftarrow \theta">

## Hyperparameters and Neural Network Architecture

After a couple of attempts hyperparameter values that could reach the minimum of 13+ cumulative rewards in 100 episodes were obtained. These are:

Hyperparameter value  | Description
------------- | -------------
n_episodes=900  | maximum number of training episodes
max_t=1000  | maximum number of timesteps per episode
eps_start=1.0  | starting value of epsilon, for epsilon-greedy action selection
eps_end=0.01  | minimum value of epsilon
eps_decay=0.995  | multiplicative factor (per episode) for decreasing epsilon
BUFFER_SIZE = int(1e5)   | replay buffer size
BATCH_SIZE = 64 | minibatch size
GAMMA = 0.99   | discount factor
TAU = 1e-3  | Value between 0 and 1 -> The closer to 1 the greater, the target weights update will be (if TAU = 1, then <img src="https://render.githubusercontent.com/render/math?math=\theta_{frozen} = \theta">)
LR = 5e-4  | learning rate for updating policy network weights
UPDATE_EVERY = 4  | how often to update the target network 

Neural Network Layers (local and target networks)  | Number of nodes 
------------- | -------------
Input Layer  | 37 Input States
1st Hidden Layer  | 64 (followed by ReLu Activation function)
2nd Hidden Layer  | 128 (followed by ReLu Activation function)
3rd Hidden Layer  | 64 (followed by ReLu Activation function)
Output Layer  | 4 Discrete Actions (left, right, forward, backwards)


## Results

A plot for the mean return every 100 episodes is shown below. We see that our trained Agent is capable of surparsing the requirements of 13+ rewards after approximately 400 Episodes. The weights of the neural network are saved in model_weights_checkpoint.pth.

![Test Agent][image2]


## Future Ideas

Implement and compare current results with advancements that were proposed in the literature, such as: double DQN, dueling DQN and prioritized experience replay.

## References
<a id="1">[1]</a> 
Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G. and Petersen, S., 2015. Human-level control through deep reinforcement learning. nature, 518(7540), pp.529-533.
