# **Policy-Gradient Based Model**
This repository contains implementations of various policy gradient methods to train reinforcement learning agents. The primary environment tested is the **BankHeist** environment. Below are details of each folder and their respective approaches.


## **PG-single**
The basic implementation of a policy gradient network using the REINFORCE algorithm to maximize Monte Carlo returns. The average episodic returns after 5000 episodes (tested on Colab) do not indicate any learning patterns. This suggests that the simple model is insufficient for training in the BankHeist environment.

## **PG-dual-fnn-a2c**
This folder implements a dual-network model designed to reduce variance and enhance training efficiency. It incorporates A **target network** and **actor-critic method** to stabilize learning.


## **PG-PPO**
This folder contains the PPO (Proximal Policy Optimization) implementation to optimize policy gradient learning. Unlike the previous models, this is the only one that effectively trains the agent for the BankHeist environment. Extensive hyper-parameter tuning was conducted for this approach.

### **Training & Tuning**
Four hyper-parameter combinations were tested to optimize training:  
1. `clip_coef = 0.3`, `ent_coef = 0.05`, `l_rate = 1.0e-3`, `v_coe = 0.5`  
2. `clip_coef = 0.1`, `ent_coef = 0.01`, `l_rate = 2.5e-3`, `v_coe = 0.5`  
3. `clip_coef = 0.1`, `ent_coef = 0.01`, `l_rate = 2.5e-4`, `v_coe = 0.5`  
4. `clip_coef = 0.1`, `ent_coef = 0.01`, `l_rate = 0.5e-4`, `v_coe = 1.0`  


### **Data & Outputs**
The following scripts and output files are provided for analyzing and visualizing training results:

- **[draw.py]**  
  Generates output graphs from event files produced by **`ppo_a2c_clip_only.py`** using `SummaryWriter`. The output graphs include:  
  - **PPO-clip-comparison.png**  
  - **PPO-clip-loss.png**  
  - **PPO-clip.png**  

- **[plot_pkl.py]**  
  Visualizes `.pkl` output files for further analysis.
