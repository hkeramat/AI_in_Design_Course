### I am preparing a repository for the course I am going to present
### 10-arm bandit problem with action value method ( comparing epsilon equal to 0.01 and 0.1)
```
import numpy as np
import matplotlib.pyplot as plt

# Define the number of arms
n_arms = 10

# Define the number of time steps
n_steps = 1000

# Define the epsilon values to test
epsilons = [0.01, 0.1]

# Define the window size for the moving average
window_size = 100

# Loop over the epsilon values
for epsilon in epsilons:

    # Initialize the action-value function Q to zero
    Q = np.zeros(n_arms)

    # Initialize the array to track the rewards over time
    rewards = np.zeros(n_steps)

    # Loop over the time steps
    for t in range(n_steps):

        # Select the action with the highest action-value with probability 1-epsilon
        if np.random.rand() > epsilon:
            action = np.argmax(Q)
        # Select a random action with probability epsilon
        else:
            action = np.random.randint(n_arms)

        # Get the reward for the selected action
        reward = np.random.normal(Q[action], 1)

        # Update the action-value function Q using the sample-average method
        Q[action] += 1/(t+1) * (reward - Q[action])

        # Track the reward over time
        rewards[t] = reward

    # Compute the moving average of the rewards
    ma_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode="valid")

    # Plot the moving average of the rewards over time
    plt.plot(np.arange(window_size-1, n_steps), ma_rewards, label="epsilon = {}".format(epsilon))

# Add labels and legend to the plot
plt.xlabel("Time step")
plt.ylabel("Average reward (moving average)")
plt.title("Average reward vs. time step")
plt.legend()

# Show the plot
plt.show()
```

### The 10-arm bandit problem with and without optimistic initial values using the epsilon-greedy algorithm
```
import numpy as np
import matplotlib.pyplot as plt

# Define the Bandit class
class Bandit:
    def __init__(self, k, q_mean=0, q_std=1):
        self.k = k
        self.q_true = np.random.normal(q_mean, q_std, k)
        self.actions = np.arange(k)
        
    def get_reward(self, action):
        return np.random.normal(self.q_true[action], 1)
    

# Define the epsilon-greedy algorithm
def epsilon_greedy(bandit, num_steps, epsilon, alpha):
    q_estimates = np.zeros(bandit.k)
    action_counts = np.zeros(bandit.k)
    rewards = np.zeros(num_steps)
    
    for step in range(num_steps):
        if np.random.rand() < epsilon:
            action = np.random.choice(bandit.actions)
        else:
            action = np.argmax(q_estimates)
        
        reward = bandit.get_reward(action)
        action_counts[action] += 1
        q_estimates[action] += alpha * (reward - q_estimates[action])
        rewards[step] = reward
        
    return action_counts, rewards


# Define parameters
num_bandits = 10
num_steps = 1000
epsilon = 0.1
alpha = 0.1
optimistic_initial_value = 5.0
initial_value = 0.0

# Define two bandits with different initial values
bandit_oiv = Bandit(num_bandits, q_mean=optimistic_initial_value)
bandit_iv = Bandit(num_bandits, q_mean=initial_value)

# Run epsilon-greedy algorithm with optimistic initial value
_, reward_oiv = epsilon_greedy(bandit_oiv, num_steps, epsilon, alpha)

# Run epsilon-greedy algorithm with 0 initial value
_, reward_iv = epsilon_greedy(bandit_iv, num_steps, epsilon, alpha)

# Compute moving average of rewards
window_size = 10
reward_oiv_ma = np.convolve(reward_oiv, np.ones(window_size)/window_size, mode='valid')
reward_iv_ma = np.convolve(reward_iv, np.ones(window_size)/window_size, mode='valid')

# Plot results
plt.plot(reward_oiv_ma, label='Optimistic Initial Value')
plt.plot(reward_iv_ma, label='0 Initial Value')
plt.xlabel('Time Step')
plt.ylabel('Average Reward')
plt.title('Effect of Optimistic Initial Value')
plt.legend()
plt.show()
```
