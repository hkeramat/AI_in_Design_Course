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
