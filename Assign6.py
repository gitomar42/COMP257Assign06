import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.trajectories import trajectory
from tf_agents.environments.wrappers import PyEnvironmentBaseWrapper
import numpy as np
import matplotlib.pyplot as plt
from tf_agents.environments import tf_py_environment
class FlattenObservationWrapper(PyEnvironmentBaseWrapper):
    def observation(self, observation):
        return np.asarray(observation).flatten()


# Set up the LunarLander environment with observation flattening
env_name = 'LunarLander-v2'
raw_env = suite_gym.load(env_name)
flat_env = FlattenObservationWrapper(raw_env)
train_env = tf_py_environment.TFPyEnvironment(flat_env)


# Define the Q-network
fc_layer_params = (100, )
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params
)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)

train_step_counter = tf.Variable(0)

# Define the DQN agent
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter
)

agent.initialize()

# Hyperparameters
num_iterations = 10000  # You can adjust this value based on your strategy
initial_num_episodes = 100
max_steps_per_episode = 1000
initial_discount_factor = 0.99

# Training loop
returns = []

for iteration in range(num_iterations):
    # Collect data from episodes
    try:
        time_step = train_env.reset()
        print("Observation Space Structure:", time_step.observation)
    except Exception as e:
        print("Error accessing observation space during reset:", e)
        break

    episode_return = 0

    for _ in range(max_steps_per_episode):
        try:
            action_step = agent.policy.action(time_step)
            next_time_step = train_env.step(action_step.action)
        except Exception as e:
            print("Error during environment step:", e)
            break

        episode_return += time_step.reward.numpy()

        time_step = next_time_step

    # Train the agent
    train_loss = agent.train()

    # Evaluate the agent periodically
    if iteration % 500 == 0:
        avg_return = common.compute_avg_return(train_env, agent.policy, num_episodes=5)
        print(f"Iteration {iteration}, Average Return: {avg_return}")

    returns.append(episode_return)
    
       


# Plotting cumulative rewards over time
cumulative_returns = [sum(returns[:i + 1]) for i in range(len(returns))]

plt.plot(cumulative_returns)
plt.xlabel('Episodes')
plt.ylabel('Cumulative Returns')
plt.title('Agent Learning Progress')
plt.show()
