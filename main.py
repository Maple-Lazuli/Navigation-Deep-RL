from collections import deque
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from src.dqn_agent import Agent
plt.style.use('ggplot')


def dqn(agent, brain_name, save_name, n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    avg_scores = []
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    episodes_total = 0
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]
        score = 0
        t = 0
        while True:
            # for t in range(max_t):
            t += 1
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            avg_scores.append(np.mean(scores_window))
        if np.mean(scores_window) >= 15:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), f'saved_models/{save_name}.pth')
            episodes_total = i_episode
            break
    return scores, avg_scores, episodes_total

if __name__ == "__main__":
    # Create Environment
    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")
    # Set brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # Reset environment and get state, state_size, and action_size
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)
    state = env_info.vector_observations[0]
    state_size = len(state)
    print('States have length:', state_size)
    # Run the three models and save the results
    results = []
    agent = Agent(state_size=state_size, action_size=action_size, seed=0, units=32)
    results.append(dqn(agent, brain_name, 'model1'))

    agent = Agent(state_size=state_size, action_size=action_size, seed=0, units=64)
    results.append(dqn(agent, brain_name, 'model2'))

    agent = Agent(state_size=state_size, action_size=action_size, seed=0, units=128)
    results.append(dqn(agent, brain_name, 'model3'))
    # close the environment
    env.close()
    # save the scores as a png
    fig, ax = plt.subplots(figsize=(15, 10))
    colors = ['red', 'green', 'blue']
    for i in range(0, 3):
        ax.plot(range(0, len(results[i][0])), results[i][0], color=colors[i], alpha=.2)
        ax.plot(np.arange(1, len(results[i][1]) + 1) * 100, results[i][1], color=colors[i], label=f"Model{i + 2}",
                alpha=.8)

    ax.axhline(y=13, color='red', linestyle='-', label="Minimum Score")
    ax.set_title("Episodes vs Score")
    ax.set_ylabel('Score')
    ax.set_xlabel('Episode #')
    fig.legend()
    fig.savefig('images/scores.png', bbox_inches='tight')
    fig.show();
