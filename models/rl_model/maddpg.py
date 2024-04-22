"""
Implementation of TS-MADDPG from the paper titled "Improved pairs trading strategy using two-level reinforcement
    learning framework" by Zhizhao Xu and Chao Luo (https://cs.nyu.edu/~shasha/papers/pairstrading_reinforcementlearning.pdf)
Heavily adapted from Phil Tabor's MADDPG implementation (https://github.com/philtabor/Multi-Agent-Deep-Deterministic-Policy-Gradients)
"""

from .actor import Actor
from .critic import Critic
import torch as T
import torch.nn.functional as F

class MADDPG:
    """Multi Agent Deep Deterministic Policy Gradient"""
    """3 coorperating actors and 1 critic."""
    def __init__(self, actor_dims, critic_dims, n_actors, action_dims,
                 alpha=0.0001, beta=0.0001, chkpt_dir='tmp/maddpg/'):
        self.actors = []
        self.n_actors = n_actors
        self.action_dims = action_dims
        for agent_idx in range(self.n_actors):
            self.actors.append(Actor(actor_dims[agent_idx], action_dims, agent_idx, alpha=alpha, chkpt_dir=chkpt_dir))

        self.critic = Critic(critic_dims, action_dims, agent_idx, beta=beta, chkpt_dir=chkpt_dir)
        self.loss = T.nn.MSELoss()


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.actors:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.actors:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.actors):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions

    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.actors[0].actor.device

        states = T.tensor(states, dtype=T.float, requires_grad=False).to(device)
        rewards = T.tensor(rewards, dtype=T.float, requires_grad=False).to(device)
        states_ = T.tensor(states_, dtype=T.float, requires_grad=False).to(device)
        dones = T.tensor(dones, requires_grad=False).to(device)

        all_actors_new_actions = []
        all_actors_new_mu_actions = []
        old_actors_actions = []

        for agent_idx, agent in enumerate(self.actors):
            new_states = T.tensor(actor_new_states[agent_idx],
                                    dtype=T.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            all_actors_new_actions.append(new_pi)
            mu_states = T.tensor(actor_states[agent_idx],
                                    dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_actors_new_mu_actions.append(pi)
            old_action = T.tensor(actions[agent_idx],
                                    dtype=T.float).to(device)
            old_actors_actions.append(old_action)

        new_actions = T.cat([acts for acts in all_actors_new_actions], dim=1)
        mu = T.cat([acts for acts in all_actors_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_actors_actions],dim=1)

        critic_value_ = self.critic.target_critic.forward(states_.detach(), new_actions.detach()).flatten()
        critic_value_[dones[:,0]] = 0.0
        critic_value = self.critic.critic.forward(states.detach(), old_actions.detach()).flatten()

        target = rewards[:,agent_idx] + agent.gamma*critic_value_
        critic_loss = self.loss(target, critic_value)
        self.critic.critic.optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic.critic.optimizer.step()
        actor_loss = self.critic.critic.forward(states.detach(), mu.detach()).flatten()
        actor_loss = -actor_loss.mean()

        for agent_idx, agent in enumerate(self.actors):
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            agent.update_network_parameters()

        self.critic.update_network_parameters()