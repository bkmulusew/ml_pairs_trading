"""
Implementation of TS-MADDPG from the paper titled "Improved pairs trading strategy using two-level reinforcement
    learning framework" by Zhizhao Xu and Chao Luo (https://cs.nyu.edu/~shasha/papers/pairstrading_reinforcementlearning.pdf)
Heavily adapted from Phil Tabor's MADDPG implementation (https://github.com/philtabor/Multi-Agent-Deep-Deterministic-Policy-Gradients)
"""

from .networks import CriticNetwork

class Critic:
    def __init__(self, critic_dims, action_dims, agent_idx, chkpt_dir,
                    beta=0.0001, fc1=64, fc2=64, gamma=0.995, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.action_dims = action_dims
        self.agent_name = 'agent_%s' % agent_idx
        self.critic = CriticNetwork(beta, critic_dims,
                            fc1, fc2, action_dims,
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')
        self.target_critic = CriticNetwork(beta, critic_dims,
                                            fc1, fc2, action_dims,
                                            chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic')

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()