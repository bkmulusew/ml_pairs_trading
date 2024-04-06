"""
Implementation of TS-MADDPG from the paper titled "Improved pairs trading strategy using two-level reinforcement
    learning framework" by Zhizhao Xu and Chao Luo (https://cs.nyu.edu/~shasha/papers/pairstrading_reinforcementlearning.pdf)
Heavily adapted from Phil Tabor's MADDPG implementation (https://github.com/philtabor/Multi-Agent-Deep-Deterministic-Policy-Gradients)
"""

from .networks import ActorNetwork
import torch as T

class Actor:
    def __init__(self, actor_dims, action_dims, agent_idx, chkpt_dir,
                    alpha=0.0001, fc1=64, fc2=64, gamma=0.995, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.action_dims = action_dims
        self.agent_name = 'agent_%s' % agent_idx
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, action_dims[agent_idx],
                                    chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, action_dims[agent_idx],
                                        chkpt_dir=chkpt_dir,
                                        name=self.agent_name+'_target_actor')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions = self.actor.forward(state)   # Should exploration be enabled?

        return actions.detach().cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()