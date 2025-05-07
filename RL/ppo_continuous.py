import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SubsetRandomSampler

from RL.attention import AttentionPolicy, AttentionCritic


class PPOAgent(object):
    def __init__(self, state_dim, action_dim, buffer, device, max_steps, gamma, gae_lambda, k_epoch, lr,
                 eps_clip, grad_clip, entropy_coef, batch_size, mini_batch_size, **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.k_epoch = k_epoch
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.kwargs = kwargs

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.lr = lr
        self.eps_clip = eps_clip
        self.grad_clip = grad_clip
        self.entropy_coef = entropy_coef

        self.device = device

        self.policy_net = None
        self.value_net = None
        self.policy_optim = None
        self.value_optim = None

        self.rolloutBuffer = buffer

    def select_action(self, state):
        # state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            dist = self.policy_net.get_distribution(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu().numpy(), log_prob.sum(1).cpu().numpy()

    def get_log_prob(self, state, action):
        # state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0).to(self.device)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            dist = self.policy_net.get_distribution(state)
            log_prob = dist.log_prob(action)
        return log_prob.sum(1).cpu().numpy()

    def train(self):
        raise NotImplementedError("please override this method")

    def lr_decay(self, step):
        factor = 1 - step / self.max_steps
        lr = factor * self.lr
        for p in self.policy_optim.param_groups:
            p["lr"] = lr
        for p in self.value_optim.param_groups:
            p["lr"] = lr
        return lr
    
    def eval(self):
        self.policy_net.eval()
        self.value_net.eval()

    def save(self, filename):
        torch.save(self.policy_net.state_dict(), "{}_policy.pt".format(filename))
        torch.save(self.value_net.state_dict(), "{}_value.pt".format(filename))
        torch.save(self.policy_optim.state_dict(), "{}_policy_optimizer.pt".format(filename))
        torch.save(self.value_optim.state_dict(), "{}_value_optimizer.pt".format(filename))

    def load(self, filename):
        self.policy_net.load_state_dict(torch.load("{}_policy.pt".format(filename)))
        self.value_net.load_state_dict(torch.load("{}_value.pt".format(filename)))
        self.policy_optim.load_state_dict(torch.load("{}_policy_optimizer.pt".format(filename)))
        self.value_optim.load_state_dict(torch.load("{}_value_optimizer.pt".format(filename)))


class AttendPPOAgent(PPOAgent):
    def __init__(self, num_mainline: int, num_ramp: int, feature_dim: int, action_dim: int, hidden_dim: int,
                 embedding_dim: int, feed_forward_dim: int, num_attend_layer: int, num_head: int, dropout: float,
                 buffer, device, max_steps, gamma, gae_lambda, k_epoch, lr, eps_clip, grad_clip, entropy_coef,
                 batch_size, mini_batch_size):
        state_dim = (num_mainline + num_ramp) * feature_dim
        super().__init__(state_dim, action_dim, buffer, device, max_steps, gamma, gae_lambda, k_epoch, lr,
                         eps_clip, grad_clip, entropy_coef, batch_size, mini_batch_size)

        self.policy_net = AttentionPolicy(num_mainline, num_ramp, feature_dim, action_dim, hidden_dim, embedding_dim,
                                          feed_forward_dim, num_attend_layer, num_head, dropout).to(self.device)
        self.value_net = AttentionCritic(num_mainline, num_ramp, feature_dim, hidden_dim, embedding_dim,
                                         feed_forward_dim, num_attend_layer, num_head, dropout).to(self.device)

        self.policy_optim = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr, eps=1e-5)
        self.value_optim = torch.optim.Adam(self.value_net.parameters(), lr=self.lr, eps=1e-5)

    def train(self):
        state, action, log_prob, reward, next_state, done = self.rolloutBuffer.pull()

        with torch.no_grad():
            # there are N = num_env independent environments, cannot flatten state here
            # let "values" match the dimension of "done"
            # state: (batch size, num_env, feature) -> (num_env, batch size, feature)
            state, next_state = state.transpose(0, 1), next_state.transpose(0, 1)
            values, next_values = [], []
            for i in range(state.size()[0]):
                values.append(self.value_net.get_value(state[i]))  # (batch size, 1)
                next_values.append(self.value_net.get_value(next_state[i]))  # (batch size, 1)
            values = torch.concatenate(tuple(values), axis=1)  # (batch size, num_env)
            next_values = torch.concatenate(tuple(next_values), axis=1)  # (batch size, num_env)
            advantage = torch.zeros_like(values).to(self.device)
            delta = reward + self.gamma * (1 - done) * next_values - values
            gae = 0.
            for t in reversed(range(self.rolloutBuffer.steps)):
                gae = delta[t] + self.gamma * self.gae_lambda * gae * (1 - done[t])
                advantage[t] = gae
            returns = advantage + values
            norm_adv = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        state = state.transpose(0, 1).contiguous()
        # -------- flatten vectorized environment --------
        state = state.view(-1, self.state_dim)
        action = action.view(-1, self.action_dim)
        log_prob = log_prob.view(-1, 1)
        returns = returns.view(-1, 1)
        norm_adv = norm_adv.view(-1, 1)
        for _ in range(self.k_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, True):
                new_dist = self.policy_net.get_distribution(state[index])
                new_log_prob = new_dist.log_prob(action[index]).sum(1, keepdim=True)
                new_values = self.value_net.get_value(state[index]).view(self.mini_batch_size, -1)
                entropy = new_dist.entropy().sum(1)
                ratios = torch.exp(new_log_prob - log_prob[index])

                surrogate1 = ratios * norm_adv[index]
                surrogate2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * norm_adv[index]
                entropy_loss = (self.entropy_coef * entropy).mean()
                actor_loss = -1 * torch.min(surrogate1, surrogate2).mean()
                policy_loss = actor_loss - entropy_loss
                # update the policy network
                self.policy_optim.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
                self.policy_optim.step()
                critic_loss = 0.5 * torch.nn.functional.mse_loss(new_values, returns[index])
                # update the value network
                self.value_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.grad_clip)
                self.value_optim.step()
        return actor_loss.item(), critic_loss.item(), entropy_loss.item()
