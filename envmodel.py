from modules import *
from utils import *
from hssm import HierarchicalStateSpaceModel


class EnvModel(nn.Module):
    def __init__(self,
                 belief_size,
                 state_size,
                 num_layers,
                 max_seg_len,
                 max_seg_num):
        super(EnvModel, self).__init__()
        ################
        # network size #
        ################
        self.belief_size = belief_size
        self.state_size = state_size
        self.num_layers = num_layers
        self.max_seg_len = max_seg_len
        self.max_seg_num = max_seg_num

        ###############
        # init models #
        ###############
        # state space model
        self.state_model = HierarchicalStateSpaceModel(belief_size=self.belief_size,
                                                       state_size=self.state_size,
                                                       num_layers=self.num_layers,
                                                       max_seg_len=self.max_seg_len,
                                                       max_seg_num=self.max_seg_num)

    def forward(self, obs_data_list, seq_size, init_size, obs_std=1.0):
        ############################
        # (1) run over state model #
        ############################
        [obs_rec_list,
         prior_boundary_log_density_list,
         post_boundary_log_density_list,
         prior_abs_state_list,
         post_abs_state_list,
         prior_obs_state_list,
         post_obs_state_list,
         boundary_data_list,
         prior_boundary_list,
         post_boundary_list] = self.state_model(obs_data_list, seq_size, init_size)

        ########################################################
        # (2) compute obs_cost (sum over spatial and channels) #
        ########################################################
        obs_target_list = obs_data_list[:, init_size:-init_size]
        obs_cost = - Normal(obs_rec_list, obs_std).log_prob(obs_target_list)
        obs_cost = obs_cost.sum(dim=[2, 3, 4])

        #######################
        # (3) compute kl_cost #
        #######################
        # compute kl related to states
        kl_abs_state_list = []
        kl_obs_state_list = []
        for t in range(seq_size):
            # read flag
            read_data = boundary_data_list[:, t].detach()

            # kl divergences (sum over dimension)
            kl_abs_state = kl_divergence(post_abs_state_list[t], prior_abs_state_list[t]) * read_data
            kl_obs_state = kl_divergence(post_obs_state_list[t], prior_obs_state_list[t])
            kl_abs_state_list.append(kl_abs_state.sum(-1))
            kl_obs_state_list.append(kl_obs_state.sum(-1))
        kl_abs_state_list = torch.stack(kl_abs_state_list, dim=1)
        kl_obs_state_list = torch.stack(kl_obs_state_list, dim=1)

        # compute kl related to boundary
        kl_mask_list = (post_boundary_log_density_list - prior_boundary_log_density_list)

        # return
        return {'rec_data': obs_rec_list,
                'mask_data': boundary_data_list,
                'obs_cost': obs_cost,
                'kl_abs_state': kl_abs_state_list,
                'kl_obs_state': kl_obs_state_list,
                'kl_mask': kl_mask_list,
                'p_mask': prior_boundary_list.mean,
                'q_mask': post_boundary_list.mean,
                'p_ent': prior_boundary_list.entropy(),
                'q_ent': post_boundary_list.entropy(),
                'beta': self.state_model.mask_beta,
                'train_loss': obs_cost.mean() + kl_abs_state_list.mean() + kl_obs_state_list.mean() + kl_mask_list.mean()}

    def jumpy_generation(self, init_obs_list, seq_size):
        return self.state_model.jumpy_generation(init_obs_list, seq_size)

    def full_generation(self, init_obs_list, seq_size):
        return self.state_model.full_generation(init_obs_list, seq_size)