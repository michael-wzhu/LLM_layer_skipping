

''' controller '''

import collections
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Controller(torch.nn.Module):
    def __init__(self, hidden_size,
                 num_hidden_layers,
                 dropout_ratio=0.4,):
        super(Controller, self).__init__()

        self.num_hidden_layers = num_hidden_layers

        self.linear_1 = nn.Sequential(
            nn.Linear(num_hidden_layers * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        self.net_2 = nn.Sequential(
            nn.Linear(hidden_size * 6, hidden_size // 2),
            nn.Dropout(p=dropout_ratio),
            nn.Mish(),
            nn.Linear(hidden_size // 2, hidden_size // 8),
            nn.Dropout(p=dropout_ratio),
            nn.GELU(),
            nn.Linear(hidden_size // 8, num_hidden_layers * 2 * 2),
            # torch.nn.Sigmoid()
        )
        # self.net_2 = nn.Sequential(
        #     nn.Linear(hidden_size * 4, hidden_size // 8),
        #     nn.Dropout(p=dropout_ratio),
        #     nn.GELU(),
        #     nn.Linear(hidden_size // 8, num_hidden_layers * 2 * 2),
        #     # torch.nn.Sigmoid()
        # )

        self.temperature = 3.0

        # pooler
        self.adap_pooler_1 = nn.AdaptiveAvgPool1d(2)
        self.adap_pooler_2 = nn.AdaptiveMaxPool1d(2)

    def forward(self, input_tensor, all_hidden_states,):

        # TODO: 先去掉池化操作
        list_features_prev = []
        for hd_ in all_hidden_states[: -1]:
            hs_ = hd_[0, -1, :]
            hs_ = hs_.view(1, -1)
            list_features_prev.append(hs_)

        hs_prev = torch.cat(list_features_prev, dim=-1)
        hs_prev = self.linear_1(hs_prev)

        input_tensor_1 = input_tensor.transpose(1, 2)  # B x D x L
        hs_1 = (self.adap_pooler_1(input_tensor_1)).transpose(1, 2)  # B x num_prompt_tokens x D
        hs_1 = hs_1.view(1, -1)
        hs_2 = (self.adap_pooler_2(input_tensor_1)).transpose(1, 2)  # B x num_prompt_tokens x D
        hs_2 = hs_2.view(1, -1)

        hs_3 = input_tensor[0, -1, :]
        hs_3 = hs_3.view(1, -1)

        hs = torch.cat([hs_1, hs_2, hs_3, hs_prev], dim=-1)
        logits = self.net_2(hs)
        return logits

    def predict(self, input_tensor, all_hidden_states, topk=12):
        logits = self(input_tensor, all_hidden_states) / self.temperature
        logits = logits.view(-1, 2)
        probs = F.softmax(logits, dim=-1)
        probs_1 = probs[:, 1]

        probs_1_attn = probs_1[0::2]
        probs_1_ffn = probs_1[1::2]

        topk_probs_attn, topk_inds_attn = torch.topk(probs_1_attn, topk)
        attn_selected_ = topk_inds_attn.detach().cpu().numpy().tolist()
        actions_attn = [0] * self.num_hidden_layers
        for idx in attn_selected_:
            actions_attn[idx] = 1

        topk_probs_ffn, topk_inds_ffn = torch.topk(probs_1_ffn, topk)
        ffn_selected_ = topk_inds_ffn.detach().cpu().numpy().tolist()
        actions_ffn = [0] * self.num_hidden_layers
        for idx in ffn_selected_:
            actions_ffn[idx] = 1

        return actions_attn, topk_probs_attn, actions_ffn, topk_probs_ffn

    def sample(self, input_tensor, all_hidden_states, count=0):
        logits = self(input_tensor, all_hidden_states)
        logits = logits.view(-1, 2)

        if random.uniform(0, 1) < 0.01:
            print("logits: ", logits)

        # increase exploration
        # logits = F.tanh(logits)
        # if random.uniform(0, 1) < 0.02:
        #     logits = logits + 0.001 * torch.randn(logits.shape).cuda()

        probs = F.softmax(logits / self.temperature, dim=-1)
        log_probs = F.log_softmax(logits / self.temperature, dim=-1)

        entropies = -(log_probs * probs).sum(-1, keepdim=False)

        actions = probs.multinomial(num_samples=1).data
        # print("logits: ", logits.shape)
        # print("probs: ", probs.shape)
        # print("entropies: ", entropies)

        if count < 5000:
            if random.uniform(0, 1) < 0.3:
                actions = [[1], [0], [0], [0], [0], [1], [0], [0]] * int(self.num_hidden_layers / 4)
                actions = torch.tensor(actions)
            elif 0.3 <= random.uniform(0, 1) < 0.6:
                actions = [[1], [0], [0], [1], [1], [0], [0], [1]] * int(self.num_hidden_layers / 4)
                actions = torch.tensor(actions)

        selected_log_probs = log_probs.gather(
            1,
            actions.cuda()
        )

        # m = torch.distributions.bernoulli.Bernoulli(sample_prob)
        # selection = m.sample()
        #
        # # log πθ(at|st)
        # select_loss = - (
        #         selection * torch.log(sample_prob + 1e-10)
        #         + (1 - selection) * torch.log(
        #             1 - sample_prob + 1e-10
        #         )
        # )

        actions = actions.cpu().numpy().tolist()
        actions = [w[0] for w in actions]
        # print("actions: ", actions)

        return actions, selected_log_probs, entropies

        # return selection, select_loss,

    def save_pretrained(
            self,
            save_directory,
            is_main_process: bool = True,
            **kwargs,
    ):
        if is_main_process:
            torch.save(self.state_dict(), os.path.join(save_directory, 'controller_model.bin'))

    def load_pretrained(
            self,
            save_directory,
            is_main_process: bool = True,
            **kwargs,
    ):
        if is_main_process:

            return self.load_state_dict(
                torch.load(os.path.join(save_directory, 'controller_model.bin'))
            )

# if __name__ == "__main__":

