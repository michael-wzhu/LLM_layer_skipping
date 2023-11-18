

''' controller '''

import collections
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Controller(torch.nn.Module):
    def __init__(self, hidden_size,
                 num_hidden_layers,
                 dropout_ratio=0.1,):
        super(Controller, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 9, hidden_size * 2),
            nn.Dropout(p=dropout_ratio),
            nn.Tanh(),
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.Dropout(p=dropout_ratio),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_hidden_layers * 2 * 2),
            # torch.nn.Sigmoid()
        )

        self.temperature = 2

        # pooler
        self.adap_pooler_1 = nn.AdaptiveAvgPool1d(4)
        self.adap_pooler_2 = nn.AdaptiveMaxPool1d(4)

    def forward(self, input_tensor,):

        # 进行池化
        input_tensor_1 = input_tensor.transpose(1, 2)  # B x D x L
        hs_1 = (self.adap_pooler_1(input_tensor_1)).transpose(1, 2)  # B x num_prompt_tokens x D
        hs_1 = hs_1.view(1, -1)
        hs_2 = (self.adap_pooler_2(input_tensor_1)).transpose(1, 2)  # B x num_prompt_tokens x D
        hs_2 = hs_2.view(1, -1)

        hs_3 = input_tensor[0, -1, :]
        hs_3 = hs_3.view(1, -1)

        hs = torch.cat([hs_1, hs_2, hs_3], dim=-1)
        logits = self.net(hs)
        return logits

    def sample(self, input_tensor):
        logits = self(input_tensor)
        logits = logits.view(-1, 2)
        probs = F.softmax(logits / self.temperature, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        entropies = -(log_probs * probs).sum(-1, keepdim=False)

        actions = probs.multinomial(num_samples=1).data
        # print("logits: ", logits.shape)
        # print("probs: ", probs.shape)
        # print("entropies: ", entropies)

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


# if __name__ == "__main__":

