

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
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Dropout(p=dropout_ratio),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.Dropout(p=dropout_ratio),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_hidden_layers * 2 * 2),
            # torch.nn.Sigmoid()
        )

    def forward(self, input_tensor,):
        logits = self.net(input_tensor)
        return logits

    def sample(self, input_tensor):
        logits = self.forward(input_tensor)
        logits = logits.view(-1, 2)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        entropies = -(log_probs * probs).sum(-1, keepdim=False)

        actions = probs.multinomial(num_samples=1).data
        selected_log_probs = log_probs.gather(
            1,
            torch.Tensor(actions).cuda()
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

