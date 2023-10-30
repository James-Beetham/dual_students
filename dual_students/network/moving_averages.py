import torch
import torch.nn as nn
import torch.nn.functional as F

class MovingAverages(nn.Module):
    def __init__(self, model_class, args_kwargs=([],{}), momentum=0):
        super(MovingAverages, self).__init__()
        self.model_train:nn.Module = model_class(*args_kwargs[0], **args_kwargs[1])
        if momentum > 0:
            self.model_test:nn.Module = model_class(*args_kwargs[0], **args_kwargs[1])
            self.model_test.eval()
        self.momentum = momentum
        self.first = True
         
    def forward(self, x, test=False, step=False, **kwargs):
        if step:
            return self.moving_averages_step()
        model = self.model_train if not test or self.momentum == 0 else self.model_test
        return model(x, **kwargs)

    def moving_averages_step(self):
        if self.momentum == 0:
            return
        with torch.no_grad():
            if self.first:
                m = 0
                self.first = False
            else:
                m = self.momentum
            for tr,te in zip(self.model_train.parameters(), self.model_test.parameters()):
                new_val = m * te + (1-m) * tr
                te.copy_(new_val.detach())
            for tr,te in zip(self.model_train.buffers(), self.model_test.buffers()):
                new_val = m * te + (1-m) * tr
                te.copy_(new_val.detach())
