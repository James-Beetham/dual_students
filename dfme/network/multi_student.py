import typing

import torch

def _pre_mean(m:list[torch.nn.Module])->torch.Tensor:
    return m
def _mean(o:list[torch.Tensor])->torch.Tensor:
    return torch.stack(o).mean(dim=0)
def _pre_first(m:list[torch.nn.Module])->torch.Tensor:
    return [m[0]]
def _first(o:list[torch.Tensor])->torch.Tensor:
    return o[0]
COMBINE_MODES:dict[str,tuple[
        typing.Callable[[list[torch.nn.Module]],list[torch.nn.Module]],
        typing.Callable[[list[torch.Tensor]],torch.Tensor]]] = dict(
    first=(_pre_first,_first),
    mean=(_pre_mean,_mean),
)

class MultiStudentModel(torch.nn.Module):
    """Process the target model outputs and remove the gradient.
    """
    def __init__(self,args,models:list[torch.nn.Module]):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.pre_combine,self.combine = COMBINE_MODES[args.combine_student_outputs]

    def forward(self,x:torch.Tensor,combine=True):
        models = self.models
        if combine: models = self.pre_combine(self.models)
        o = []
        for m in models: o.append(m(x))
        if combine: o = self.combine(o)
        return o
