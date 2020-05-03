import torch
import torch.nn as nn
import copy


class Pruner(object):
    
    def __init__(self, model):
        self.model = model
        
        self.children = list(self.model.named_children())

    def prune(self, layer_index, index):
        assert layer_index < len(self.children)
        
        model = copy.deepcopy(self.model)

        module = getattr(model, self.children[layer_index][0])
        to_use = []
        for i in range(module.weight.shape[1]): # 1 is the input channel
            if i not in index:
                to_use.append(i)
                
        new_weight = module.weight[:, to_use, ...]

        module.weight = nn.Parameter(new_weight)

        return model

    def restore(self, model, layer_index, index):
        module = getattr(model, self.children[layer_index][0])
        origin_module = getattr(self.model, self.children[layer_index][0])
        shape = list(module.weight.size())
        shape[1] += len(index)
        origin_index = list(range(shape[1]))

        for i in index:
            origin_index.remove(i)
        
        new_weight = torch.zeros(tuple(shape))
        new_weight[:, origin_index, ...] = module.weight
        new_weight[:, index, ...] = origin_module.weight[:, index, ...]
        
        module.weight = nn.Parameter(new_weight)
        return model
         
    
if __name__ == "__main__":
    a = nn.Sequential(
        nn.Linear(3, 10)
    )
    
    b = Pruner(a)
    c = b.prune(0, [1])

    
    c(torch.zeros(3, 2))
    
    d = b.restore(c, 0, [1])
    d(torch.zeros(3, 3))