import torch


class BaseDistMgr:
    dist_model_cls = None
    sync_bn_cls = None

    def __init__(self):
        print(f'using {self.__class__.__name__}')

    def init_dist(self):
        raise NotImplementedError

    def get_rank(self):
        raise NotImplementedError

    def get_world_size(self):
        raise NotImplementedError

    def convert_sync_batchnorm(self, module):
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            print(f'replacing {module} to {self.sync_bn_cls}')
            module_output = self.sync_bn_cls(module.num_features,
                                             module.eps, module.momentum,
                                             module.affine,
                                             module.track_running_stats)
            if module.affine:
                module_output.weight.data = module.weight.data.clone(memory_format=torch.preserve_format).detach()
                module_output.bias.data = module.bias.data.clone(memory_format=torch.preserve_format).detach()
                # keep requires_grad unchanged
                module_output.weight.requires_grad = module.weight.requires_grad
                module_output.bias.requires_grad = module.bias.requires_grad
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
        for name, child in module.named_children():
            module_output.add_module(name, self.convert_sync_batchnorm(child))
        del module
        return module_output

    def get_dist_module(self, module):
        assert self.dist_model_cls is not None \
               and self.sync_bn_cls is not None, NotImplementedError(
            f'should specify {self.__class__.__name__}.dist_model_cls and {self.__class__.__name__}.sync_bn_cls')
        module = self.convert_sync_batchnorm(module)
        dist = self.dist_model_cls(module)
        return dist

    def allreduce_mean(self, tensor):
        """
        inplace allreduce-mean
        """
        raise NotImplementedError


class DistModuleImpl(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def sync_gradients(self):
        pass

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
