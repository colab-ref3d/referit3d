from .base import BaseDistMgr, DistModuleImpl


class LocalDistModuleWrapper(DistModuleImpl):
    pass


class LocalDistMgr(BaseDistMgr):
    def allreduce_sum(self, tensor):
        pass

    def init_dist(self):
        return self.get_rank(), self.get_world_size()

    def get_rank(self):
        return 0

    def get_world_size(self):
        return 1

    def get_dist_module(self, module):
        return LocalDistModuleWrapper(module)
