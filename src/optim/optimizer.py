import pdb
import torch
import numpy as np
import types
import math
from torch import inf
from functools import wraps
import warnings
import weakref
from bisect import bisect_right

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from itertools import chain

def opt_params(parm, lr):
    return {'params': chain_params(parm), 'lr':lr}

def chain_params(p):
    return list(chain(*[trainable_params_(p)]))
 

def trainable_params_(m):
    return [p for p in m.parameters() if p.requires_grad]

class LayerOptimizer(object):
	def __init__(self, layer_groups, lrs, wds):
		self.layer_groups, self.lrs, self.wds = layer_groups, lrs, wds
		self.opt = torch.optim.Adam(self.opt_params())

	def opt_params(self):
		self.layers_groups = [layer for layer in self.layer_groups if list(layer.parameters())]
		n_layers = len(self.layer_groups)
		if not isinstance(self.lrs, list):self.lrs = [self.lrs]*n_layers
		params = list(zip(self.layer_groups, self.lrs))
		return [opt_params(*p) for p in params]

class _Optimizer(object):
	def __init__(self, model, lrs, wds=None):
		self.lrs = lrs
		self.wds = wds
		self.model = model
		
	def child(self, x):
		return list(x.children())
	
	def recursive_(self, child):
		if hasattr(child, 'children'):
			if len(self.child(child)) != 0:
				child = self.child(child)
				return self.recursive_(child)
		return child

	def get_layer_groups(self):
		children = []
		for child in self.child(self.model):
			children.extend(self.recursive_(child))
		return children

	def get_layer_opt(self):
		return LayerOptimizer(self.get_layer_groups(), self.lrs, self.wds)

class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_epoch = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step(last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule."
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class STLR(_LRScheduler):
	def __init__(self, optimizer, T_max, last_epoch=-1, ratio=32):
		self.T_max = T_max
		self.cut =  np.floor(T_max*0.1)
		self.ratio = ratio
		super(STLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		if self.last_epoch < self.cut:
			p = self.last_epoch/self.cut
		else:
			fraction = (self.last_epoch - self.cut)/(self.cut*(1/self.ratio - 1))
			p = 1 - fraction
		return [base_lr*(1 + p*(self.ratio - 1))/self.ratio for base_lr in self.base_lrs]
