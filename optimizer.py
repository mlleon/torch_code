from collections import defaultdict, abc as container_abcs

import torch
from copy import deepcopy
from itertools import chain
import warnings
import functools


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class Optimizer(object):
    """
        Optimizer类有两个重要的成员变量，self.param_groups和self.state。
            self.param_groups用于存储模型参数和优化器本身的一些超参数（如学习率等）。
            self.state用于存储更新过程中模型参数对应的各种临时状态，如MSGD中每个参数需要对应一个动量。
                而每个参数可能不止需要对应一个临时状态。因此self.state是一个键值对类型为parameter: dict的有序字典。

        add_param_group方法，用来初始化self.param_groups。但self.state的初始化需要在某个具体的优化器子类中进行。

    """

    r"""Base class for all optimizers.
    
    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.
        注意，各参数的顺序必须保证每次运行都一致。比如dict和set数据结构不满足这个条件，所以要将参数params转换成列表param_groups

    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies torch.Tensor)what Tensors should be optimized.
            params的数据类型有两种:
                可迭代的torch.tensor
                传入单个字典(dict), {"params":Iterable value}
                传入多个字典(dict), 需要将多个dict放入一个列表容器中.
                    [{"params":Iterable value1}, {"params":Iterable value2}]
            Example_1:
                >>> optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            Example_2:
                >>> optim.SGD([
                        {'params': model.base.parameters()},
                        {'params': model.classifier.parameters(), 'lr': 0.001}],
                        lr = 0.01, momentum = 0.9)

        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    def __init__(self, params, defaults):
        torch._C._log_api_usage_once("python.optimizer")

        # 字典类型，defaults是所有超参数的初始值（子类传入）
        self.defaults = defaults
        self._hook_for_profile()

        if isinstance(params, torch.Tensor):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            torch.typename(params))

        # 将state属性设置为"defaultdict(<class 'list'>, {'':{}, '':{}})"
        self.state = defaultdict(dict)

        self.param_groups = []  # 将参数params转换成列表param_groups
        """
            1、params对象（传入单个dict）是一个生成器，使用list()方法将params对象转化为列表（param_groups）
            Example：
                >>> params = model.parameters()
            Result(params):
                <generator object Module.parameters at 0x7f05f549d660>  
            list[params]转换后 >> param_groups：
                [Parameter containing:
                 tensor([[ 0.2365, -0.1118, -0.3801,  0.0275,  0.4168],
                         [-0.1995, -0.1456,  0.3497, -0.0622, -0.1708],
                         [-0.0901,  0.0164, -0.3643, -0.1278,  0.4336],
                         [-0.0959,  0.4073, -0.1746, -0.1799, -0.1333]], requires_grad=True)]
                         
            2、params对象（传入多个dict）是一个生成器列表，使用list()方法后数据类型不变，仍是一个生成器列表（param_groups）
            Example：
                >>> params = [{"params":model._modules['linear1'].parameters(), "lr":0.01},
                            {"params":model._modules['linear3'].parameters(), "lr":0.05}]
            Result(params):
                [{'params': <generator object Module.parameters at 0x7f05f54813c0>,'lr': 0.01},
                    {'params': <generator object Module.parameters at 0x7f05f5453660>,'lr': 0.05}]
            list[params]转换后 >> param_groups：
                [{'params': <generator object Module.parameters at 0x7f05f54813c0>,'lr': 0.01},
                    {'params': <generator object Module.parameters at 0x7f05f5453660>,'lr': 0.05}]
            list(params[0]['params'])：
                [Parameter containing:
                 tensor([[ 0.2365, -0.1118, -0.3801,  0.0275,  0.4168],
                         [-0.1995, -0.1456,  0.3497, -0.0622, -0.1708],
                         [-0.0901,  0.0164, -0.3643, -0.1278,  0.4336],
                         [-0.0959,  0.4073, -0.1746, -0.1799, -0.1333]], requires_grad=True)]
        """
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            # 将参数列表param_groups转换成由列表封装的字典
            param_groups = [{'params': param_groups}]
        for param_group in param_groups:
            """
                1、params对象（传入单个dict）是一个生成器, param_group = params = param_groups,每个param_group仅包含“params”
                Example_1:  param_group = params
                    >>> optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
                2、params对象（传入多个dict）是一个生成器列表, param_group = params[i] = param_groups，
                    每个param_group可以包含{'params':, 'lr', 'momentum':, 'dampening':, 'weight_decay':, 'nesterov':}等一个或者多个键值对。
                Example_2: 
                    >>> optim.SGD([
                            {'params': model.base.parameters()},
                            {'params': model.classifier.parameters(), 'lr': 0.001}],
                            lr = 0.01, momentum = 0.9)
            """
            self.add_param_group(param_group)

        # Allows _cuda_graph_capture_health_check to rig a poor man's TORCH_WARN_ONCE in python,
        # which I don't think exists
        # https://github.com/pytorch/pytorch/issues/72948
        self._warned_capturable_if_run_uncaptured = True

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._hook_for_profile()  # To support multiprocessing pickle/unpickle.

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += 'Parameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key])
        format_string += ')'
        return format_string

    # Currently needed by Adam and AdamW
    def _cuda_graph_capture_health_check(self):
        if torch.has_cuda and torch.cuda.is_available():
            capturing = torch.cuda.is_current_stream_capturing()

            if capturing and not self.defaults['capturable']:
                raise RuntimeError("Attempting CUDA graph capture of step() for an instance of " +
                                   self.__class__.__name__ +
                                   " but this instance was constructed with capturable=False.")

            if (
                    (not getattr(self, "_warned_capturable_if_run_uncaptured", False))
                    and self.defaults["capturable"]
                    and (not capturing)
            ):
                print("Warning: This instance was constructed with capturable=True, but step() " +
                      "is running without CUDA graph capture. If you never intend to graph-capture this " +
                      "instance, capturable=True can impair performance, and you should set capturable=False.")
                self._warned_capturable_if_run_uncaptured = True

    def _hook_for_profile(self):
        self._zero_grad_profile_name = "Optimizer.zero_grad#{}.zero_grad".format(self.__class__.__name__)

        def profile_hook_step(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                obj, *_ = args
                profile_name = "Optimizer.step#{}.step".format(obj.__class__.__name__)
                with torch.autograd.profiler.record_function(profile_name):
                    return func(*args, **kwargs)

            return wrapper

        hooked = getattr(self.__class__.step, "hooked", None)
        if not hooked:
            self.__class__.step = profile_hook_step(self.__class__.step)
            self.__class__.step.hooked = True

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a list containing all parameter groups where each
            parameter group is a dict
        """
        # Save order indices instead of Tensors
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            # 对"params"和其它的键采用不同规则
            packed = {k: v for k, v in group.items() if k != 'params'}
            param_mappings.update({id(p): i for i, p in enumerate(group['params'], start_index)
                                   if id(p) not in param_mappings})
            # 这里并没有保存参数的值，而是保存参数的id
            packed['params'] = [param_mappings[id(p)] for p in group['params']]
            start_index += len(packed['params'])
            return packed

        # 对self.param_groups进行遍历
        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state = {(param_mappings[id(k)] if isinstance(k, torch.Tensor) else k): v
                        for k, v in self.state.items()}
        # 返回状态和参数组，其中参数组才是优化器的参数
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        # 应该是防止函数中对输入的state_dict进行改动
        # 因为字典是可变数据类型
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        # 参数的长度检测，保证输入的state_dict和优化器的参数数目一致
        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        # 用输入的state_dict更新当前state_dict的状态
        id_map = {old_id: p for old_id, p in
                  zip(chain.from_iterable((g['params'] for g in saved_groups)),
                      chain.from_iterable((g['params'] for g in groups)))}

        def cast(param, value, key=None):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                # Make sure state['step'] is not casted https://github.com/pytorch/pytorch/issues/74424
                if (key != "step"):
                    if param.is_floating_point():
                        value = value.to(param.dtype)
                    value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v, key=k) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group

        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    def zero_grad(self, set_to_none: bool = False):
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        foreach = self.defaults.get('foreach', False)

        if not hasattr(self, "_zero_grad_profile_name"):
            self._hook_for_profile()
        if foreach:
            per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
        with torch.autograd.profiler.record_function(self._zero_grad_profile_name):
            """
                遍历过程就是获取optimizer的param_groups属性的字典，其中的["params"]，通过遍历设定每个参数的梯度值为0。
            """
            # 获取每一组参数
            for group in self.param_groups:
                # 遍历当前参数组所有的params
                for p in group['params']:
                    if p.grad is not None:
                        if set_to_none:
                            p.grad = None
                        else:
                            if p.grad.grad_fn is not None:
                                p.grad.detach_()
                            else:
                                p.grad.requires_grad_(False)
                            if (not foreach or p.grad.is_sparse):
                                p.grad.zero_()
                            else:
                                per_device_and_dtype_grads[p.grad.device][p.grad.dtype].append(p.grad)
            if foreach:
                for _, per_dtype_grads in per_device_and_dtype_grads.items():
                    for grads in per_dtype_grads.values():
                        torch._foreach_zero_(grads)

    def step(self, closure):
        """
            step()方法负责更新参数值，但是其具体实现对于不同的优化算法是不同的，
                所以optimizer基类只是定义了这种行为，但是并没有给出具体实现。
            有些优化算法会多次重新计算函数（比如Conjugate Gradient、LBFGS），
                这样的话需要使用一个闭包（closure）来支持多次计算model的操作。
                这个闭包（closure）的运行过程是：清除梯度，计算loss，返回loss。

            Example：
            >>> for input, target in dataset:
                >>> def closure():
                    >>> optimizer.zero_grad()
                    >>> output = model(input)
                    >>> loss = loss_fn(output, target)
                    >>> loss.backward()
                    >>> return loss
                >>> optimizer.step(closure)

        """

        r"""Performs a single optimization step (parameter update).

        Args:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        raise NotImplementedError

    # add_param_group方法先将优化器超参数self.defaults放入param_group，然后再把param_group存到self.param_groups。
    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.
            添加一个param group到:class:`Optimizer`的param_groups

        This can be useful when fine-tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.
        对预训练的网络进行微调时很有用，因为冻结的层可以被训练，并随着训练的进行添加到:class: ' Optimizer '

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        # Result:[Parameter containing:tensor([-0.3999, -0.2694,  0.2703, -0.3355], requires_grad=True)]
        params = param_group['params']

        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        # 将优化器超参数self.defaults放入param_group
        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                 name)
            else:
                param_group.setdefault(name, default)

        params = param_group['params']
        if len(params) != len(set(params)):
            warnings.warn("optimizer contains a parameter group with duplicate parameters; "
                          "in future, this will cause an error; "
                          "see github.com/pytorch/pytorch/issues/40967 for more information", stacklevel=3)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        # 把param_group字典存到param_groups列表
        self.param_groups.append(param_group)
