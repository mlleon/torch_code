#******************************_utils.py**********************************
import torch
from typing import Optional, List, DefaultDict, Any
import warnings
from collections import defaultdict
import sys
import traceback



def _type(self, dtype=None, non_blocking=False, **kwargs):
    """Returns the type if `dtype` is not provided, else casts this object to
    the specified type.

    If this is already of the correct type, no copy is performed and the
    original object is returned.

    Args:
        dtype (type or string): The desired type
        non_blocking (bool): If ``True``, and the source is in pinned memory
            and destination is on the GPU or vice versa, the copy is performed
            asynchronously with respect to the host. Otherwise, the argument
            has no effect.
        **kwargs: For compatibility, may contain the key ``async`` in place of
            the ``non_blocking`` argument. The ``async`` arg is deprecated.
    """
    non_blocking = _get_async_or_non_blocking('type', non_blocking, kwargs)
    if dtype is None:
        return self.__module__ + '.' + self.__class__.__name__

    if isinstance(dtype, str):
        dtype = _import_dotted_name(dtype)
    if dtype == type(self):
        return self
    if self.is_sparse:
        if not dtype.is_sparse:
            raise RuntimeError("Cannot cast sparse tensor to dense tensor")
        new_module_name = dtype.__module__.replace('.sparse', '')
        new_values_type_name = new_module_name + '.' + dtype.__name__
        new_values = torch.Tensor._values(self).type(new_values_type_name, non_blocking)
        new_indices_type_name = new_module_name + '.LongTensor'
        new_indices = torch.Tensor._indices(self).type(new_indices_type_name, non_blocking)
        return dtype(new_indices, new_values, self.size())
    if dtype.is_sparse:
        raise RuntimeError("Cannot cast dense tensor to sparse tensor")
    return dtype(self.size()).copy_(self, non_blocking)


def _cuda(self, device=None, non_blocking=False, **kwargs):
    """Returns a copy of this object in CUDA memory.

    If this object is already in CUDA memory and on the correct device, then
    no copy is performed and the original object is returned.

    Args:
        device (int): The destination GPU id. Defaults to the current device.
        non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect.
        **kwargs: For compatibility, may contain the key ``async`` in place of
            the ``non_blocking`` argument.
    """
    non_blocking = _get_async_or_non_blocking('cuda', non_blocking, kwargs)
    if self.is_cuda:
        if device is None:
            device = torch.cuda.current_device()
        if self.get_device() == device:
            return self
    else:
        if device is None:
            device = -1
    with torch.cuda.device(device):
        if self.is_sparse:
            new_type = getattr(torch.cuda.sparse, self.__class__.__name__)
            indices = torch.Tensor._indices(self).cuda(device, non_blocking)
            values = torch.Tensor._values(self).cuda(device, non_blocking)
            return new_type(indices, values, self.size())
        else:
            return torch._UntypedStorage(self.size(), device=torch.device('cuda')).copy_(self, non_blocking)


def _get_async_or_non_blocking(function_name, non_blocking, kwargs):
    if not kwargs:
        return non_blocking
    if len(kwargs) != 1 or 'async' not in kwargs:
        message = "{}() got an unexpected keyword argument '{}'"
        argument = list(kwargs.keys()).pop()
        raise TypeError(message.format(function_name, argument))
    warnings.warn("'async' is deprecated; use 'non_blocking'")
    return kwargs['async']


# Note [Don't serialize hooks]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Since time immemorial, we have serialized the backward hooks associated with
# variables.  This kind of half-worked--Python can pickle global functions
# (but not closures!)--but there were problems.
#
#   - It's fragile.  If you serialize a backward hook into a saved
#     model, and then you rename the function associated with the hook,
#     now your saved model is broken and you can't load it anymore.
#
#   - It's not actually used.  The standard recommendation is to
#     serialize the *state_dict* of a model, not the model itself
#     (since this is more stable to code changes affecting the model
#     serialization), and the state dict saves "data" only, thus
#     stripping the the backward hooks.  In some cases, hooks are
#     essential to the well-functioning of a model (e.g., DDP),
#     but DDP already manages readding the hooks!
#
#   - We didn't serialize them in many cases.  Prior to #10220, we
#     were dropping backward hooks in ForkingPickler.  We "fixed" this
#     to be convenient with other serialization sites, but lack of
#     serializing backward hooks wasn't actually the root cause of
#     the bug.
#
# With these cases in mind, we have decided that a better strategy
# is to just NOT serialize hooks at all.
#
# Since this is a BC-breaking change, we should warn when we previously
# serialized a hook, but no longer do so. This will be done by adding a special
# sentinel property to hooks will be used to suppress this warning. If a hook
# has the property _torch_serialize_ignore, we will not emit a warning if we
# attempt to serialize a Tensor with this hook attached to it.
#
# By the way, when _backward_hooks is skipped, we must give an EMPTY
# OrderedDict(), if you pass a None you'll run afoul #12219.


# TODO: Once we decide to break serialization FC, `storage` no longer needs to
# be a _TypedStorage
def _rebuild_tensor(storage, storage_offset, size, stride):
    # first construct a tensor with the correct dtype/device
    t = torch.tensor([], dtype=storage.dtype, device=storage._untyped().device)
    return t.set_(storage._untyped(), storage_offset, size, stride)


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    tensor = _rebuild_tensor(storage, storage_offset, size, stride)
    tensor.requires_grad = requires_grad
    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    tensor._backward_hooks = backward_hooks
    return tensor


_sparse_tensors_to_validate: List["torch.Tensor"] = []

# In _legacy_load() in serialization.py we unpickle storages after the sparse
# tensors have been already unpickled. Those storages contain data necessary for
# validating sparse tensors: indices and values. That's why sparse tensors are
# first unpickled without any validation, and then this function is called just
# before _legacy_load() returns, so that all the sparse tensors can be validated
# in bulk.
#
# The same procedure must be followed by _load() in serialization.py because due
# to Pickler semantics, we have to use the same (non-validating) function for
# unpickling sparse tensors, regardless of the caller.
def _validate_loaded_sparse_tensors():
    try:
        for t in _sparse_tensors_to_validate:
            if t.is_sparse:
                torch._validate_sparse_coo_tensor_args(t._indices(), t._values(),
                                                       t.size())
            elif t.is_sparse_csr:
                # TODO: Validation currently involves an expensive traversal
                # on CPU, which may include a device transfer.
                torch._validate_sparse_csr_tensor_args(t.crow_indices(), t.col_indices(),
                                                       t.values(), t.size())
            else:
                raise NotImplementedError(
                    '_validate_loaded_sparse_tensors for layout `%s`' % (t.layout))

    finally:
        _sparse_tensors_to_validate.clear()

def _rebuild_sparse_tensor(layout, data):
    if layout == torch.sparse_coo:
        indices, values, size = data
        result = torch._sparse_coo_tensor_unsafe(indices, values, size)
        _sparse_tensors_to_validate.append(result)
        return result

    raise NotImplementedError("rebuilding sparse tensor for layout %s" % (layout))

def _rebuild_sparse_csr_tensor(layout, data):
    if layout == torch.sparse_csr:
        crow_indices, col_indices, values, size = data
        result = torch._sparse_csr_tensor_unsafe(crow_indices, col_indices, values, size)
        _sparse_tensors_to_validate.append(result)
        return result

    raise NotImplementedError("rebuilding sparse tensor for layout %s" % (layout))


def _rebuild_device_tensor_from_numpy(data, dtype, device, requires_grad):
    tensor = torch.from_numpy(data).to(dtype=dtype, device=device)
    tensor.requires_grad = requires_grad
    return tensor


# Should not be used, only here to be able to load Tensors serialized with older versions of pytorch
_rebuild_xla_tensor = _rebuild_device_tensor_from_numpy


def _rebuild_meta_tensor_no_storage(dtype, size, stride, requires_grad):
    return torch.empty_strided(size, stride, dtype=dtype, device='meta', requires_grad=requires_grad)


def _rebuild_wrapper_subclass(cls, dtype, size, stride, storage_offset, layout, device, requires_grad):
    return torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
        cls, size, strides=stride, storage_offset=storage_offset, layout=layout,
        device=device, requires_grad=requires_grad)


# TODO: Once we decide to break serialization FC, `storage` no longer needs to
# be a _TypedStorage
def _rebuild_qtensor(storage, storage_offset, size, stride, quantizer_params, requires_grad, backward_hooks):
    qscheme = quantizer_params[0]
    if qscheme == torch.per_tensor_affine:
        _, scale, zero_point = quantizer_params
        tensor = torch._empty_affine_quantized(size, scale=scale, zero_point=zero_point, dtype=storage.dtype)
    elif qscheme in (torch.per_channel_affine, torch.per_channel_affine_float_qparams):
        _, scales, zero_points, axis = quantizer_params
        if type(scales) is list and type(zero_points) is list:
            if qscheme == torch.per_channel_affine:
                scales = torch.tensor(scales, dtype=torch.double)
                zero_points = torch.tensor(zero_points, dtype=torch.long)
            else:
                scales = torch.tensor(scales, dtype=torch.float)
                zero_points = torch.tensor(zero_points, dtype=torch.float)
        tensor = torch._empty_per_channel_affine_quantized(
            size, scales=scales, zero_points=zero_points, axis=axis, dtype=storage.dtype)
    else:
        raise RuntimeError("Can't deserialize quantized tensor with qscheme {}".format(qscheme))
    tensor.set_(storage, storage_offset, size, stride)
    tensor.requires_grad = requires_grad
    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    tensor._backward_hooks = backward_hooks
    return tensor

def _rebuild_parameter(data, requires_grad, backward_hooks):
    param = torch.nn.Parameter(data, requires_grad)
    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    param._backward_hooks = backward_hooks

    return param


def _import_dotted_name(name):
    components = name.split('.')
    obj = __import__(components[0])
    for component in components[1:]:
        obj = getattr(obj, component)
    return obj


# Taken from python 3.5 docs
def _accumulate(iterable, fn=lambda x, y: x + y):
    'Return running totals'
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total


def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.

    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.

    Args:
        tensors (Iterable[Tensor]): dense tensors to flatten.

    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    return torch._C._nn.flatten_dense_tensors(tensors)


def _flatten_sparse_tensors(tensors):
    """Flatten sparse tensors into two contiguous 1D buffers, one of indices and
    one of values. Assume tensors are of same sparse type.

    Args:
        tensors (Iterable[Tensor]): sparse tensors to flatten.

    Returns:
        A tuple of two contiguous 1D buffers, one containing input tensors'
        indices and the other containing the values.
    """
    flat_indices = torch._C._nn.flatten_dense_tensors([torch.Tensor._indices(t) for t in tensors])
    flat_values = torch._C._nn.flatten_dense_tensors([torch.Tensor._values(t) for t in tensors])
    return flat_indices, flat_values


def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.

    Args:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    return torch._C._nn.unflatten_dense_tensors(flat, tensors)


def _unflatten_sparse_tensors(flat, tensors):
    """View flat buffer (containing indices and values) using the sizes of
    tensors. Assume that tensors are of same sparse type, and that flat is given
    by _flatten_sparse_tensors.

    Args:
        flat (tuple(Tensor, Tensor)): flattened indices and values of sparse
          tensors to unflatten.
        tensors (Iterable[Tensor]): sparse tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened sparse tensors with sizes same as tensors and values from
        flat.
    """
    flat_indices, flat_values = flat
    indices = torch._C._nn.unflatten_dense_tensors(flat_indices, [torch.Tensor._indices(t) for t in tensors])
    values = torch._C._nn.unflatten_dense_tensors(flat_values, [torch.Tensor._values(t) for t in tensors])
    outputs = []
    for t, i, v in zip(tensors, indices, values):
        outputs.append(t.new(i, v, t.size()))
    return tuple(outputs)


def _reorder_tensors_as(tensors, ordered_tensors):
    """Assume that tensors are of same order as ordered_tensors within their
    types, e.g., from _take_tensors. Reorder them to be of same order as
    ordered_tensors.

    Args:
        tensors (Iterable[Tensor]): tensors to be reordered. They should be of
          the same order as ordered_tensors within their own types.
        ordered_tensors (Iterable[Tensor]): tensors whose order will be the
          reference.

    Returns:
        Ordered tuple of tensors with contents from tensors and order of
        ordered_tensors.
    """
    type_dict = defaultdict(list)
    for tensor in tensors:
        type_dict[tensor.type()].append(tensor)
    type_dict_ = {t: iter(coll) for t, coll in type_dict.items()}
    return tuple(next(type_dict_[tensor.type()]) for tensor in ordered_tensors)


def _take_tensors(tensors, size_limit):
    """Group tensors into chunks. This generator yields a chunk at each time,
    each containing tensors of same type up to certain byte limit in total size.

    Args:
        tensors (Sequence): A sequence of tensors to be separated into chunks.
        size_limit (int): The limit of each chunk in bytes.

    Yields:
        Blocks of tensors of same type and within size_limit. The yielded
        tensors are only ordered as the original sequence within its types.
    """
    buf_dict: DefaultDict[str, List] = defaultdict(lambda: [[], 0])
    for tensor in tensors:
        t = tensor.type()
        if tensor.is_sparse:
            indices = torch.Tensor._indices(tensor)
            values = torch.Tensor._values(tensor)
            size = indices.numel() * indices.element_size() + values.numel() * values.element_size()
        else:
            size = tensor.numel() * tensor.element_size()
        buf_and_size = buf_dict[t]
        if buf_and_size[1] + size > size_limit and buf_and_size[1] > 0:
            yield buf_and_size[0]
            buf_and_size = buf_dict[t] = [[], 0]
        buf_and_size[0].append(tensor)
        buf_and_size[1] += size
    for buf, _ in buf_dict.values():
        if len(buf) > 0:
            yield buf


# annotation decorator to get annotations in a way that is compatible
# with both Python 2 and 3
def annotate(ret, **kwargs):
    def dec(fun):
        fun.__annotations__ = dict(kwargs)
        fun.__annotations__['return'] = ret
        return fun
    return dec


# NOTE [ Python Traceback Reference Cycle Problem ]
#
# When using sys.exc_info(), it is important to **not** store the exc_info[2],
# which is the traceback, because otherwise you will run into the traceback
# reference cycle problem, i.e., the traceback holding reference to the frame,
# and the frame (which holds reference to all the object in its temporary scope)
# holding reference the traceback.

class KeyErrorMessage(str):
    r"""str subclass that returns itself in repr"""
    def __repr__(self):
        return self


class ExceptionWrapper(object):
    r"""Wraps an exception plus traceback to communicate across threads"""
    def __init__(self, exc_info=None, where="in background"):
        # It is important that we don't store exc_info, see
        # NOTE [ Python Traceback Reference Cycle Problem ]
        if exc_info is None:
            exc_info = sys.exc_info()
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))
        self.where = where

    def reraise(self):
        r"""Reraises the wrapped exception in the current thread"""
        # Format a message such as: "Caught ValueError in DataLoader worker
        # process 2. Original Traceback:", followed by the traceback.
        msg = "Caught {} {}.\nOriginal {}".format(
            self.exc_type.__name__, self.where, self.exc_msg)
        if self.exc_type == KeyError:
            # KeyError calls repr() on its argument (usually a dict key). This
            # makes stack traces unreadable. It will not be changed in Python
            # (https://bugs.python.org/issue2651), so we work around it.
            msg = KeyErrorMessage(msg)
        elif getattr(self.exc_type, "message", None):
            # Some exceptions have first argument as non-str but explicitly
            # have message field
            raise self.exc_type(message=msg)
        try:
            exception = self.exc_type(msg)
        except TypeError:
            # If the exception takes multiple arguments, don't try to
            # instantiate since we don't know how to
            raise RuntimeError(msg) from None
        raise exception


def _get_available_device_type():
    if torch.cuda.is_available():
        return "cuda"
    # add more available device types here
    return None


def _get_device_attr(get_member):

    device_type = _get_available_device_type() # 判断是否支持CUDA(支持返回"CUDA"，不支持返回None)
    if device_type and device_type.lower() == "cuda":
        # get_member是一个lambda方法，如get_member=lambda m: list(range(m.device_count()))
        return get_member(torch.cuda)
    # add more available device types here
    # 在这里添加更多可用的设备类型
    return None # 如果不支持CUDA，返回None


def _get_current_device_index():
    # current device index
    return _get_device_attr(lambda m: m.current_device())


def _get_all_device_indices(): # 返回所有设备索引
    # all device index
    return _get_device_attr(lambda m: list(range(m.device_count())))


def _get_devices_properties(device_ids):
    # all device properties(返回所有显卡的各种信息)
    return [_get_device_attr(lambda m: m.get_device_properties(i)) for i in device_ids]

def get_current_device_index() -> int:
    r"""Checks if there are CUDA devices available and
    returns the device index of the current default CUDA device.
    Returns -1 in case there are no CUDA devices available.
    Arguments: ``None``
    核查是否有CUDA设备可用，如果有返回当前默认CUDA设备的索引，没有CUDA设备可用返回-1
    """
    if torch.cuda.device_count() > 0:
        return torch.cuda.current_device()
    return -1

def _get_device_index(device: Any, optional: bool = False, allow_cpu: bool = False) -> int:
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.
    获取attr:`device`的设备索引，可以是torch.device、Python integer, or ``None``

    If :attr:`device` is a torch.device object, returns the device index if it
    has index. Note that for a device without a specified index,
    i.e., ``torch.device('xxx')``, this will return the current default
    device of that type if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.
    如果attr:`device`是torch.device对象，返回这个设备的索引（如果有索引）
    如果attr:`optional=True，对于没有指定索引的device，将返回当前默认设备
    如果attr:`allow_cpu=True，CPU设备将被接受，并且在本例中将返回' ' -1 ' '。

    If :attr:`device` is a Python integer, it is returned as is.
    如果attr:`device`是一个Python integer（整数），它会按原样返回

    If :attr:`device` is ``None``, this will return the current default
    device of the supported runtime platform if :attr:`optional` is ``True``.
    i.e., the current default CUDA device will be returned if CUDA runtime is supported.
    如果attr:`device=None，并且如果attr:`optional=True，这将返回当前的受支持的运行时平台的默认设备
    例如，如果支持CUDA运行时，则返回当前默认的CUDA设备。
    """

    if isinstance(device, str):
        device = torch.device(device)
    device_idx: Optional[int] = None
    if isinstance(device, torch.device):
        if not allow_cpu and device.type == 'cpu':
            raise ValueError('Expected a non cpu device, but got: {}'.format(device))
        device_idx = -1 if device.type == 'cpu' else device.index
    if isinstance(device, int):
        device_idx = device
    if device_idx is None:
        if optional:
            # The eager API _get_current_device_index uses `lambda` functions which are
            # not supported in JIT and hence not scriptable. The JIT equivalent API to get
            # the current device index is `get_current_device_index()` which can
            # be scripted. We use is_scripting to check the mode we are in and call the
            # appropriate API.
            if torch.jit.is_scripting():
                device_idx = get_current_device_index()
            else:
                device_idx = _get_current_device_index()
        else:
            raise ValueError('Expected a torch.device with a specified index '
                             'or an integer, but got:{}'.format(device))
            # 要求使用一个具有指定索引的torch.device或一个整数
    return device_idx


def _handle_complex(tensor):
    """
    Returns a real view of a tensor if complex dtype else just the tensor
    need to check if a UninitializedParameter because otherwise checking is_complex is an error for a LazyModule
    """
    return torch.view_as_real(tensor) if not isinstance(tensor,
                                                        torch.nn.UninitializedParameter) and tensor.is_complex() else tensor

def _element_size(dtype):
    """
    Returns the element size for a dtype, in bytes
    """
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f'expected torch.dtype, but got {type(dtype)}')

    if dtype.is_complex:
        return torch.finfo(dtype).bits >> 2
    elif dtype.is_floating_point:
        return torch.finfo(dtype).bits >> 3
    elif dtype == torch.bool:
        # NOTE: torch.bool is not supported in torch.iinfo()
        return 1
    else:
        return torch.iinfo(dtype).bits >> 3

class _ClassPropertyDescriptor:
    def __init__(self, fget, fset=None):
        self.fget = fget

    def __get__(self, instance, owner=None):
        if owner is None:
            owner = type(instance)
        return self.fget.__get__(instance, owner)()

def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)
    return _ClassPropertyDescriptor(func)


# ************************** _functions.py ***********************************
import warnings

import torch
from . import comm
from torch.autograd import Function
from torch._utils import _get_device_index
from typing import List, Optional

#  Broadcast.apply
class Broadcast(Function):

    @staticmethod
    def forward(ctx, target_gpus, *inputs):
        assert all(i.device.type != 'cpu' for i in inputs), (
            'Broadcast function not implemented for CPU tensors'
        )
        target_gpus = [_get_device_index(x, True) for x in target_gpus]
        ctx.target_gpus = target_gpus
        if len(inputs) == 0:
            return tuple()
        ctx.num_inputs = len(inputs)
        # input 放在 device[0]
        ctx.input_device = inputs[0].get_device()
        # 和 detach 的情况一样
        outputs = comm.broadcast_coalesced(inputs, ctx.target_gpus)
        """
            # comm.broadcast_coalesced 的代码
            # tensors 必须在同一个设备，CPU 或者 GPU； devices 即是要拷贝到的设备；buffer_size 则是最大的buffer
            # 这里用到 buffer 将小张量合并到缓冲区以减少同步次数
            # def broadcast_coalesced(tensors, devices, buffer_size=10485760):
            # devices = [_get_device_index(d) for d in devices]
            # return torch._C._broadcast_coalesced(tensors, devices, buffer_size)
        """
        non_differentiables = []
        for idx, input_requires_grad in enumerate(ctx.needs_input_grad[1:]):
            if not input_requires_grad:
                for output in outputs:
                    non_differentiables.append(output[idx])
        ctx.mark_non_differentiable(*non_differentiables)
        return tuple([t for tensors in outputs for t in tensors])

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None,) + ReduceAddCoalesced.apply(ctx.input_device, ctx.num_inputs, *grad_outputs)


class ReduceAddCoalesced(Function):

    @staticmethod
    def forward(ctx, destination, num_inputs, *grads):
        ctx.target_gpus = [grads[i].get_device() for i in range(0, len(grads), num_inputs)]

        grads_ = [grads[i:i + num_inputs]
                  for i in range(0, len(grads), num_inputs)]
        return comm.reduce_add_coalesced(grads_, destination)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None,) + Broadcast.apply(ctx.target_gpus, *grad_outputs)


class Gather(Function):

    @staticmethod
    def forward(ctx, target_device, dim, *inputs):
        assert all(i.device.type != 'cpu' for i in inputs), (
            'Gather function not implemented for CPU tensors'
        )
        if (target_device == 'cpu'):
            ctx.target_device = 'cpu'
        else:
            target_device = _get_device_index(target_device, True)
            ctx.target_device = target_device
        ctx.dim = dim
        ctx.input_gpus = tuple(i.get_device() for i in inputs)
        if all(t.dim() == 0 for t in inputs) and dim == 0:
            inputs = tuple(t.view(1) for t in inputs)
            warnings.warn('Was asked to gather along dimension 0, but all '
                          'input tensors were scalars; will instead unsqueeze '
                          'and return a vector.')
            ctx.unsqueezed_scalar = True
        else:
            ctx.unsqueezed_scalar = False
        ctx.input_sizes = tuple(i.size(ctx.dim) for i in inputs)
        return comm.gather(inputs, ctx.dim, ctx.target_device)

    @staticmethod
    def backward(ctx, grad_output):
        scattered_grads = Scatter.apply(ctx.input_gpus, ctx.input_sizes, ctx.dim, grad_output)
        if ctx.unsqueezed_scalar:
            scattered_grads = tuple(g[0] for g in scattered_grads)
        return (None, None) + scattered_grads


class Scatter(Function):
    # 针对 tensor 的函数

    @staticmethod
    def forward(ctx, target_gpus, chunk_sizes, dim, input):
        target_gpus = [_get_device_index(x, True) for x in target_gpus]
        ctx.dim = dim
        ctx.input_device = input.get_device() if input.device.type != "cpu" else -1
        streams = None
        if torch.cuda.is_available() and ctx.input_device == -1:
            # Perform CPU to GPU copies in a background stream
            # 新建 cuda stream
            streams = [_get_stream(device) for device in target_gpus]

        # 真正的操作（comm.scatter 依赖于 C++）
        outputs = comm.scatter(input, target_gpus, chunk_sizes, ctx.dim, streams)
        # Synchronize with the copy stream
        if streams is not None:
            for i, output in enumerate(outputs):
                with torch.cuda.device(target_gpus[i]):
                    main_stream = torch.cuda.current_stream()
                    main_stream.wait_stream(streams[i])
                    output.record_stream(main_stream)
        return outputs

    # staticmethod用于修饰类中的方法, 使其可以在不创建类实例的情况下调用方法，这样做的好处是执行效率比较高
    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, Gather.apply(ctx.input_device, ctx.dim, *grad_output)


# background streams used for copying
_streams: Optional[List[Optional[torch.cuda.Stream]]] = None


def _get_stream(device: int):
    """Gets a background stream for copying between CPU and GPU"""
    global _streams
    if device == -1:
        return None
    if _streams is None:
        _streams = [None] * torch.cuda.device_count()
    if _streams[device] is None:
        _streams[device] = torch.cuda.Stream(device)
    return _streams[device]


# ************************** scatter_gather.py *******************************
# 路径：~/conda/lib/python3.9/site-packages/torch/nn/parallel/scatter_gather.py

import torch
from ._functions import Scatter, Gather

def is_namedtuple(obj):
    # Check if type was created from collections.namedtuple or a typing.NamedTuple.
    return (
        isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")
    )


def scatter(inputs, target_gpus, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    将张量切成近似相等的块，并将它们分给不同的 GPU
    对其他的数据类型，则是复制多份分给不同的 GPU

    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            # 针对 tensor 的函数
            return Scatter.apply(target_gpus, None, dim, obj)
        if is_namedtuple(obj):
            return [type(obj)(*args) for args in zip(*map(scatter_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(scatter_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None
    return res


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    # 主要函数
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    # 用空项补全使 inputs 和 kwargs 长度相当
    if len(inputs) < len(kwargs):
        inputs.extend(() for _ in range(len(kwargs) - len(inputs)))
    elif len(kwargs) < len(inputs):
        kwargs.extend({} for _ in range(len(inputs) - len(kwargs)))
    # 返回 tuple
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


def gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device.
    Use 'cpu' for CPU to avoid a deprecation warning.
    """
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all(len(out) == len(d) for d in outputs):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)((k, gather_map([d[k] for d in outputs]))
                             for k in out)
        if is_namedtuple(out):
            return type(out)._make(map(gather_map, zip(*outputs)))
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        res = gather_map(outputs)
    finally:
        gather_map = None
    return res


# **************************** replicate.py **********************************
from . import comm
from torch._utils import _get_device_index

from collections import OrderedDict


def _is_script_module(module):
    import torch.jit
    return isinstance(module, torch.jit.ScriptModule)


def _is_script_method(module):
    import torch.jit
    return isinstance(module, torch._C.ScriptMethod)


def _init_script_module():
    import torch.jit
    return torch.jit.ScriptModule()


def _is_jit_enabled():
    import torch.jit
    return torch.jit._state._enabled


# Check if we can safely replicate the module.
# there are two types of module:
# 1. python modules
# 2. ScriptModule
#
# currently a module cannot be replicated properly if the descendants of
# any ScriptModule contains python module (type 1 above)
def _replicatable_module(module, memo=None):

    # module.modules() contains module itself as the first element
    def descendant_modules(module):
        gen = module.modules()
        next(gen)
        return gen

    if not _is_jit_enabled():
        return True
    if memo is None:
        memo = set()

    # memoize visited modules
    memo.add(module)
    if _is_script_module(module):
        memo.update(descendant_modules(module))
        return all(_is_script_module(descendant) for
                   descendant in descendant_modules(module))

    for child in module.children():
        # since any unreplicatable module will cause the check to return
        # False early, visited modules here can be safely ignored.
        if child in memo:
            continue
        if not _replicatable_module(child, memo):
            return False

    return True

# ！！！从replicate来看这里
def _broadcast_coalesced_reshape(tensors, devices, detach=False):
    from ._functions import Broadcast
    # 先看 else 的 comment，因为不 detach 也会用到同样的函数
    if detach:
        return comm.broadcast_coalesced(tensors, devices)
    else:
        # Use the autograd function to broadcast if not detach
        if len(tensors) > 0:
            # 下拉看源码
            tensor_copies = Broadcast.apply(devices, *tensors)
            return [tensor_copies[i:i + len(tensors)]
                    for i in range(0, len(tensor_copies), len(tensors))]
        else:
            return []


def replicate(network, devices, detach=False):
    if not _replicatable_module(network):
        raise RuntimeError("Cannot replicate network where python modules are "
                           "childrens of ScriptModule")

    if not devices:
        return []

    # 需要复制到哪些 GPU， 复制多少份
    devices = [_get_device_index(x, True) for x in devices]
    num_replicas = len(devices)

    # 复制 parameters
    params = list(network.parameters())
    param_indices = {param: idx for idx, param in enumerate(params)}
    # 拉到代码块底部看原函数，然后再回来
    param_copies = _broadcast_coalesced_reshape(params, devices, detach)

    # 复制 buffers
    buffers = list(network.buffers())
    buffers_rg = []
    buffers_not_rg = []
    for buf in buffers:
        if buf.requires_grad and not detach:
            buffers_rg.append(buf)
        else:
            buffers_not_rg.append(buf)

    # 记录需要和不需要求导的 buffer 的 index
    buffer_indices_rg = {buf: idx for idx, buf in enumerate(buffers_rg)}
    buffer_indices_not_rg = {buf: idx for idx, buf in enumerate(buffers_not_rg)}

    # 分别拷贝，这个咱们已经会了
    buffer_copies_rg = _broadcast_coalesced_reshape(buffers_rg, devices, detach=detach)
    buffer_copies_not_rg = _broadcast_coalesced_reshape(buffers_not_rg, devices, detach=True)

    """
        # 现在开始拷贝网络
        # 准备过程：将 network.modules() 变成list
        # 然后再为之后复制的模型准备好空的 list 和 indices
    """
    modules = list(network.modules())
    module_copies = [[] for device in devices]
    module_indices = {}

    for i, module in enumerate(modules):
        module_indices[module] = i
        for j in range(num_replicas):
            replica = module._replicate_for_data_parallel()
            # This is a temporary fix for DDP. DDP needs to access the
            # replicated model parameters. It used to do so through
            # `mode.parameters()`. The fix added in #33907 for DP stops the
            # `parameters()` API from exposing the replicated parameters.
            # Hence, we add a `_former_parameters` dict here to support DDP.
            replica._former_parameters = OrderedDict()

            module_copies[j].append(replica)

    # 接下来分别复制 module，param，buffer
    for i, module in enumerate(modules):
        for key, child in module._modules.items():
            if child is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._modules[key] = None
            else:
                module_idx = module_indices[child]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    setattr(replica, key, module_copies[j][module_idx])
        for key, param in module._parameters.items():
            if param is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._parameters[key] = None
            else:
                param_idx = param_indices[param]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    param = param_copies[j][param_idx]
                    # parameters in replicas are no longer leaves,
                    # so setattr them as non-parameter attributes
                    setattr(replica, key, param)
                    # expose the parameter for DDP
                    replica._former_parameters[key] = param
        for key, buf in module._buffers.items():
            if buf is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._buffers[key] = None
            else:
                if buf.requires_grad and not detach:
                    buffer_copies = buffer_copies_rg
                    buffer_idx = buffer_indices_rg[buf]
                else:
                    buffer_copies = buffer_copies_not_rg
                    buffer_idx = buffer_indices_not_rg[buf]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    setattr(replica, key, buffer_copies[j][buffer_idx])

    return [module_copies[j][0] for j in range(num_replicas)]


# ************************** parallel_apply.py *******************************
import threading
import torch
from torch.cuda._utils import _get_device_index
from torch.cuda.amp import autocast
from torch._utils import ExceptionWrapper


def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None

# threading 实现，用前面准备好的 replica 和输入数据，然后for 循环启动多线程
def parallel_apply(modules, inputs, kwargs_tup=None, devices=None):
    r"""Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.

    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices

    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    # 每个 GPU 都有模型和输入数据
    assert len(modules) == len(inputs)
    # 确保每个 GPU 都有相应的数据，如没有就空白补全
    if kwargs_tup is not None:
        # 在 scatter 已经补全了
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    devices = [_get_device_index(x, True) for x in devices]
    # 多线程实现
    lock = threading.Lock()
    results = {}
    grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()

    # 定义 worker
    def _worker(i, module, input, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device), autocast(enabled=autocast_enabled):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module(*input, **kwargs)
            with lock:
                # 并行计算得到输出
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        # 如有一个进程控制多个 GPU ，起多个线程
        # 需要强调一下，虽然 DDP 推荐单卡单进程，即每次调用 DDP device_ids 都只输入一张卡的 id（通常是 args.local_rank），
        # 但是如果输入多个 device_id，此时 DDP 就是单进程多线程控制多卡，和 DP 一样
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device))
                   for i, (module, input, kwargs, device) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        # 一个 GPU 一个进程 （ DDP 推荐操作）
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        # error handle
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    # 输出 n 个计算结果
    return outputs


# ******************************* comm.py ************************************
import warnings
import torch
from torch.cuda import nccl
from torch._utils import _take_tensors, _flatten_dense_tensors, \
    _unflatten_dense_tensors, _reorder_tensors_as, _get_device_index, _handle_complex
from typing import List

def broadcast(tensor, devices=None, *, out=None):
    r"""Broadcasts a tensor to specified GPU devices.

    Args:
        tensor (Tensor): tensor to broadcast. Can be on CPU or GPU.
        devices (Iterable[torch.device, str or int], optional): an iterable of
          GPU devices, among which to broadcast.
        out (Sequence[Tensor], optional, keyword-only): the GPU tensors to
          store output results.

    .. note::
        Exactly one of :attr:`devices` and :attr:`out` must be specified.

    Returns:
        - If :attr:`devices` is specified,
            a tuple containing copies of :attr:`tensor`, placed on
            :attr:`devices`.
        - If :attr:`out` is specified,
            a tuple containing :attr:`out` tensors, each containing a copy of
            :attr:`tensor`.
    """
    tensor = _handle_complex(tensor)
    if not ((devices is None) ^ (out is None)):
        raise RuntimeError(
            "Exactly one of 'devices' and 'out' must be specified, but got "
            "devices={} and out={}".format(devices, out))
    if devices is not None:
        devices = [_get_device_index(d) for d in devices]
        return torch._C._broadcast(tensor, devices)
    else:
        return torch._C._broadcast_out(tensor, out)


def broadcast_coalesced(tensors, devices, buffer_size=10485760):
    """Broadcasts a sequence tensors to the specified GPUs.
    Small tensors are first coalesced into a buffer to reduce the number
    of synchronizations.

    Args:
        tensors (sequence): tensors to broadcast. Must be on the same device,
          either CPU or GPU.
        devices (Iterable[torch.device, str or int]): an iterable of GPU
          devices, among which to broadcast.
        buffer_size (int): maximum size of the buffer used for coalescing

    Returns:
        A tuple containing copies of :attr:`tensor`, placed on :attr:`devices`.
    """
    devices = [_get_device_index(d) for d in devices]
    tensors = [_handle_complex(t) for t in tensors]
    return torch._C._broadcast_coalesced(tensors, devices, buffer_size)


def reduce_add(inputs, destination=None):
    """Sums tensors from multiple GPUs.

    All inputs should have matching shapes, dtype, and layout. The output tensor
    will be of the same shape, dtype, and layout.

    Args:
        inputs (Iterable[Tensor]): an iterable of tensors to add.
        destination (int, optional): a device on which the output will be
            placed (default: current device).

    Returns:
        A tensor containing an elementwise sum of all inputs, placed on the
        :attr:`destination` device.
    """
    destination = _get_device_index(destination, optional=True)
    input_size = inputs[0].size()
    root_index = None  # index of input tensor that already is on the correct device
    for i, inp in enumerate(inputs):
        assert inp.device.type != "cpu", "reduce_add expects all inputs to be on GPUs"
        if inp.get_device() == destination:
            root_index = i
        if inp.size() != input_size:
            got = 'x'.join(str(x) for x in inp.size())
            expected = 'x'.join(str(x) for x in input_size)
            raise ValueError("input {} has invalid size: got {}, but expected "
                             "{}".format(i, got, expected))
    if root_index is None:
        raise RuntimeError("reduce_add expects destination to be on the same GPU with one of the tensors")

    if len(inputs) == 1:
        return inputs[0]

    if nccl.is_available(inputs):
        result = torch.empty_like(inputs[root_index])
        nccl.reduce(inputs, output=result, root=root_index)
    else:
        destination_device = torch.device(inputs[root_index].device.type, destination)
        nonroot = [t for i, t in enumerate(inputs) if i != root_index]
        # make a new tensor w/o clone
        result = inputs[root_index] + nonroot[0].to(device=destination_device, non_blocking=True)
        for other in nonroot[1:]:
            result.add_(other.to(device=destination_device, non_blocking=True))
    return result


def reduce_add_coalesced(inputs, destination=None, buffer_size=10485760):
    """Sums tensors from multiple GPUs.

    Small tensors are first coalesced into a buffer to reduce the number
    of synchronizations.

    Args:
        inputs (Iterable[Iterable[Tensor]]): iterable of iterables that
            contain tensors from a single device.
        destination (int, optional): a device on which the output will be
            placed (default: current device).
        buffer_size (int): maximum size of the buffer used for coalescing

    Returns:
        A tuple of tensors containing an elementwise sum of each group of
        inputs, placed on the ``destination`` device.
    """
    # TODO: When `len(inputs) == 1` and all inputs are on `destination`, just
    #       return `inputs`.
    dense_tensors: List[List] = [[] for _ in inputs]  # shape (num_gpus, num_tensors)
    output = []
    ref_order = []
    # process sparse ones first since they may have different sizes on different gpus
    for tensor_at_gpus in zip(*inputs):
        if all(t.is_sparse for t in tensor_at_gpus):
            result = reduce_add(tensor_at_gpus, destination)  # this will be sparse too
            output.append(result)
            ref_order.append(tensor_at_gpus[0])
        else:
            for coll, t in zip(dense_tensors, tensor_at_gpus):
                coll.append(t.to_dense() if t.is_sparse else t)
            ref_order.append(dense_tensors[0][-1])
    itrs = [_take_tensors(tensors, buffer_size) for tensors in dense_tensors]
    # now the dense ones, which have consistent sizes
    for chunks in zip(*itrs):
        flat_tensors = [_flatten_dense_tensors(chunk) for chunk in chunks]  # (num_gpus,)
        flat_result = reduce_add(flat_tensors, destination)
        for t in _unflatten_dense_tensors(flat_result, chunks[0]):
            # The unflattened tensors do not share storage, and we don't expose
            # base flat tensor anyways, so give them different version counters.
            # See NOTE [ Version Counter in comm.*_coalesced ]
            output.append(t.data)
    return tuple(_reorder_tensors_as(output, ref_order))


def scatter(tensor, devices=None, chunk_sizes=None, dim=0, streams=None, *, out=None):
    """Scatters tensor across multiple GPUs.

    Args:
        tensor (Tensor): tensor to scatter. Can be on CPU or GPU.
        devices (Iterable[torch.device, str or int], optional): an iterable of
          GPU devices, among which to scatter.
        chunk_sizes (Iterable[int], optional): sizes of chunks to be placed on
          each device. It should match :attr:`devices` in length and sums to
          ``tensor.size(dim)``. If not specified, :attr:`tensor` will be divided
          into equal chunks.
        dim (int, optional): A dimension along which to chunk :attr:`tensor`.
          Default: ``0``.
        streams (Iterable[Stream], optional): an iterable of Streams, among
          which to execute the scatter. If not specified, the default stream will
          be utilized.
        out (Sequence[Tensor], optional, keyword-only): the GPU tensors to
          store output results. Sizes of these tensors must match that of
          :attr:`tensor`, except for :attr:`dim`, where the total size must
          sum to ``tensor.size(dim)``.

    .. note::
        Exactly one of :attr:`devices` and :attr:`out` must be specified. When
        :attr:`out` is specified, :attr:`chunk_sizes` must not be specified and
        will be inferred from sizes of :attr:`out`.

    Returns:
        - If :attr:`devices` is specified,
            a tuple containing chunks of :attr:`tensor`, placed on
            :attr:`devices`.
        - If :attr:`out` is specified,
            a tuple containing :attr:`out` tensors, each containing a chunk of
            :attr:`tensor`.
    """
    tensor = _handle_complex(tensor)
    if out is None:
        devices = [_get_device_index(d) for d in devices]
        return tuple(torch._C._scatter(tensor, devices, chunk_sizes, dim, streams))
    else:
        if devices is not None:
            raise RuntimeError(
                "'devices' must not be specified when 'out' is specified, but "
                "got devices={}".format(devices))
        if chunk_sizes is not None:
            raise RuntimeError(
                "'chunk_sizes' must not be specified when 'out' is specified, "
                "but got chunk_sizes={}".format(chunk_sizes))
        return tuple(torch._C._scatter_out(tensor, out, dim, streams))

# comm.gather 涉及到 C++，具体实现咱也不讲了 ；)
# Gathers tensors from multiple GPU devices.
def gather(tensors, dim=0, destination=None, *, out=None):
    r"""Gathers tensors from multiple GPU devices.

    Args:
        tensors (Iterable[Tensor]): an iterable of tensors to gather.
          Tensor sizes in all dimensions other than :attr:`dim` have to match.
        dim (int, optional): a dimension along which the tensors will be
          concatenated. Default: ``0``.
        destination (torch.device, str, or int, optional): the output device.
          Can be CPU or CUDA. Default: the current CUDA device.
        out (Tensor, optional, keyword-only): the tensor to store gather result.
          Its sizes must match those of :attr:`tensors`, except for :attr:`dim`,
          where the size must equal ``sum(tensor.size(dim) for tensor in tensors)``.
          Can be on CPU or CUDA.

    .. note::
        :attr:`destination` must not be specified when :attr:`out` is specified.

    Returns:
        - If :attr:`destination` is specified,
            a tensor located on :attr:`destination` device, that is a result of
            concatenating :attr:`tensors` along :attr:`dim`.
        - If :attr:`out` is specified,
            the :attr:`out` tensor, now containing results of concatenating
            :attr:`tensors` along :attr:`dim`.
    """
    tensors = [_handle_complex(t) for t in tensors]
    if out is None:
        if destination == -1:
            warnings.warn(
                'Using -1 to represent CPU tensor is deprecated. Please use a '
                'device object or string instead, e.g., "cpu".')
        destination = _get_device_index(destination, allow_cpu=True, optional=True)
        return torch._C._gather(tensors, dim, destination)
    else:
        if destination is not None:
            raise RuntimeError(
                "'destination' must not be specified when 'out' is specified, but "
                "got destination={}".format(destination))
        return torch._C._gather_out(tensors, out, dim)


# **************************** DATA_PARALLEL *********************************
# SOURCE CODE FOR TORCH.NN.PARALLEL.DATA_PARALLEL

import operator
import torch
import warnings
from itertools import chain
from ..modules import Module
from . import scatter_kwargs, gather
from .replicate import replicate
from .parallel_apply import parallel_apply
from torch._utils import (scatter_gather
    _get_all_device_indices,
    _get_available_device_type,
    _get_device_index,
    _get_devices_properties
)

__all__ = ['DataParallel', 'data_parallel']


def _check_balance(device_ids):
    imbalance_warn = """
    There is an imbalance between your GPUs. You may want to exclude GPU {} which
    has less than 75% of the memory or cores of GPU {}. You can do so by setting
    the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES
    environment variable."""
    device_ids = [_get_device_index(x, True) for x in device_ids] # 返回所有显卡索引
    dev_props = _get_devices_properties(device_ids)    # 返回所有显卡各种信息

    def warn_imbalance(get_prop):
        # get_prop是一个lambda, 如get_prop=lambda props: props.total_memory
        values = [get_prop(props) for props in dev_props]
        min_pos, min_val = min(enumerate(values), key=operator.itemgetter(1))
        max_pos, max_val = max(enumerate(values), key=operator.itemgetter(1))
        if min_val / max_val < 0.75:
            warnings.warn(imbalance_warn.format(device_ids[min_pos], device_ids[max_pos]))
            return True
        return False

    if warn_imbalance(lambda props: props.total_memory):
        return
    if warn_imbalance(lambda props: props.multi_processor_count):
        return


class DataParallel(Module):
    r"""Implements data parallelism at the module level.

    This container parallelizes the application of the given :attr:`module` by
    splitting the input across the specified devices by chunking in the batch
    dimension (other objects will be copied once per device). In the forward
    pass, the module is replicated on each device, and each replica handles a
    portion of the input. During the backwards pass, gradients from each replica
    are summed into the original module.

    The batch size should be larger than the number of GPUs used.

    .. warning::
        It is recommended to use :class:`~torch.nn.parallel.DistributedDataParallel`,
        instead of this class, to do multi-GPU training, even if there is only a single
        node. See: :ref:`cuda-nn-ddp-instead` and :ref:`ddp`.

    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel but some types are specially handled. tensors will be
    **scattered** on dim specified (default 0). tuple, list and dict types will
    be shallow copied. The other types will be shared among different threads
    and can be corrupted if written to in the model's forward pass.

    The parallelized :attr:`module` must have its parameters and buffers on
    ``device_ids[0]`` before running this :class:`~torch.nn.DataParallel`
    module.

    .. warning::
        In each forward, :attr:`module` is **replicated** on each device, so any
        updates to the running module in ``forward`` will be lost. For example,
        if :attr:`module` has a counter attribute that is incremented in each
        ``forward``, it will always stay at the initial value because the update
        is done on the replicas which are destroyed after ``forward``. However,
        :class:`~torch.nn.DataParallel` guarantees that the replica on
        ``device[0]`` will have its parameters and buffers sharing storage with
        the base parallelized :attr:`module`. So **in-place** updates to the
        parameters or buffers on ``device[0]`` will be recorded. E.g.,
        :class:`~torch.nn.BatchNorm2d` and :func:`~torch.nn.utils.spectral_norm`
        rely on this behavior to update the buffers.

    .. warning::
        Forward and backward hooks defined on :attr:`module` and its submodules
        will be invoked ``len(device_ids)`` times, each with inputs located on
        a particular device. Particularly, the hooks are only guaranteed to be
        executed in correct order with respect to operations on corresponding
        devices. For example, it is not guaranteed that hooks set via
        :meth:`~torch.nn.Module.register_forward_pre_hook` be executed before
        `all` ``len(device_ids)`` :meth:`~torch.nn.Module.forward` calls, but
        that each such hook be executed before the corresponding
        :meth:`~torch.nn.Module.forward` call of that device.

    .. warning::
        When :attr:`module` returns a scalar (i.e., 0-dimensional tensor) in
        :func:`forward`, this wrapper will return a vector of length equal to
        number of devices used in data parallelism, containing the result from
        each device.

    .. note::
        There is a subtlety in using the
        ``pack sequence -> recurrent network -> unpack sequence`` pattern in a
        :class:`~torch.nn.Module` wrapped in :class:`~torch.nn.DataParallel`.
        See :ref:`pack-rnn-unpack-with-data-parallelism` section in FAQ for
        details.


    Args:
        module (Module): module to be parallelized # 需要并行化的模块
        device_ids (list of int or torch.device): CUDA devices (default: all devices)# CUDA设备（默认所有设备）
        output_device (int or torch.device): device location of output (default: device_ids[0])# 输出设备位置（默认device_ids[0]）

    Attributes:
        module (Module): the module to be parallelized

    Example::

        >>> # xdoctest: +SKIP
        >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)  # input_var can be on any device, including CPU
    """

    # TODO: update notes/cuda.rst when this class handles 8+ GPUs well

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__()
        torch._C._log_api_usage_once("torch.nn.parallel.DataParallel")

        # 检查是否有可用的 GPU
        device_type = _get_available_device_type()
        if device_type is None:
            self.module = module
            self.device_ids = []    # GPU索引初始化
            return

        # 默认使用所有可用的 GPU（返回所有设备序号列表）
        if device_ids is None:
            device_ids = _get_all_device_indices()

        # 默认 server 是 device_ids 列表上第一个
        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module    # 待并行计算的模型
        self.device_ids = [_get_device_index(x, True) for x in device_ids]  # 获取所有设备索引（如果有）
        self.output_device = _get_device_index(output_device, True)     # 输出设备位置
        self.src_device_obj = torch.device(device_type, self.device_ids[0]) # 默认输出设备设为device_ids[0]

        # 检查负载是否平衡， 不平衡（指内存或者处理器 min/max < 0.75 会有警告）
        _check_balance(self.device_ids)

        # 使用单卡GPU训练
        if len(self.device_ids) == 1:
            self.module.to(self.src_device_obj)

    """
        前向传播：
            Scatter 函数将数据从 device[0] 分配并复制到不同的卡，
            Replicate 函数将模型从 device[0] 复制到不同的卡，
            各个卡都有了同样的模型和不同的数据，各个卡分别使用parallel_apply函数调用 forward 计算损失和梯度。
            
        后向传播：
            gather函数将各个卡的损失和梯度收集到设备device[0] 然后在设备device[0] 更新参数。
    """
    def forward(self, *inputs, **kwargs):
        with torch.autograd.profiler.record_function("DataParallel.forward"):
            # 没 GPU 可用self.device_ids=None
            if not self.device_ids: # 直接返回待并行化模型使用CPU计算
                return self.module(*inputs, **kwargs)

            # """
            #     运行GPU device_ids[0]（即server）之前，其必须有 parallelized module 的parameters 和 buffers
            #     因为 DP 保证 GPU device_ids[0] 和 base parallelized module 共享存储
            #     所以在device[0] 上的 in-place 更新也会被保留下来，其他的则不会
            # """

            for t in chain(self.module.parameters(), self.module.buffers()):
                # t.device返回模型数据和模型使用的设备类型
                if t.device != self.src_device_obj:
                    # 模块必须在设备self.src_device_obj(默认设备0)上有它的参数和缓冲区
                    raise RuntimeError("module must have its parameters and buffers "
                                       "on device {} (device_ids[0]) but found one of "
                                       "them on device: {}".format(self.src_device_obj, t.device))

            # """
            #     现在 device[0] 上已经有了 module 和 input（即parallelized module 的parameters 和 buffers），
            #     接下来就可以执行 Parameter Server 算法了。
            #     Parameter Server：GPU 0将数据分成五份分到各个卡上，每张卡负责自己的那一份mini-batch的训练，
            #     得到grad后，返回给GPU 0上做累积，得到更新的权重参数后，再分发给各个卡。
            # """

            # Scatter 函数将数据从 device[0] 分配并复制到不同的卡（将一个 batch 近似等分成更小的 batch分给不同的GPU）
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)

            if not inputs and not kwargs:
                """
                    # for forward function without any inputs, empty list and dict will be created
                    # 前向传播没有任何输入，将创建空list和空dict
                    # so the module can be executed on one device which is the first one in device_ids
                    # 因此，模块可以在一个设备上执行，这是device_ids中的第一个设备
                """
                inputs = ((),)
                kwargs = ({},)

            # 如果仅有单卡可用，直接单卡计算，不用并行
            if len(self.device_ids) == 1:
                return self.module(*inputs[0], **kwargs[0])

            # Replicate 函数将模型从 device[0] 复制到不同的卡
            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
            # 各个卡都有了同样的模型和不同的数据，parallel_apply函数让各个卡分别调用 forward 计算损失和梯度。
            outputs = self.parallel_apply(replicas, inputs, kwargs)
            # gather函数将各个卡的损失和梯度收集到设备device[0] 然后在设备device[0] 更新参数。（反向传播）
            return self.gather(outputs, self.output_device)

    # Replicate 函数将模型从 device[0] 复制到不同的卡
    def replicate(self, module, device_ids):
        return replicate(module, device_ids, not torch.is_grad_enabled())

     # Scatter 函数将数据从 device[0] 分配并复制到不同的卡
    def scatter(self, inputs, kwargs, device_ids):
        """
        scatter_kwargs 函数中最重要的就是 scatter 函数，
        负责将 tensor 分成大概相等的块并将他们分给不同的 GPU。
        将其他的数据类型，则是复制多份分给不同的GPU
        """
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    # 各个卡都有了同样的模型和不同的数据，parallel_apply函数让各个卡分别调用 forward 计算损失和梯度。
    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    # gather函数将各个卡的损失和梯度收集到设备device[0] 然后在设备device[0] 更新参数。（反向传播）
    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


def data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None):
    r"""Evaluates module(input) in parallel across the GPUs given in device_ids.

    This is the functional version of the DataParallel module.

    Args:
        module (Module): the module to evaluate in parallel
        inputs (Tensor): inputs to the module
        device_ids (list of int or torch.device): GPU ids on which to replicate module
        output_device (list of int or torch.device): GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Tensor containing the result of module(input) located on
        output_device
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,) if inputs is not None else ()

    device_type = _get_available_device_type()

    if device_ids is None:
        device_ids = _get_all_device_indices()

    if output_device is None:
        output_device = device_ids[0]

    device_ids = [_get_device_index(x, True) for x in device_ids]
    output_device = _get_device_index(output_device, True)
    src_device_obj = torch.device(device_type, device_ids[0])

    for t in chain(module.parameters(), module.buffers()):
        if t.device != src_device_obj:
            raise RuntimeError("module must have its parameters and buffers "
                               "on device {} (device_ids[0]) but found one of "
                               "them on device: {}".format(src_device_obj, t.device))

    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    # for module without any inputs, empty list and dict will be created
    # so the module can be executed on one device which is the first one in device_ids
    if not inputs and not module_kwargs:
        inputs = ((),)
        module_kwargs = ({},)

    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[:len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    return gather(outputs, output_device, dim)

