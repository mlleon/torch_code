"""
Sampler类的主要职责就是提供访问dataset需要的index，可以通过Sampler类提供不同的index，从而控制数据加载的顺序。

为了提高模型的泛化性，避免过拟合，每个epoch的数据加载顺序都要打乱（shuffle参数功能），
DataLoader中配合shuffle参数自动构建一个顺序或者随机的采样器sampler，
或者，用户也可以使用sampler参数指定一个自定义Sampler对象，每次产生下一个要获取的index。

Sampler也是一个可迭代对象（iterable），PyTorch内置了以下几种Sampler：

    SequentialSampler（顺序采样）
    RandomSampler （随机采样）
    SubsetRandomSampler （索引随机采样）
    WeightedRandomSampler （加权随机采样）
    BatchSampler （批采样）

上面几种类型的Sampler都是继承自Sampler类，这是一个抽象类，规定Sampler的子类都必须提供__iter__方法，保证其必须是可迭代对象。
索引随机采样和加权随机采样都是随机采样的特殊化。

"""
import torch
from torch import Tensor

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

__all__ = [
    "BatchSampler",
    "RandomSampler",
    "Sampler",
    "SequentialSampler",
    "SubsetRandomSampler",
    "WeightedRandomSampler",
]

T_co = TypeVar('T_co', covariant=True)


class Sampler(Generic[T_co]):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source: Optional[Sized]) -> None:
        pass

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

    # NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    #
    # Many times we have an abstract class representing a collection/iterable of
    # data, e.g., `torch.utils.data.Sampler`, with its subclasses optionally
    # implementing a `__len__` method. In such cases, we must make sure to not
    # provide a default implementation, because both straightforward default
    # implementations have their issues:
    #
    #   + `return NotImplemented`:
    #     Calling `len(subclass_instance)` raises:
    #       TypeError: 'NotImplementedType' object cannot be interpreted as an integer
    #
    #   + `raise NotImplementedError()`:
    #     This prevents triggering some fallback behavior. E.g., the built-in
    #     `list(X)` tries to call `len(X)` first, and executes a different code
    #     path if the method is not found or `NotImplemented` is returned, while
    #     raising an `NotImplementedError` will propagate and and make the call
    #     fail where it could have use `__iter__` to complete the call.
    #
    # Thus, the only two sensible things to do are
    #
    #   + **not** provide a default `__len__`.
    #
    #   + raise a `TypeError` instead, which is what Python uses when users call
    #     a method that is not defined on an object.
    #     (@ssnl verifies that this works on at least Python 3.7.)


class SequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from（Dataset类型数据集）
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
                        若为True，则表示可以重复采样，即同一个样本可以重复采样，这样可能导致有的样本采样不到。
        num_samples (int): number of samples to draw, default=`len(dataset)`.指定采样的样本量，默认是所有
                        若replacement=True可以设置num_samples>len(dataset)来增加采样数量，使得每个样本都尽可能被采样到。
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime（数据集大小可能在运行时更改）
        if self._num_samples is None:
            return len(self.data_source)  # 未设定_num_samples，返回原数据集样本数
        return self._num_samples  # 设定_num_samples，返回设定的样本数量设定_num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)  # 获取数据集的样本数
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:  # 若为True，则表示可以重复采样，即同一个样本可以重复采样，这样可能导致有的样本采样不到
            # 如果replacement=True，迭代获取采样器样本时，先获取迭代对象1的样本，再获取迭代对象2的样本，两者总数为self.num_samples
            for _ in range(self.num_samples // 32):  # 随机采样um_samples // 32次，每次采样32个样本，并生成迭代对象1
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            # 随机采样num_samples % 32个样本，并生成迭代对象2
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64,
                                     generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples  # 返回采样的样本数量


class SubsetRandomSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        for i in torch.randperm(len(self.indices), generator=self.generator):
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)


class WeightedRandomSampler(Sampler[int]):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.
    """

    """
        weights(sequence)：所有样本（整个待采样数据集）中每个样本的权重。
            权重序列的长度可以等于数据集的长度。每个权重值则是你抽选该样本的可能性。这里的权重值大小并不需要加和为1。
            同一个类别样本的权重值应当都设为该类别占比的倒数。比如正样本占比20%，其权重应设为 5；对应的负样本的占比为 80 ％ ，其权重应设为=1.25。
        num_samples(int):采样数量，可以和数据集一致，也可以不一致。
        replacement (bool) ：是否有放回抽样，一般都要放回，不仅保证数据分布没变，还能让少的那一类被重复抽到，以保证数量与多数类平衡。
        generator (Generator) ：设定随机数生成方式
    """

    weights: Tensor
    num_samples: int
    replacement: bool

    def __init__(self, weights: Sequence[float], num_samples: int,
                 replacement: bool = True, generator=None) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples


class BatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        传入一个基采样器中：  SequentialSampler（顺序采样）
                            RandomSampler （随机采样）
                            SubsetRandomSampler （索引随机采样）
                            WeightedRandomSampler （加权随机采样）
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:  # 如果drop_last=True，丢弃最后不足一个batch样本数的batch
            # 这里因为没有对迭代器调用for循环，先使用iter()生成一个迭代器，然后使用next()返回样本值
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch  # 生成一个包含多个样本的batch的生成器索引
                except StopIteration:
                    break
        else:  # 如果drop_last=False，返回最后不足一个batch样本数的batch
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1  # 每个batch中样本数累计，最后一组可能不足一个batch长度的样本数
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0    # 满足一个batch长度后将batch数据清空
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:  # 最后一组可能不足一个batch长度样本数
                yield batch[:idx_in_batch]

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]

