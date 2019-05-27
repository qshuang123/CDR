from tensorflow import shape, stack, reshape, concat, gather
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_math_ops


def batch_gather(params, indices, name=None):
  """Gather slices from `params` according to `indices` with leading batch dims.

  This operation assumes that the leading dimensions of `indices` are dense,
  and the gathers on the axis corresponding to the last dimension of `indices`.
  More concretely it computes:

  result[i1, ..., in] = params[i1, ..., in-1, indices[i1, ..., in]]

  Therefore `params` should be a Tensor of shape [A1, ..., AN, B1, ..., BM],
  `indices` should be a Tensor of shape [A1, ..., AN-1, C] and `result` will be
  a Tensor of size `[A1, ..., AN-1, C, B1, ..., BM]`.

  In the case in which indices is a 1D tensor, this operation is equivalent to
  `tf.gather`.

  See also `tf.gather` and `tf.gather_nd`.

  Args:
    params: A Tensor. The tensor from which to gather values.
    indices: A Tensor. Must be one of the following types: int32, int64. Index
        tensor. Must be in range `[0, params.shape[axis]`, where `axis` is the
        last dimension of `indices` itself.
    name: A name for the operation (optional).

  Returns:
    A Tensor. Has the same type as `params`.

  Raises:
    ValueError: if `indices` has an unknown shape.
  """

  with ops.name_scope(name):
    indices = ops.convert_to_tensor(indices, name="indices")
    params = ops.convert_to_tensor(params, name="params")
    indices_shape = shape(indices)
    params_shape = shape(params)
    ndims = indices.shape.ndims
    if ndims is None:
      raise ValueError("batch_gather does not allow indices with unknown "
                       "shape.")
    batch_indices = indices
    accum_dim_value = 1
    for dim in range(ndims-1, 0, -1):
      dim_value = params_shape[dim-1]
      accum_dim_value *= params_shape[dim]
      dim_indices = gen_math_ops._range(0, dim_value, 1)
      dim_indices *= accum_dim_value
      dim_shape = stack([1] * (dim - 1) + [dim_value] + [1] * (ndims - dim),
                        axis=0)
      batch_indices += reshape(dim_indices, dim_shape)

    flat_indices = reshape(batch_indices, [-1])
    outer_shape = params_shape[ndims:]
    flat_inner_shape = gen_math_ops.prod(
        params_shape[:ndims], [0], False)

    flat_params = reshape(
        params, concat([[flat_inner_shape], outer_shape], axis=0))
    flat_result = gather(flat_params, flat_indices)
    result = reshape(flat_result, concat([indices_shape, outer_shape], axis=0))
    final_shape = indices.get_shape()[:ndims-1].merge_with(
        params.get_shape()[:ndims -1])
    final_shape = final_shape.concatenate(indices.get_shape()[ndims-1])
    final_shape = final_shape.concatenate(params.get_shape()[ndims:])
    result.set_shape(final_shape)
    return result