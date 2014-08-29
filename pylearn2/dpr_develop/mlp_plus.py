__authors__ = "David Reichert"
__license__ = "3-clause BSD"

import functools
import numpy as np
import warnings
from collections import OrderedDict
import logging
import theano.tensor as T
from theano.sandbox import cuda

from pylearn2.models.mlp import Layer, BadInputSpaceError, MLP, max_pool, mean_pool, Sigmoid
from pylearn2.models.maxout import MaxoutConvC01B
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace
from pylearn2.models.model import Model

from pylearn2.linear import conv2d
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.linear.conv2d_c01b import setup_detector_layer_c01b
from pylearn2.linear import local_c01b
if cuda.cuda_available:
    from pylearn2.sandbox.cuda_convnet.pool import max_pool_c01b
from pylearn2.sandbox.cuda_convnet import check_cuda
from pylearn2.utils import wraps, safe_zip, safe_union, sharedX, py_integer_types, function
from pylearn2.utils.data_specs import is_flat_space


logger = logging.getLogger(__name__)


class FlexLayer(Layer):
    """
    An attempt to unify and generalize the various mlp layers. Intended usage is
    (all operations optional [apart from transform]):

    * take input state, living in self.input_space;
    * convert to self.desired_input_space as required by subsequent operations;
    * apply space-preserving operation (e.g. normalization) self.input_op;
    * apply transformation self.transform (e.g. linear, such as dense matrix
      mul or convolution, or nonlinear such as pooling). The transformation is in
      general not space-preserving, and the outcome lives in self.output_space.
    * apply space-preserving operation (e.g. normalization) self.pre_activation_op;
    * apply activation function self.activation_function (e.g. sigmoid).
    * apply space-preserving operation (e.g. normalization) self.post_activation_op.

    'transform': Instead of inherting from FlexLayer and overriding the transform
    method, the transformation can be passed as argument, for convenience. transform
    is the only op implemented as class method because the transformation is what
    usually requires most extra machinery, thus it makes sense to use inheritance
    [initially I just assigned a class method to self.transform, but that causes
    pickle problems.]
    """
    def __init__(self,
                 layer_name,
                 input_space=None,
                 desired_input_space=None,
                 input_op=None,
                 transform=None,
                 pre_activation_op=None,
                 activation_function=None,
                 post_activation_op=None):

        # Note that the layer abstract class does not have layer_name,
        # but all derived layers do and the mlp class assumes it's there, too,
        # so I'm adding it here.
        self.layer_name = layer_name

        super(FlexLayer, self).__init__()
        self.input_space = input_space
        self.desired_input_space = desired_input_space
        self.input_op = input_op
        if transform is not None:
            self.transform = transform
        self.pre_activation_op = pre_activation_op
        self.activation_function = activation_function
        self.post_activation_op = post_activation_op

    def transform(self, state_below):
        """
        Does the forward prop transformation for this layer.

        Parameters
        ----------
        state_below : member of self.desired_input_space
            A minibatch of states of the layer below, formatted to lie in
            a space compatible with the transformation.

        Returns
        -------
        state : member of self.output_space
            A minibatch of states of this layer, before any activation function
            or further operations are applied.
        """
        raise NotImplementedError(str(type(self))+" does not implement transform.")

    def set_input_space(self, space):
        """
        Tells the layer to prepare for input formatted according to the
        given space.

        Parameters
        ----------
        space : Space
            The Space the input to this layer will lie in.

        Notes
        -----
        This usually resets parameters.

        This layer should set input and output spaces, initialize transformation
        and ops, etc., as necessary (if __init__ hasn't done so already).
        """
        raise NotImplementedError(str(type(self)) + " does not implement "
                "set_input_space.")


    def fprop(self, state_below):
        self.input_space.validate(state_below)

        if not self.desired_input_space is None:
            state_below = self.input_space.format_as(
                state_below, self.desired_input_space)

        state = state_below

        for op in [self.input_op,
                   self.transform,
                   self.pre_activation_op,
                   self.activation_function,
                   self.post_activation_op]:

            if not op is None:
                state = op(state)

        self.output_space.validate(state)
        return state



class NoParamLayer(FlexLayer):
    """
    Layer that does not have parameters to be optimized (e.g. pooling).
    """
    def get_params(self):
        return []

    def get_l1_weight_decay(self):
        return 0.

    def get_weight_decay(self, coeff):
        return 0.

    # There's an interface mismatch: the model base class wants a data
    # argument but the mlp layers implement get_monitoring_channels without it.
    def get_monitoring_channels(self):
        return OrderedDict()






class ParallelLayer(FlexLayer):
    """
    Runs several layers in parallel, applying each to a different component
    of a composite input space (adapted from mlp.CompositeLayer; the
    latter applies the layers all to the same input).

    [weight decay? LR scalers?]

    """

    def __init__(self, layer_name, layers, **kwargs):
        """
        .. todo::

            WRITEME properly

        layers: a list or tuple of Layers.
        """
        self.layers = layers
        super(ParallelLayer, self).__init__(layer_name, **kwargs)

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        logger.info('Layer %s', self.layer_name)
        if not isinstance(space, CompositeSpace):
            raise BadInputSpaceError("ParallelLayer input space needs to be a CompositeSpace.")

        self.input_space = space
        for layer, component_space in safe_zip(self.layers, space.components):
            layer.set_input_space(component_space)

        self.output_space = CompositeSpace(tuple(layer.get_output_space()
            for layer in self.layers))

    @functools.wraps(FlexLayer.transform)
    def transform(self, state_below):
        return tuple(layer.fprop(component_state) for
                     layer, component_state in safe_zip(self.layers, state_below))

    @wraps(Layer.get_params)
    def get_params(self):

        rval = []

        for layer in self.layers:
            rval = safe_union(layer.get_params(), rval)

        return rval

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):

        return sum(layer.cost(Y_elem, Y_hat_elem) for layer, Y_elem, Y_hat_elem in \
                safe_zip(self.layers, Y, Y_hat))

    @wraps(Layer.set_mlp)
    def set_mlp(self, mlp):

        super(ParallelLayer, self).set_mlp(mlp)
        for layer in self.layers:
            layer.set_mlp(mlp)


    def get_weight_decay(self, coeff):
        """
        Uses the same weight decay coefficient for all component layers.
        """

        decay = 0.

        for layer in self.layers:
            # The mlp class should be made more general to have a get_weight_decay method.
            if isinstance(layer, MLP):
                for mlp_layer in layer.layers:
                    decay = decay + mlp_layer.get_weight_decay(coeff)
            else:
                decay = decay + layer.get_weight_decay(coeff)

        return decay

    def get_monitoring_channels(self):
        channels = self.layers[0].get_monitoring_channels()
        for layer in self.layers[1:]:
            channels.update(layer.get_monitoring_channels())
        return channels


    @wraps(Layer.censor_updates)
    def censor_updates(self, updates):
        for layer in self.layers:
            layer.censor_updates(updates)



class ElemSumLayer(NoParamLayer):
    """
    Adds the inputs coming from a composite space elementwise. Has no parameters.
    """

    def set_input_space(self, space):
        logger.info('Layer %s', self.layer_name)
        if not isinstance(space, CompositeSpace) or not is_flat_space(space):
            raise BadInputSpaceError("ElemSumLayer input space needs to be flat CompositeSpace.")

        first_component = space.components[0]
        for component in space.components:
            if not first_component == component:
                # This requirement could be loosened by formatting the other components
                # to the first component space (for example), if possible.
                raise BadInputSpaceError("The component spaces of the ElemSumLayer input space"
                                         " need to be equal.")

        self.input_space = space
        self.output_space = first_component

    @functools.wraps(FlexLayer.transform)
    def transform(self, state_below):
        out = state_below[0]

        for state in state_below[1:]:
            out = out + state
        return out


class SplitChannelsLayer(NoParamLayer):
    """
    Split channels of a Conv2D input space into separate
    Conv2DSpaces in a CompositeSpace.

    num_splits: Number of output components. If None (default),
    split each channel, i.e. num_splits = num_channels of input space.
    """
    def __init__(self,
                 layer_name,
                 num_splits=None,
                 **kwargs):
        super(SplitChannelsLayer, self).__init__(layer_name, **kwargs)
        self.num_splits = num_splits


    def set_input_space(self, space):
        logger.info('Layer %s', self.layer_name)

        if not isinstance(space, Conv2DSpace):
            raise BadInputSpaceError("SplitChannelLayer.set_input_space "
                                     "expected a Conv2DSpace, got " +
                                     str(space) + " of type " +
                                     str(type(space)))
        if self.num_splits is None:
            self.num_splits = space.num_channels
        else:
            if self.num_splits > space.num_channels:
                raise ValueError('self.num_splits > input_space.num_channels.')
            if space.num_channels % self.num_splits != 0:
                warnings.warn('Number of channels not divisible by number of splits;'
                              ' last split will have extra channels.')

        self.input_space = space
        self.desired_input_space = space

        output_components = [Conv2DSpace(
            space.shape, axes=space.axes, num_channels=space.num_channels // self.num_splits)
                             for i_split in range(self.num_splits - 1)]
        output_components.append(
            Conv2DSpace(
                space.shape, axes=space.axes,
                num_channels=space.num_channels // self.num_splits +
                space.num_channels % self.num_splits))
        channels_covered = 0
        for component in output_components:
            channels_covered += component.num_channels
        assert channels_covered == space.num_channels

        self.output_space = CompositeSpace(output_components)



    @functools.wraps(FlexLayer.transform)
    def transform(self, state_below):
        channel_axis = self.input_space.axes.index('c')
        splits_size = [component.num_channels for
                       component in self.output_space.components]
        return tuple(T.split(state_below, splits_size, self.num_splits, axis=channel_axis))



class PairwiseMultLayer(NoParamLayer):
    """
    Multiply adjacent channels in a Conv2D input space (non-overlapping pairs).

    """
    def set_input_space(self, space):
        logger.info('Layer %s', self.layer_name)

        if not isinstance(space, Conv2DSpace):
            raise BadInputSpaceError("PairwiseMultLayer.set_input_space "
                                     "expected a Conv2DSpace, got " +
                                     str(space) + " of type " +
                                     str(type(space)))
        if space.num_channels % 2 != 0:
            raise BadInputSpaceError('Number of channels not divisible by two.')

        self.input_space = space
        self.desired_input_space = space
        self.output_space = Conv2DSpace(
            space.shape, axes=space.axes, num_channels=space.num_channels / 2)

    @functools.wraps(FlexLayer.transform)
    def transform(self, state_below):
        channel_axis = self.input_space.axes.index('c')
        #row_axis = self.input_space.axes.index(0)
        #num_rows = self.input_space.shape[0]
        #num_splits = self.input_space.num_channels / 2
        #splits_size = [2 for _ in range(num_splits)]
        #splits = T.split(state_below, splits_size, num_splits, axis=channel_axis)
        #tall_state = T.concatenate(splits, row_axis)
        #tall_state_multiplied = tall_state.prod(axis=channel_axis, keepdims=True)
        ## This should be a reshape (would need to ensure correct ordering)
        #splits_size = [num_rows for _ in range(num_splits)]
        #out = T.split(tall_state_multiplied, splits_size, num_splits, axis=row_axis)
        #out = T.concatenate(out, channel_axis)
        #return out
        slices_even = [slice(None), slice(None), slice(None), slice(None)]
        slices_odd = [slice(None), slice(None), slice(None), slice(None)]
        slices_even[channel_axis] = slice(None, None, 2)
        slices_odd[channel_axis] = slice(1, None, 2)
        # Theano doesn't accept state_below[slices] like numpy?
        return state_below[slices_even[0], slices_even[1], slices_even[2], slices_even[3]] *\
               state_below[slices_odd[0], slices_odd[1], slices_odd[2], slices_odd[3]]




class DenseLinear(FlexLayer):
    """"
    Flex version of the standard Linear layer [todo: just copy docu?].
    """
    def __init__(self,
                 dim,
                 layer_name,
                 irange=None,
                 istdev=None,
                 sparse_init=None,
                 sparse_stdev=1.,
                 include_prob=1.0,
                 init_bias=0.,
                 W_lr_scale=None,
                 b_lr_scale=None,
                 mask_weights=None,
                 max_row_norm=None,
                 max_col_norm=None,
                 min_col_norm=None,
                 softmax_columns=None,
                 copy_input=None,
                 use_abs_loss=False,
                 use_bias=True,
                 **kwargs):

        super(DenseLinear, self).__init__(layer_name, **kwargs)
        del kwargs
        del layer_name

        if use_bias and init_bias is None:
            init_bias = 0.

        self.__dict__.update(locals())
        del self.self

        if use_bias:
            self.b = sharedX(np.zeros((self.dim,)) + init_bias,
                             name=(self.layer_name + '_b'))
        else:
            assert b_lr_scale is None
            init_bias is None


    @wraps(Layer.get_lr_scalers)
    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        logger.info('Layer %s', self.layer_name)
        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        logger.info('Input dim: %s', self.input_dim)

        self.output_space = VectorSpace(self.dim)

        logger.info('Ouput dim: %s', self.output_space.dim)

        rng = self.mlp.rng
        if self.irange is not None:
            assert self.istdev is None
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,
                            self.irange,
                            (self.input_dim, self.dim)) * \
                (rng.uniform(0., 1., (self.input_dim, self.dim))
                 < self.include_prob)
        elif self.istdev is not None:
            assert self.sparse_init is None
            W = rng.randn(self.input_dim, self.dim) * self.istdev
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.input_dim, self.dim))

            def mask_rejects(idx, i):
                if self.mask_weights is None:
                    return False
                return self.mask_weights[idx, i] == 0.

            for i in xrange(self.dim):
                assert self.sparse_init <= self.input_dim
                for j in xrange(self.sparse_init):
                    idx = rng.randint(0, self.input_dim)
                    while W[idx, i] != 0 or mask_rejects(idx, i):
                        idx = rng.randint(0, self.input_dim)
                    W[idx, i] = rng.randn()
            W *= self.sparse_stdev

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W, = self.transformer.get_params()
        assert W.name is not None

        if self.mask_weights is not None:
            expected_shape = (self.input_dim, self.dim)
            if expected_shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape " +
                                 str(expected_shape)+" but got " +
                                 str(self.mask_weights.shape))
            self.mask = sharedX(self.mask_weights)

    @wraps(Layer.censor_updates)
    def censor_updates(self, updates):

        if self.mask_weights is not None:
            W, = self.transformer.get_params()
            if W in updates:
                updates[W] = updates[W] * self.mask

        if self.max_row_norm is not None:
            W, = self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=1))
                desired_norms = T.clip(row_norms, 0, self.max_row_norm)
                scales = desired_norms / (1e-7 + row_norms)
                updates[W] = updated_W * scales.dimshuffle(0, 'x')

        if self.max_col_norm is not None or self.min_col_norm is not None:
            assert self.max_row_norm is None
            if self.max_col_norm is not None:
                max_col_norm = self.max_col_norm
            if self.min_col_norm is None:
                self.min_col_norm = 0
            W, = self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                if self.max_col_norm is None:
                    max_col_norm = col_norms.max()
                desired_norms = T.clip(col_norms,
                                       self.min_col_norm,
                                       max_col_norm)
                updates[W] = updated_W * desired_norms / (1e-7 + col_norms)

    @wraps(Layer.get_params)
    def get_params(self):

        W, = self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        if self.use_bias:
            assert self.b.name is not None
            assert self.b not in rval
            rval.append(self.b)
        return rval

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * abs(W).sum()

    @wraps(Layer.get_weights)
    def get_weights(self):

        if self.requires_reformat:
            # This is not really an unimplemented case.
            # We actually don't know how to format the weights
            # in design space. We got the data in topo space
            # and we don't have access to the dataset
            raise NotImplementedError()
        W, = self.transformer.get_params()

        W = W.get_value()

        if self.softmax_columns:
            P = np.exp(W)
            Z = np.exp(W).sum(axis=0)
            rval = P / Z
            return rval
        return W

    @wraps(Layer.set_weights)
    def set_weights(self, weights):

        W, = self.transformer.get_params()
        W.set_value(weights)

    @wraps(Layer.set_biases)
    def set_biases(self, biases):

        self.b.set_value(biases)

    @wraps(Layer.get_biases)
    def get_biases(self):
        """
        .. todo::

            WRITEME
        """
        return self.b.get_value()

    @wraps(Layer.get_weights_format)
    def get_weights_format(self):

        return ('v', 'h')

    @wraps(Layer.get_weights_topo)
    def get_weights_topo(self):

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()

        W, = self.transformer.get_params()

        W = W.T

        W = W.reshape((self.dim, self.input_space.shape[0],
                       self.input_space.shape[1],
                       self.input_space.num_channels))

        W = Conv2DSpace.convert(W, self.input_space.axes, ('b', 0, 1, 'c'))

        return function([], W)()

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):

        W, = self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([('row_norms_min',  row_norms.min()),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max()), ])

    @wraps(Layer.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state, target=None):

        rval = OrderedDict()

        mx = state.max(axis=0)
        mean = state.mean(axis=0)
        mn = state.min(axis=0)
        rg = mx - mn

        rval['range_x_max_u'] = rg.max()
        rval['range_x_mean_u'] = rg.mean()
        rval['range_x_min_u'] = rg.min()

        rval['max_x_max_u'] = mx.max()
        rval['max_x_mean_u'] = mx.mean()
        rval['max_x_min_u'] = mx.min()

        rval['mean_x_max_u'] = mean.max()
        rval['mean_x_mean_u'] = mean.mean()
        rval['mean_x_min_u'] = mean.min()

        rval['min_x_max_u'] = mn.max()
        rval['min_x_mean_u'] = mn.mean()
        rval['min_x_min_u'] = mn.min()

        return rval


    # I think the cost should be factored out as well. Not sure how.

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):

        return self.cost_from_cost_matrix(self.cost_matrix(Y, Y_hat))

    @wraps(Layer.cost_from_cost_matrix)
    def cost_from_cost_matrix(self, cost_matrix):

        return cost_matrix.sum(axis=1).mean()

    @wraps(Layer.cost_matrix)
    def cost_matrix(self, Y, Y_hat):

        if(self.use_abs_loss):
            return T.abs_(Y - Y_hat)
        else:
            return T.sqr(Y - Y_hat)

    @functools.wraps(FlexLayer.transform)
    def transform(self, state_below):
        """
        Parameters
        ----------
        state_below : member of input_space

        Returns
        -------
        output : theano matrix
            Affine transformation of state_below
        """
        self.input_space.validate(state_below)

        if self.requires_reformat:
            state_below = self.input_space.format_as(state_below,
                                                     self.desired_space)

        # Support old pickle files
        if not hasattr(self, 'softmax_columns'):
            self.softmax_columns = False

        if self.softmax_columns:
            W, = self.transformer.get_params()
            W = W.T
            W = T.nnet.softmax(W)
            W = W.T
            z = T.dot(state_below, W)
            if self.use_bias:
                z += self.b
        else:
            z = self.transformer.lmul(state_below)
            if self.use_bias:
                z += self.b

        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        return z




class ConvLinearBC01(FlexLayer):
    """
    Implements the convolution transformation from the standard ConvRectifiedLinear
    layer but without the activation function (rectification) and pooling. Based on
    theano's BC01 formatted convolution. [TODO: ConvRectifiedLinear talks about
    theano using B01C but when I digg into the code it looks like BC01! I need
    to check with the list...]

    There as a lot of redundancy between this and the DenseLinear layer. It should
    be possible to inherit everything from a base Linear/Affine layer, with the
    inheriting classes assigning a transformer and overwriting methods as necessary.
    I'm not doing that now because there's a couple of differences I'm not sure
    how to best deal with / why they are there (flag use_biases, get_weights missing in
    one case, softmax_columns, equivalent vars having different names (output_channls
    vs num_channels), etc. It would be much more efficient if the people
    who wrote the other layers were to do this...
    """

    def __init__(self,
                 layer_name,
                 output_channels,
                 kernel_shape,
                 irange=None,
                 border_mode='valid',
                 sparse_init=None,
                 include_prob=1.0,
                 init_bias=0.,
                 W_lr_scale=None,
                 b_lr_scale=None,
                 max_kernel_norm=None,
                 kernel_stride=(1, 1),
                 **kwargs):

        super(ConvLinearBC01, self).__init__(layer_name, **kwargs)
        del kwargs
        del layer_name

        if (irange is None) and (sparse_init is None):
            raise AssertionError("You should specify either irange or "
                                 "sparse_init when calling the constructor of "
                                 "ConvRectifiedLinear.")
        elif (irange is not None) and (sparse_init is not None):
            raise AssertionError("You should specify either irange or "
                                 "sparse_init when calling the constructor of "
                                 "ConvRectifiedLinear and not both.")

        self.__dict__.update(locals())
        del self.self


    @wraps(Layer.get_lr_scalers)
    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        logger.info('Layer %s', self.layer_name)
        self.input_space = space

        if not isinstance(space, Conv2DSpace):
            raise BadInputSpaceError("ConvRectifiedLinear.set_input_space "
                                     "expected a Conv2DSpace, got " +
                                     str(space) + " of type " +
                                     str(type(space)))

        # DPR: The convolution transformer itself handles axes ordering but
        # I'm dealing with it here for consistency with other layers.
        self.desired_input_space = Conv2DSpace(
            space.shape, num_channels=space.num_channels, axes=('b', 'c', 0, 1))


        rng = self.mlp.rng

        if self.border_mode == 'valid':
            output_shape = [(self.desired_input_space.shape[0]-self.kernel_shape[0]) /
                            self.kernel_stride[0] + 1,
                            (self.desired_input_space.shape[1]-self.kernel_shape[1]) /
                            self.kernel_stride[1] + 1]
        elif self.border_mode == 'full':
            output_shape = [(self.desired_input_space.shape[0]+self.kernel_shape[0]) /
                            self.kernel_stride[0] - 1,
                            (self.desired_input_space.shape[1]+self.kernel_shape[1]) /
                            self.kernel_stride[1] - 1]

        self.output_space = Conv2DSpace(shape=output_shape,
                                          num_channels=self.output_channels,
                                          axes=('b', 'c', 0, 1))

        if self.irange is not None:
            assert self.sparse_init is None
            self.transformer = conv2d.make_random_conv2D(
                irange=self.irange,
                input_space=self.desired_input_space,
                output_space=self.output_space,
                kernel_shape=self.kernel_shape,
                batch_size=self.mlp.batch_size,
                subsample=self.kernel_stride,
                border_mode=self.border_mode,
                rng=rng)
        elif self.sparse_init is not None:
            self.transformer = conv2d.make_sparse_random_conv2D(
                num_nonzero=self.sparse_init,
                input_space=self.desired_input_space,
                output_space=self.output_space,
                kernel_shape=self.kernel_shape,
                batch_size=self.mlp.batch_size,
                subsample=self.kernel_stride,
                border_mode=self.border_mode,
                rng=rng)
        W, = self.transformer.get_params()
        W.name = 'W'

        self.b = sharedX(self.output_space.get_origin() + self.init_bias)
        self.b.name = 'b'

        logger.info('Input shape: %s', self.input_space.shape)
        logger.info('Output space: %s', self.output_space.shape)

    @wraps(Layer.censor_updates)
    def censor_updates(self, updates):
        """
        .. todo::

            WRITEME
        """

        if self.max_kernel_norm is not None:
            W, = self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=(1, 2, 3)))
                desired_norms = T.clip(row_norms, 0, self.max_kernel_norm)
                scales = desired_norms / (1e-7 + row_norms)
                updates[W] = updated_W * scales.dimshuffle(0, 'x', 'x', 'x')

    @wraps(Layer.get_params)
    def get_params(self):
        """
        .. todo::

            WRITEME
        """
        assert self.b.name is not None
        W, = self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * abs(W).sum()

    @wraps(Layer.set_weights)
    def set_weights(self, weights):

        W, = self.transformer.get_params()
        W.set_value(weights)

    @wraps(Layer.set_biases)
    def set_biases(self, biases):

        self.b.set_value(biases)

    @wraps(Layer.get_biases)
    def get_biases(self):

        return self.b.get_value()

    @wraps(Layer.get_weights_format)
    def get_weights_format(self):

        return ('v', 'h')

    @wraps(Layer.get_weights_topo)
    def get_weights_topo(self):

        outp, inp, rows, cols = range(4)
        raw = self.transformer._filters.get_value()

        return np.transpose(raw, (outp, rows, cols, inp))

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):

        W, = self.transformer.get_params()

        assert W.ndim == 4

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=(1, 2, 3)))

        return OrderedDict([('kernel_norms_min',  row_norms.min()),
                            ('kernel_norms_mean', row_norms.mean()),
                            ('kernel_norms_max',  row_norms.max()), ])


    @functools.wraps(Layer.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state):
        # The standard ConvRectifiedLinear does not implement this but the other
        # layers do, so I'm including it.
        # Note that this are prepooling states, the standard mlp layers report
        # post pooling states.

        P = state

        rval = OrderedDict()

        vars_and_prefixes = [(P, '')]

        for var, prefix in vars_and_prefixes:
            assert var.ndim == 4
            v_max = var.max(axis=(1, 2, 3))
            v_min = var.min(axis=(1, 2, 3))
            v_mean = var.mean(axis=(1, 2, 3))
            v_range = v_max - v_min

            # max_x.mean_u is "the mean over *u*nits of the max over
            # e*x*amples" The x and u are included in the name because
            # otherwise its hard to remember which axis is which when reading
            # the monitor I use inner.outer rather than outer_of_inner or
            # something like that because I want mean_x.* to appear next to
            # each other in the alphabetical list, as these are commonly
            # plotted together
            for key, val in [('max_x.max_u',    v_max.max()),
                             ('max_x.mean_u',   v_max.mean()),
                             ('max_x.min_u',    v_max.min()),
                             ('min_x.max_u',    v_min.max()),
                             ('min_x.mean_u',   v_min.mean()),
                             ('min_x.min_u',    v_min.min()),
                             ('range_x.max_u',  v_range.max()),
                             ('range_x.mean_u', v_range.mean()),
                             ('range_x.min_u',  v_range.min()),
                             ('mean_x.max_u',   v_mean.max()),
                             ('mean_x.mean_u',  v_mean.mean()),
                             ('mean_x.min_u',   v_mean.min())]:
                rval[prefix+key] = val

        return rval


    @functools.wraps(FlexLayer.transform)
    def transform(self, state_below):
        # This is very similar to the dense layer linear part (apart from extra stuff
        # like optional biases or 'softmax columns') and could be further unified.

        z = self.transformer.lmul(state_below) + self.b
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        return z




class ConvLinearC01B(FlexLayer):
    """
    Adapted from MaxoutConvC01B to get the convolution transformation
    implemented by Alex Krizhevsky. Again lots of redundancy with
    the other linear layers.

    """
    def __init__(self,
                 layer_name,
                 num_channels,
                 kernel_shape,
                 irange=None,
                 init_bias=0.,
                 W_lr_scale=None,
                 b_lr_scale=None,
                 pad=0,
                 fix_kernel_shape=False,
                 partial_sum=1,
                 tied_b=False,
                 max_kernel_norm=None,
                 kernel_stride=(1, 1),
                 **kwargs):

        super(ConvLinearC01B, self).__init__(layer_name, **kwargs)
        del kwargs
        del layer_name

        detector_channels = num_channels

        check_cuda(str(type(self)))

        self.__dict__.update(locals())
        del self.self



    @functools.wraps(Model.get_lr_scalers)
    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    def set_input_space(self, space):
        """
        Tells the layer to use the specified input space.

        This resets parameters! The kernel tensor is initialized with the
        size needed to receive input from this space.

        Parameters
        ----------
        space : Space
            The Space that the input will lie in.
        """
        logger.info('Layer %s', self.layer_name)

        desired_input_space = Conv2DSpace(
            shape=space.shape,
            channels=space.num_channels,
            axes=('c', 0, 1, 'b'))

        # This overwrites the input space.
        setup_detector_layer_c01b(layer=self,
                                  input_space=desired_input_space,
                                  rng=self.mlp.rng)

        self.input_space = space
        self.desired_input_space = desired_input_space

        self.output_space = self.detector_space

        logger.info('Output space: %s', self.output_space.shape)


    def censor_updates(self, updates):
        """
        Replaces the values in `updates` if needed to enforce the options set
        in the __init__ method, including `max_kernel_norm`.

        Parameters
        ----------
        updates : OrderedDict
            A dictionary mapping parameters (including parameters not
            belonging to this model) to updated values of those parameters.
            The dictionary passed in contains the updates proposed by the
            learning algorithm. This function modifies the dictionary
            directly. The modified version will be compiled and executed
            by the learning algorithm.
        """

        if self.max_kernel_norm is not None:
            W, = self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=(0, 1, 2)))
                desired_norms = T.clip(row_norms, 0, self.max_kernel_norm)
                scales = desired_norms / (1e-7 + row_norms)
                updates[W] = (updated_W * scales.dimshuffle('x', 'x', 'x', 0))

    @functools.wraps(Model.get_params)
    def get_params(self):
        assert self.b.name is not None
        W, = self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

    @functools.wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    @functools.wraps(Layer.set_weights)
    def set_weights(self, weights):
        W, = self.transformer.get_params()
        W.set_value(weights)

    @functools.wraps(Layer.set_biases)
    def set_biases(self, biases):
        self.b.set_value(biases)

    @functools.wraps(Layer.get_biases)
    def get_biases(self):
        return self.b.get_value()

    @functools.wraps(Model.get_weights_topo)
    def get_weights_topo(self):
        return self.transformer.get_weights_topo()

    @functools.wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):

        W, = self.transformer.get_params()

        assert W.ndim == 4

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=(0, 1, 2)))

        return OrderedDict([('kernel_norms_min',  row_norms.min()),
                            ('kernel_norms_mean', row_norms.mean()),
                            ('kernel_norms_max',  row_norms.max()), ])

    @functools.wraps(FlexLayer.transform)
    def transform(self, state_below):
        check_cuda(str(type(self)))

        # Alex's code requires # input channels to be <= 3 or a multiple of 4
        # so we add dummy channels if necessary
        if not hasattr(self, 'dummy_channels'):
            self.dummy_channels = 0
        if self.dummy_channels > 0:
            zeros = T.zeros_like(state_below[0:self.dummy_channels, :, :, :])
            state_below = T.concatenate((state_below, zeros), axis=0)

        z = self.transformer.lmul(state_below)
        if not hasattr(self, 'tied_b'):
            self.tied_b = False
        if self.tied_b:
            b = self.b.dimshuffle(0, 'x', 'x', 'x')
        else:
            b = self.b.dimshuffle(0, 1, 2, 'x')

        z = z + b
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        #DPR: ?
        assert self.detector_space.num_channels % 16 == 0

        return z

    @functools.wraps(Model.get_weights_view_shape)
    def get_weights_view_shape(self):
        # Let the PatchViewer decide how to arrange the units
        # when they're not pooled
        raise NotImplementedError()

    @functools.wraps(Layer.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state):
        # Note that this are prepooling states, the standard mlp layers report
        # post pooling states.

        P = state

        rval = OrderedDict()

        vars_and_prefixes = [(P, '')]

        for var, prefix in vars_and_prefixes:
            assert var.ndim == 4
            v_max = var.max(axis=(1, 2, 3))
            v_min = var.min(axis=(1, 2, 3))
            v_mean = var.mean(axis=(1, 2, 3))
            v_range = v_max - v_min

            # max_x.mean_u is "the mean over *u*nits of the max over
            # e*x*amples" The x and u are included in the name because
            # otherwise its hard to remember which axis is which when reading
            # the monitor I use inner.outer rather than outer_of_inner or
            # something like that because I want mean_x.* to appear next to
            # each other in the alphabetical list, as these are commonly
            # plotted together
            for key, val in [('max_x.max_u',    v_max.max()),
                             ('max_x.mean_u',   v_max.mean()),
                             ('max_x.min_u',    v_max.min()),
                             ('min_x.max_u',    v_min.max()),
                             ('min_x.mean_u',   v_min.mean()),
                             ('min_x.min_u',    v_min.min()),
                             ('range_x.max_u',  v_range.max()),
                             ('range_x.mean_u', v_range.mean()),
                             ('range_x.min_u',  v_range.min()),
                             ('mean_x.max_u',   v_mean.max()),
                             ('mean_x.mean_u',  v_mean.mean()),
                             ('mean_x.min_u',   v_mean.min())]:
                rval[prefix+key] = val

        return rval




class PoolBC10(NoParamLayer):
    def __init__(self,
                 pool_shape,
                 pool_stride,
                 layer_name,
                 pool_type='max',
                 **kwargs):
        super(PoolBC10, self).__init__(layer_name, **kwargs)
        self.pool_shape = pool_shape
        self.pool_stride = pool_stride
        self.layer_name = layer_name
        self.pool_type = pool_type


    def set_input_space(self, space):
        logger.info('Layer %s', self.layer_name)
        self.input_space = space

        if not isinstance(space, Conv2DSpace):
            raise BadInputSpaceError("PoolBC10.set_input_space "
                                     "expected a Conv2DSpace, got " +
                                     str(space) + " of type " +
                                     str(type(space)))

        self.desired_input_space = Conv2DSpace(
            space.shape, num_channels=space.num_channels, axes=('b', 'c', 0, 1))

        logger.info('Input shape: %s', self.input_space.shape)

        assert self.pool_type in ['max', 'mean']

        dummy_batch_size = self.mlp.batch_size
        if dummy_batch_size is None:
            dummy_batch_size = 2
        dummy_input = sharedX(
            self.desired_input_space.get_origin_batch(dummy_batch_size))
        dummy_p = self.transform(dummy_input)
        dummy_p = dummy_p.eval()

        self.output_space = Conv2DSpace(shape=[dummy_p.shape[2],
                                               dummy_p.shape[3]],
                                        num_channels=self.desired_input_space.num_channels,
                                        axes=('b', 'c', 0, 1))

        logger.info('Output shape: %s', self.output_space.shape)

    def transform(self, bc01):
        if self.pool_type == 'max':
            return max_pool(bc01=bc01,
                            pool_shape=self.pool_shape,
                            pool_stride=self.pool_stride,
                            image_shape=self.desired_input_space.shape)
        elif self.pool_type == 'mean':
            return mean_pool(bc01=bc01,
                             pool_shape=self.pool_shape,
                             pool_stride=self.pool_stride,
                             image_shape=self.desired_input_space.shape)


class MaxPoolC01B(NoParamLayer):
    """
    Max pool layer based on the Krizhevsky implementation.
    """
    def __init__(self,
                 pool_shape,
                 pool_stride,
                 layer_name,
                 fix_pool_shape=False,
                 fix_pool_stride=False,
                 **kwargs):
        super(MaxPoolC01B, self).__init__(layer_name, **kwargs)
        self.pool_shape = pool_shape
        self.pool_stride = pool_stride
        self.fix_pool_shape = fix_pool_shape
        self.fix_pool_stride = fix_pool_stride


    def set_input_space(self, space):

        self.input_space = space

        if not isinstance(space, Conv2DSpace):
            raise BadInputSpaceError("MaxPoolC01B.set_input_space "
                                     "expected a Conv2DSpace, got " +
                                     str(space) + " of type " +
                                     str(type(space)))


        self.desired_input_space = Conv2DSpace(
            space.shape, num_channels=space.num_channels, axes=('c', 0, 1, 'b'))


        # Make sure number of channels is supported by cuda-convnet
        # (multiple of 4 or <= 3)
        # If not supported, pad the input with dummy channels
        ch = self.input_space.num_channels
        rem = ch % 4
        if ch > 3 and rem != 0:
            self.dummy_channels = 4 - rem
        else:
            self.dummy_channels = 0
        self.dummy_space = Conv2DSpace(shape=self.input_space.shape,
                                       channels=self.input_space.num_channels
                                       + self.dummy_channels,
                                       axes=('c', 0, 1, 'b'))

        input_shape = self.input_space.shape
        logger.info('Layer %s', self.layer_name)
        logger.info('Input space: %s', input_shape)

        def handle_pool_shape(idx):
            if self.pool_shape[idx] < 1:
                raise ValueError("bad pool shape: " + str(self.pool_shape))
            if self.pool_shape[idx] > input_shape[idx]:
                if self.fix_pool_shape:
                    assert input_shape[idx] > 0
                    self.pool_shape[idx] = input_shape[idx]
                else:
                    raise ValueError("Pool shape exceeds detector layer shape "
                                     "on axis %d" % idx)

        map(handle_pool_shape, [0, 1])

        assert self.pool_shape[0] == self.pool_shape[1]
        assert self.pool_stride[0] == self.pool_stride[1]
        assert all(isinstance(elem, py_integer_types)
                   for elem in self.pool_stride)
        if self.pool_stride[0] > self.pool_shape[0]:
            if self.fix_pool_stride:
                warnings.warn("Fixing the pool stride")
                ps = self.pool_shape[0]
                assert isinstance(ps, py_integer_types)
                self.pool_stride = [ps, ps]
            else:
                raise ValueError("Stride too big.")
        assert all(isinstance(elem, py_integer_types)
                   for elem in self.pool_stride)

        dummy_input = sharedX(
            self.dummy_space.get_origin_batch(2)[0:16, :, :, :])

        dummy_p = self.transform(dummy_input)
        dummy_p = dummy_p.eval()
        self.output_space = Conv2DSpace(
            shape=[dummy_p.shape[1],
                   dummy_p.shape[2]],
            num_channels=self.desired_input_space.num_channels,
            axes=('c', 0, 1, 'b'))

        logger.info('Output space: %s', self.output_space.shape)

    def transform(self, c01b):
        if self.dummy_channels > 0:
            zeros = T.zeros_like(c01b[0:self.dummy_channels, :, :, :])
            c01b = T.concatenate((c01b, zeros), axis=0)
        return max_pool_c01b(c01b=c01b,
                             pool_shape=self.pool_shape,
                             pool_stride=self.pool_stride,
                             image_shape=self.desired_input_space.shape)



class MinNormConvLinearC01B(ConvLinearC01B):
    """
    Like ConvLinearC01B but also provides min_kernel_norm.
    """
    def __init__(self, *args, **kwargs):
        if "min_kernel_norm" in kwargs:
            self.min_kernel_norm = kwargs.pop("min_kernel_norm")

        else:
            self.min_kernel_norm = None

        super(MinNormConvLinearC01B, self).__init__(*args, **kwargs)


    def censor_updates(self, updates):
        """
        .. todo::

            WRITEME
        """

        if self.max_kernel_norm is not None or self.min_kernel_norm is not None:

            W ,= self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=(0,1,2)))
                if self.max_kernel_norm is None:
                    max_kernel_norm = row_norms.max()
                else:
                    max_kernel_norm = self.max_kernel_norm
                if self.min_kernel_norm is None:
                    min_kernel_norm = 0
                else:
                    min_kernel_norm = self.min_kernel_norm

                desired_norms = T.clip(row_norms, min_kernel_norm,
                                       max_kernel_norm)
                updates[W] = updated_W * (
                    desired_norms / (1e-7 + row_norms)).dimshuffle('x', 'x', 'x', 0)



class ConvSigmoidC01B(ConvLinearC01B):
    """
    Convolutional layer with sigmoid activation function and the costs/monitoring
    channels of the mlp Sigmoid layer (for the 'detection' monitoring style).

    [I think the cost should perhaps be factored out from the layers. Then I wouldn't
    have to write a new class for this (but refactoring might take some work; for example,
    the dropout cost assumes that the underlying cost is provided by the mlp, which
    assumes a cost being provided by the last layer).]
    """
    def __init__(self, layer_name, **kwargs):
        if 'activation_function' in kwargs:
            raise TypeError('activation_function parameter is superfluous, '
                            'ConvSigmoid already provides the function.')
        super(ConvSigmoidC01B, self).__init__(
            layer_name, activation_function=T.nnet.sigmoid, **kwargs)


    def kl(self, Y, Y_hat):
        """
        Returns a batch (vector) of mean across units of KL divergence for
        each example KL(P || Q) where P is defined by Y and Q is defined
        by Y_hat. See mlp.Sigmoid for details.

        I need to reimplement this here because the mlp.Sigmoid assumes
        states living in a VectorSpace.
        """
        # DPR: added checks
        self.output_space.validate(Y)
        self.output_space.validate(Y_hat)

        # Pull out the argument to the sigmoid
        assert hasattr(Y_hat, 'owner')
        owner = Y_hat.owner
        assert owner is not None
        op = owner.op

        if not hasattr(op, 'scalar_op'):
            raise ValueError("Expected Y_hat to be generated by an Elemwise "
                             "op, got "+str(op)+" of type "+str(type(op)))
        assert isinstance(op.scalar_op, T.nnet.sigm.ScalarSigmoid)
        z, = owner.inputs

        term_1 = Y * T.nnet.softplus(-z)
        term_2 = (1 - Y) * T.nnet.softplus(z)

        total = term_1 + term_2

        # DPR: only part modified:
        batch_axis = self.output_space.axes.index('b')
        mean_axes = range(4)
        mean_axes.remove(batch_axis)
        ave = total.mean(axis=mean_axes)

        assert ave.ndim == 1

        return ave


    def cost(self, Y, Y_hat):

        #Sigm
        total = self.kl(Y=Y, Y_hat=Y_hat)

        ave = total.mean()

        return ave

    def get_detection_channels_from_state(self, state, target):
        """
        .. todo::

            WRITEME
        """
        batch_axis = self.output_space.axes.index('b')

        rval = OrderedDict()
        y_hat = state > 0.5
        y = target > 0.5
        wrong_bit = T.cast(T.neq(y, y_hat), state.dtype)
        rval['01_loss'] = wrong_bit.mean()
        rval['kl'] = self.cost(Y_hat=state, Y=target)

        y = T.cast(y, state.dtype)
        y_hat = T.cast(y_hat, state.dtype)
        tp = (y * y_hat).sum()
        fp = ((1-y) * y_hat).sum()
        precision = tp / T.maximum(1., tp + fp)
        recall = tp / T.maximum(1., y.sum())
        rval['precision'] = precision
        rval['recall'] = recall
        rval['f1'] = 2. * precision * recall / T.maximum(1, precision + recall)

        tp = (y * y_hat).sum(axis=batch_axis)
        fp = ((1-y) * y_hat).sum(axis=batch_axis)
        precision = tp / T.maximum(1., tp + fp)

        rval['per_output_precision.max'] = precision.max()
        rval['per_output_precision.mean'] = precision.mean()
        rval['per_output_precision.min'] = precision.min()

        recall = tp / T.maximum(1., y.sum(axis=batch_axis))

        rval['per_output_recall.max'] = recall.max()
        rval['per_output_recall.mean'] = recall.mean()
        rval['per_output_recall.min'] = recall.min()

        f1 = 2. * precision * recall / T.maximum(1, precision + recall)

        rval['per_output_f1.max'] = f1.max()
        rval['per_output_f1.mean'] = f1.mean()
        rval['per_output_f1.min'] = f1.min()

        return rval

    def get_monitoring_channels_from_state(self, state, target=None):
        rval = super(ConvSigmoidC01B, self).get_monitoring_channels_from_state(
            state)

        if target is not None:
                rval.update(self.get_detection_channels_from_state(
                    state, target))
        return rval





#todo: find non redundant solution (with C01B version)
class ConvSigmoidBC01(ConvLinearBC01):
    """
    Convolutional layer with sigmoid activation function and the costs/monitoring
    channels of the mlp Sigmoid layer (for the 'detection' monitoring style).

    [I think the cost should perhaps be factored out from the layers. Then I wouldn't
    have to write a new class for this (but refactoring might take some work; for example,
    the dropout cost assumes that the underlying cost is provided by the mlp, which
    assumes a cost being provided by the last layer).]
    """
    def __init__(self, layer_name, **kwargs):
        if 'activation_function' in kwargs:
            raise TypeError('activation_function parameter is superfluous, '
                            'ConvSigmoid already provides the function.')
        super(ConvSigmoidBC01, self).__init__(
            layer_name, activation_function=T.nnet.sigmoid, **kwargs)


    def kl(self, Y, Y_hat):
        """
        Returns a batch (vector) of mean across units of KL divergence for
        each example KL(P || Q) where P is defined by Y and Q is defined
        by Y_hat. See mlp.Sigmoid for details.

        I need to reimplement this here because the mlp.Sigmoid assumes
        states living in a VectorSpace.
        """
        # DPR: added checks
        self.output_space.validate(Y)
        self.output_space.validate(Y_hat)

        # Pull out the argument to the sigmoid
        assert hasattr(Y_hat, 'owner')
        owner = Y_hat.owner
        assert owner is not None
        op = owner.op

        if not hasattr(op, 'scalar_op'):
            raise ValueError("Expected Y_hat to be generated by an Elemwise "
                             "op, got "+str(op)+" of type "+str(type(op)))
        assert isinstance(op.scalar_op, T.nnet.sigm.ScalarSigmoid)
        z, = owner.inputs

        term_1 = Y * T.nnet.softplus(-z)
        term_2 = (1 - Y) * T.nnet.softplus(z)

        total = term_1 + term_2

        # DPR: only part modified:
        batch_axis = self.output_space.axes.index('b')
        mean_axes = range(4)
        mean_axes.remove(batch_axis)
        ave = total.mean(axis=mean_axes)

        assert ave.ndim == 1

        return ave


    def cost(self, Y, Y_hat):

        #Sigm
        total = self.kl(Y=Y, Y_hat=Y_hat)

        ave = total.mean()

        return ave

    def get_detection_channels_from_state(self, state, target):
        """
        .. todo::

            WRITEME
        """
        batch_axis = self.output_space.axes.index('b')

        rval = OrderedDict()
        y_hat = state > 0.5
        y = target > 0.5
        wrong_bit = T.cast(T.neq(y, y_hat), state.dtype)
        rval['01_loss'] = wrong_bit.mean()
        rval['kl'] = self.cost(Y_hat=state, Y=target)

        y = T.cast(y, state.dtype)
        y_hat = T.cast(y_hat, state.dtype)
        tp = (y * y_hat).sum()
        fp = ((1-y) * y_hat).sum()
        precision = tp / T.maximum(1., tp + fp)
        recall = tp / T.maximum(1., y.sum())
        rval['precision'] = precision
        rval['recall'] = recall
        rval['f1'] = 2. * precision * recall / T.maximum(1, precision + recall)

        tp = (y * y_hat).sum(axis=batch_axis)
        fp = ((1-y) * y_hat).sum(axis=batch_axis)
        precision = tp / T.maximum(1., tp + fp)

        rval['per_output_precision.max'] = precision.max()
        rval['per_output_precision.mean'] = precision.mean()
        rval['per_output_precision.min'] = precision.min()

        recall = tp / T.maximum(1., y.sum(axis=batch_axis))

        rval['per_output_recall.max'] = recall.max()
        rval['per_output_recall.mean'] = recall.mean()
        rval['per_output_recall.min'] = recall.min()

        f1 = 2. * precision * recall / T.maximum(1, precision + recall)

        rval['per_output_f1.max'] = f1.max()
        rval['per_output_f1.mean'] = f1.mean()
        rval['per_output_f1.min'] = f1.min()

        return rval

    def get_monitoring_channels_from_state(self, state, target=None):
        rval = super(ConvSigmoidBC01, self).get_monitoring_channels_from_state(
            state)

        if target is not None:
                rval.update(self.get_detection_channels_from_state(
                    state, target))
        return rval



def rectify(x):
    return x * (x > 0.)

def square(x):
    return T.square(x)

# Obsolete:


class MinNormMaxoutConvC01B(MaxoutConvC01B):
    """
    Like MaxoutConvC01B but also provides min_kernel_norm.
    """
    def __init__(self, *args, **kwargs):
        if "min_kernel_norm" in kwargs:
            self.min_kernel_norm = kwargs.pop("min_kernel_norm")

        else:
            self.min_kernel_norm = None

        super(MinNormMaxoutConvC01B, self).__init__(*args, **kwargs)


    def censor_updates(self, updates):
        """
        .. todo::

            WRITEME
        """

        if self.max_kernel_norm is not None or self.min_kernel_norm is not None:

            W ,= self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=(0,1,2)))
                if self.max_kernel_norm is None:
                    max_kernel_norm = row_norms.max()
                else:
                    max_kernel_norm = self.max_kernel_norm
                if self.min_kernel_norm is None:
                    min_kernel_norm = 0
                else:
                    min_kernel_norm = self.min_kernel_norm

                desired_norms = T.clip(row_norms, min_kernel_norm,
                                       max_kernel_norm)
                updates[W] = updated_W * (
                    desired_norms / (1e-7 + row_norms)).dimshuffle('x', 'x', 'x', 0)


