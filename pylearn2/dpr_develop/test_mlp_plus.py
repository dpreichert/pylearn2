import numpy as np
from collections import OrderedDict
import logging
import theano.tensor as T
from theano import function
import pylearn2.models.mlp
import pylearn2.space

from pylearn2.dpr_develop import mlp_plus

logger = logging.getLogger(__name__)


def test_conv_rectified_linear_equivalent():
    input_np = 1. * np.arange(11*11*3*6,dtype='float32').reshape(3, 11, 11, 6)
    input_np = input_np / input_np.max() * 100


    # I'd test mean pooling but that's buggy (gives nans for this setting).
    for pool_type in ['max']:

        model_mlp_plus = pylearn2.models.mlp.MLP(
            batch_size=6,
            input_space = pylearn2.space.Conv2DSpace(
                shape = (11, 11),
                num_channels = 3,
                axes=('c', 0, 1, 'b')),
            layers = [
                mlp_plus.ConvLinearBC01(
                    layer_name = 'h0',
                    output_channels = 4,
                    irange = .05,
                    kernel_shape = [5, 5],
                    kernel_stride = [2, 2],
                    activation_function = mlp_plus.rectify),
                mlp_plus.PoolBC10(
                    layer_name='h0_pool',
                    pool_shape = [3, 3],
                    pool_stride = [2, 2],
                    pool_type=pool_type),
                pylearn2.models.mlp.Sigmoid(
                    layer_name = 'y',
                    dim = 5,
                    istdev= .05)])

        input_symbolic = model_mlp_plus.input_space.make_theano_batch(batch_size=6)
        output = model_mlp_plus.fprop(input_symbolic)
        f_mlp_plus = function([input_symbolic], output) # mode='DEBUG_MODE')
        result_mlp_plus = f_mlp_plus(input_np)


        model_mlp = pylearn2.models.mlp.MLP(
            batch_size=6,
            input_space = pylearn2.space.Conv2DSpace(
                shape = (11, 11),
                num_channels = 3,
                axes=('c', 0, 1, 'b')),
            layers = [
                pylearn2.models.mlp.ConvRectifiedLinear(
                    layer_name = 'h0',
                    output_channels = 4,
                    irange = .05,
                    pool_type = pool_type,
                    kernel_shape = [5, 5],
                    kernel_stride = [2, 2],
                    pool_shape = [3, 3],
                    pool_stride = [2, 2]),
                pylearn2.models.mlp.Sigmoid(
                    layer_name = 'y',
                    dim = 5,
                    istdev= .05)])

        input_symbolic = model_mlp.input_space.make_theano_batch(batch_size=6)
        output = model_mlp.fprop(input_symbolic)
        f_mlp = function([input_symbolic], output) #mode='DEBUG_MODE')
        result_mlp = f_mlp(input_np)

        assert np.all(result_mlp == result_mlp_plus)


def test_conv_rectified_linear_from_maxout_equivalent():
    # This maxout setting corresponds to a normal conv rect linear layer
    # but using the Krizhevsky backend.
    input_np = 1. * np.arange(11*11*3*6,dtype='float32').reshape(6, 11, 11, 3)
    input_np = input_np / input_np.max() * 100

    def rectify(x):
        return x * (x > 0.)

    model_mlp_plus = pylearn2.models.mlp.MLP(
        batch_size=6,
        input_space = pylearn2.space.Conv2DSpace(
            shape = (11, 11),
            num_channels = 3,
           axes=('b', 0, 1, 'c')), # Test reformatting
        layers = [
            mlp_plus.ConvLinearC01B(
                layer_name = 'h0',
                num_channels = 16,
                irange = .05,
                kernel_shape = [5, 5],
                kernel_stride = [2, 2],
                activation_function = rectify),
            mlp_plus.MaxPoolC01B(
                layer_name='h0_pool',
                pool_shape = [3, 3],
                pool_stride = [2, 2]),
            pylearn2.models.mlp.Sigmoid(
                layer_name = 'y',
                dim = 5,
                istdev= .05)])

    input_symbolic = model_mlp_plus.input_space.make_theano_batch(batch_size=6)
    output = model_mlp_plus.fprop(input_symbolic)
    f_mlp_plus = function([input_symbolic], output) # mode='DEBUG_MODE')
    result_mlp_plus = f_mlp_plus(input_np)

    # Maxout does not support reformatting!
    input_np = input_np.transpose(3, 1, 2, 0)

    model_mlp = pylearn2.models.mlp.MLP(
        batch_size=6,
        input_space = pylearn2.space.Conv2DSpace(
            shape = (11, 11),
            num_channels = 3,
            axes=('c', 0, 1, 'b')), # Maxout does not support reformatting!
        layers = [
            pylearn2.models.maxout.MaxoutConvC01B(
                layer_name = 'h0',
                num_channels = 16,
                num_pieces = 1, #this and min_zero make it a conv relu layer
                min_zero = True,
                irange = .05,
                kernel_shape = [5, 5],
                kernel_stride = [2, 2],
                pool_shape = [3, 3],
                pool_stride = [2, 2]),
            pylearn2.models.mlp.Sigmoid(
                layer_name = 'y',
                dim = 5,
                istdev= .05)])

    input_symbolic = model_mlp.input_space.make_theano_batch(batch_size=6)
    output = model_mlp.fprop(input_symbolic)
    f_mlp = function([input_symbolic], output) #mode='DEBUG_MODE')
    result_mlp = f_mlp(input_np)

    assert np.all(result_mlp == result_mlp_plus)


def test_sigmoid_equivalent():
    # This maxout setting corresponds to a normal conv rect linear layer
    # but using the Krizhevsky backend.
    input_np = 1. * np.arange(11*11*3*6,dtype='float32').reshape(6, 11, 11, 3)
    input_np = input_np / input_np.max() * 100

    model_mlp_plus = pylearn2.models.mlp.MLP(
        batch_size=6,
        input_space = pylearn2.space.Conv2DSpace(
            shape = (11, 11),
            num_channels = 3,
        axes=('b', 0, 1, 'c')), # Test reformatting
        layers = [
            mlp_plus.DenseLinear(
                layer_name = 'h0',
                dim = 101,
                istdev= .05,
                init_bias=-2,
                activation_function=T.nnet.sigmoid),
            pylearn2.models.mlp.Sigmoid(
                layer_name = 'y',
                dim = 5,
                istdev= .05)])

    input_symbolic = model_mlp_plus.input_space.make_theano_batch(batch_size=6)
    output = model_mlp_plus.fprop(input_symbolic)
    f_mlp_plus = function([input_symbolic], output)
    result_mlp_plus = f_mlp_plus(input_np)


    model_mlp = pylearn2.models.mlp.MLP(
        batch_size=6,
        input_space = pylearn2.space.Conv2DSpace(
            shape = (11, 11),
            num_channels = 3,
            axes=('b', 0, 1, 'c')), # Test reformatting
        layers = [
            pylearn2.models.mlp.Sigmoid(
                layer_name = 'h0',
                dim = 101,
                istdev= .05,
                init_bias=-2,
                ),
            pylearn2.models.mlp.Sigmoid(
                layer_name = 'y',
                dim = 5,
                istdev= .05)])

    input_symbolic = model_mlp.input_space.make_theano_batch(batch_size=6)
    output = model_mlp.fprop(input_symbolic)
    f_mlp = function([input_symbolic], output)
    result_mlp = f_mlp(input_np)

    assert np.all(result_mlp == result_mlp_plus)




if __name__ == "__main__":
    test_conv_rectified_linear_equivalent()
    test_conv_rectified_linear_from_maxout_equivalent()
    test_sigmoid_equivalent()