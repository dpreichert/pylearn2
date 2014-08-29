from pylearn2.train_extensions import TrainExtension
from pylearn2.gui import get_weights_report
from pylearn2.utils import serial as plserial
from pylearn2.utils.data_specs import is_flat_space
from pylearn2.space import CompositeSpace
from pylearn2.costs.cost import SumOfCosts
from pylearn2.utils.data_specs import DataSpecsMapping
import os
import os.path
import logging
import theano
import numpy as np
import time
pl2_logger = logging.getLogger('pylearn2')


class SaveChannelsExtension(TrainExtension):

    def __init__(self, save_path):
        """
        Save monitoring channels to file after each monitoring step.
        """
        self.save_path = save_path
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

    def on_monitor(self, model, dataset, algorithm):
        channels = model.monitor.channels

        for channel_name in channels:
            if hasattr(channels[channel_name], 'dataset'):
                del channels[channel_name].dataset
        t0 = time.time()
        plserial.save(self.save_path, channels)
        t1 = time.time()
        pl2_logger.info("SaveChannelsExtension: saved to %s in %ss." % (
            self.save_path, t1 - t0))


class EveryNthEpochExtension(TrainExtension):
    """
    Apply some method every Nth epoch. Abstract base class.
    """

    def __init__(self, nth_epoch, including_zero=True):
        self.nth_epoch = nth_epoch
        self.including_zero = including_zero
        self._count = 0

    def on_monitor(self, model, dataset, algorithm):
        """
        .. todo::

            WRITEME
        """
        if self._count == 0 and self.including_zero \
           or self._count !=0 and self._count % self.nth_epoch == 0:
            self.apply(model, dataset, algorithm)

        self._count += 1

    def apply(self, model, dataset, algorithm):
        raise NotImplementedError



def recursive_save_rfs(layer, save_path_base, save_path_suffix, rescale='individual'):
    # Assumes parallel layer or similar, i.e. tries to plot all layers
    if hasattr(layer, 'layers'):
        for l in layer.layers:
            recursive_save_rfs(l, save_path_base, save_path_suffix, rescale=rescale)
    else:
        patch_viewer = get_weights_report.get_weights_report(
            model=layer, rescale=rescale)
        patch_viewer.save(save_path_base + '_' + layer.layer_name +
                          '_' + save_path_suffix)

class PlotRFsExtension(EveryNthEpochExtension):

    def __init__(self, nth_epoch, save_path, including_zero=True, rescale_list=['individual',
                                                                                'global']):
        """
        The image will be saved with the epoch appended to the save_path.
        """
        self.save_path = save_path
        self.rescale_list = rescale_list
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        super(PlotRFsExtension, self).__init__(nth_epoch, including_zero=including_zero)

    def apply(self, model, dataset, algorithm):
        layers = model.layers
        for rescale in self.rescale_list:
            recursive_save_rfs(
                model.layers[0], self.save_path,
                '%s_epoch_%s.png' % (rescale, self._count), rescale=rescale)



class ComputeF1Extension(EveryNthEpochExtension):
    """
    Compute F1 sore every n epochs for mlp model.

    datasets: dictionary of datasets to compute the score on.

    save_path: if provided, save scores to file (updated whenever the model is saved
    using the regular saving from within the training algorithm, and every nth_epoch).
    """
    def __init__(self, nth_epoch, datasets, save_path=None, including_zero=True):
        self.save_path = save_path
        if os.path.exists(save_path):
            os.remove(save_path)
        self.datasets = datasets
        if save_path is not None:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
        super(ComputeF1Extension, self).__init__(nth_epoch, including_zero=including_zero)

        self.scores_at_epochs = {}
        for split in self.datasets.keys():
            self.scores_at_epochs[split] = {'precision':{}, 'recall':{}, 'f1':{}}

    def apply(self, model, dataset, algorithm):
        dataset = None
        for split in self.datasets.keys():
            pl2_logger.info("ComputeF1Extension: computing F1 score on split %s." % split)
            dataset = self.datasets[split]
            data_specs =  ((CompositeSpace([model.get_input_space(), model.get_output_space()])),
                           (model.get_input_source(), model.get_target_source()))
            # Need flat specs for iterator.
            # Taken from SGD.. is this how it is supposed to be?
            mapping = DataSpecsMapping(data_specs)
            space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
            source_tuple = mapping.flatten(data_specs[1], return_tuple=True)
            flat_data_specs = (CompositeSpace(space_tuple), source_tuple)
            iterator = dataset.iterator(mode='sequential',
                                        batch_size=algorithm.batch_size,
                                        data_specs=flat_data_specs,
                                        num_batches = algorithm.batches_per_iter)
            tp = 0
            fp = 0
            total_targets = 0

            for flat_batch in iterator:
                batch, targets = mapping.nest(flat_batch)
                if isinstance(batch, tuple):
                    # Am I handling theano correctly?
                    model_output = self._mlp_fprop(*batch)
                else:
                    model_output = self._mlp_fprop(batch)
                y_hat = model_output > 0.5
                y = targets > 0.5
                #Nope, batches can be uneven.
                #assert y.sum() == algorithm.batch_size
                tp += (y * y_hat).sum()
                fp += ((1-y) * y_hat).sum()
                total_targets += y.sum()

            precision = tp / np.maximum(1., tp + fp)
            recall = tp / np.maximum(1., total_targets)

            f1 = 2. * precision * recall / np.maximum(1, precision + recall)
            pl2_logger.info('precision: %s, recall: %s, f1: %s' % (
            precision, recall, f1))
            self.scores_at_epochs[split]['precision'][self._count] = precision
            self.scores_at_epochs[split]['recall'][self._count] = recall
            self.scores_at_epochs[split]['f1'][self._count] = f1

        if self.save_path is not None:
            pl2_logger.info("ComputeF1Extension: saving to %s" % self.save_path)
            plserial.save(self.save_path, self.scores_at_epochs)

    def setup(self, model, dataset, algorithm):
        pl2_logger.info("ComputeF1Extension: compiling fprop.")
        assert is_flat_space(model.get_input_space())
        inputs = model.get_input_space().make_theano_batch()
        if isinstance(model.get_input_space(), CompositeSpace):
            theano_inputs = inputs
        else:
            theano_inputs = [inputs]

        self._mlp_fprop = theano.function(theano_inputs, model.fprop(inputs),
                                          name='ComputeF1Extension fprop')

    def on_save(self, model, dataset, algorithm):
        if self.save_path is not None:
            pl2_logger.info("ComputeF1Extension: saving to %s" % self.save_path)
            plserial.save(self.save_path, self.scores_at_epochs)
