__authors__ = "David Reichert"
__license__ = "3-clause BSD"

import functools

import numpy as np
from collections import OrderedDict
from theano import config
import theano
import copy

import h5py
import Image

import os
import os.path
import types

import pylearn2.utils.serial as pl2serial
from pylearn2.datasets import Dataset
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace
from pylearn2.utils.data_specs import is_flat_specs, is_flat_space, DataSpecsMapping
from pylearn2.utils.iteration import (
    FiniteDatasetIterator,
    resolve_iterator_class
)
from pylearn2.utils import safe_zip
from pylearn2.base import Block

from pylearn2.dpr_train import paths


def handle_image(image_path, resize_to=None, resize_short_dim=True):
    """
    Loads image from file, converts to RGB, removes alpha channel.

    If resize_to is True, resize the image so that the shorter or longer
    image dimension fits resize_to, depending on resize_sort_dim, keeping
    the aspect ratio.
    """
    img = Image.open(image_path)

    if img.mode == 'CMYK':
        img = img.convert(mode='RGB')

    if resize_to is not None:

        if resize_short_dim:
            resize_dim = np.argmin(img.size)
        else:
            resize_dim = np.argmax(img.size)
        ratio = 1. * resize_to / img.size[resize_dim]
        new_size = (int(np.ceil(img.size[0] * ratio)),
                    int(np.ceil(img.size[1] * ratio)))
        img = img.resize(new_size, Image.ANTIALIAS)

    img = np.array(img)

    if len(img.shape) == 2:
        # Grayscale to RGB.
        img = np.dstack([img, img, img])
    elif img.shape[2] == 4:
        print 'removing alpha: %s' % image_path
        img = img[:, :, :3]

    return img


def crop_to_square(img, width):
    row_start = img.shape[0] / 2 - width / 2
    col_start = img.shape[1] / 2 - width / 2
    return img[row_start:row_start + width,
              col_start:col_start + width, ...]

def pad_to_square(img, return_bounding_box=False):
    width = np.max(img.shape)
    if len(img.shape) == 3:
        full_image = np.zeros((width, width, img.shape[2]),
                              dtype=img.dtype)
    else:
        full_image = np.zeros((width, width), dtype=img.dtype)
    row_start = (width - img.shape[0])//2
    col_start = (width - img.shape[1])//2
    full_image[
        row_start : row_start + img.shape[0],
        col_start : col_start + img.shape[1], ...] = img

    if return_bounding_box:
        return full_image, [row_start, row_start + img.shape[0],
                     col_start, col_start + img.shape[1]]
    else:
        return full_image




def prepare_hdf5_data(base_dir, dirs_to_use=None, h5_filename=None,
                      dtype=None, conv2d_format=False, shuffle=False,
                       shuffle_seed=130820, start=None, stop=None,
                       custom_file_handlers=None):
    """
    Populates a single hdf5 data file from directories containing images, precomputed features,
    targets, etc., in individual files. Assumes that individual files across directories
    correspond to each other, as indicated through filenames (the initial parts of the file
    names need to match).

    base_dir: path to parent directory containing subdirectories with the data.

    dirs_to_use: list of strings indicating which of the subdirectories to use. By default,
    use all.

    conv2d_format: if True, final datasets will be of shape (num_data, x, y, num_channels).
    If input data has len(shape) == 2, channel axis will be added (with 1 channel).
    Throws exception if len(shape) != 2 and len(shape) != 3.

    shuffle: shuffle the ordering of the files to read out the data from.

    start / stop: out of all files, take only the files between the start'th and stop'th
    files (starting from a random file ordering if shuffle is used).

    [?: In hdf5, there are datasets (like arrays) and groups (like folders/dictionaries).
    In principle, different images (etc.) can have different dimensions, hence it would
    make sense to give each image its own dataset. However, reading things up on the internet
    seems to suggest that using many different datasets can slow things down a lot. Thus,
    we'll scale all images in a directory to the same dimensions and put them into a single
    dataset (but dimensions between directories are, in principle, allowed to differ).]
    [scaling: todo; maybe separate, don't know how to deal with features etc. For now assume
    all same size (within directories)]

    custom_file_handlers: optional dict with keys corresponding to a subset of dirs_to_use,
    and values being functions that are called to load the files in the corresponding
    directories (once per file, the filename is passed as argument). Passed to the function
    are the current file path as well as a dictionary to which meta data can be assigned:
    this will be stored in the attr attribute of the current h5py dataset, corresponding to
    the current directory. The values in the dictionary should be lists.
    These will be converted to arrays after the directory has beend handled and then assigned
    to dataset.attrs.
    """

    if stop is not None and start is not None and stop < start:
        raise ValueError
    if start is None:
        start = 0

    if dirs_to_use is None:
        #get all subdirectories
        dirs_to_use = os.walk(base_dir).next()[1]
    else:
        for direc in dirs_to_use:
            full_direc = os.path.join(base_dir, direc)
            if not os.path.isdir(full_direc):
                raise ValueError('%s is not an existing directory' % full_direc)

    if h5_filename is None:
        h5_filename = os.path.basename(os.path.normpath(base_dir))

    if custom_file_handlers is None:
        custom_file_handlers = {}
    else:
        for key in custom_file_handlers.keys():
            if key not in dirs_to_use:
                raise ValueError('Custom filehandler key %s not in dirs_to_use.' % key)

    h5_filename = h5_filename + '.h5'

    #check that filenames match across directories

    all_filenames = OrderedDict()
    file_indices = None

    for direc in dirs_to_use:
        full_direc = os.path.join(base_dir, direc)
        filenames = os.walk(full_direc).next()[2]
        filenames.sort()

        if stop is None:
            stop = len(filenames)
        else:
            if stop > len(filenames):
                raise ValueError, 'stop is larger than the number of files in directory %s' % direc

        if file_indices is None:
            if shuffle:
                shuffle_rng = np.random.RandomState(shuffle_seed)
                file_indices = shuffle_rng.permutation(len(filenames))[start:stop]
            else:
                file_indices = range(start, stop)

        filenames = np.array(filenames)[file_indices]

        filestems = []

        if len(all_filenames.values()) > 0:
            last_dir = all_filenames.keys()[-1]
            last_filenames = all_filenames[last_dir]
            if len(last_filenames) != len(filenames):
                raise RuntimeError('Number of files does not match between %s and %s: '
                                   '%i vs. %i'
                                   % (last_dir, direc, len(last_filenames), len(filenames)))
            for i_file in range(len(last_filenames)):
                filestem = os.path.commonprefix([last_filenames[i_file], filenames[i_file]])
                filestems.append(filestem)
                if len(filestems) != len(set(filestems)):
                    raise RuntimeError('No correspondence between files in direcories %s and %s '
                                       '(initial filestems between pairs of files need to match;'
                                       ' shared filestems need to be unique for each pair). '
                                       ' last two filestems: %s and %s.'
                                       % (last_dir, direc, filestems[-2], filestems[-1]))
        all_filenames[direc] = filenames

    n_dir_files = len(filenames)

    #create new file (overwrite if exist)
    #the with statement makes sure the h5 file is closed at the end (even with exceptions)
    with h5py.File(os.path.join(base_dir, h5_filename), 'w') as h5_file:
        h5_file.attrs['filestems'] = np.array(filestems)
        for direc in dirs_to_use:
            dataset = None
            original_shape = None
            filenames = all_filenames[direc]
            dataset_attrs = OrderedDict()
            for i_file, filename in enumerate(filenames):
                current_path = os.path.join(base_dir, direc, filename)
                if direc in custom_file_handlers.keys():
                    current_data = custom_file_handlers[direc](
                        current_path, dataset_attrs)
                else:
                    try:
                        current_data = Image.open(current_path)
                        current_data = np.array(current_data)
                        if current_data.shape[-1] == 4:
                            # Strip alpha channel.
                            current_data = current_data[..., :3]
                    except IOError:
                        current_data = pl2serial.load(current_path)
                    except:
                        print('Could not open %s either with Image or '
                              'pylearn2 serial, and no custom file handler '
                              'was provided for this directory. ' % filename)
                        raise

                if dataset is None:
                    data_shape = current_data.shape
                    original_shape = data_shape
                    if conv2d_format:
                        if len(data_shape) != 2 and len(data_shape) != 3:
                            raise ValueError('For conv2d_format, input data shape must have'
                                             'length 3 or 4. Directory: %s' % direc)
                        if len(data_shape) == 2:
                            data_shape = data_shape + (1,)

                    if dtype is None:
                        dtype = current_data.dtype

                    dataset = h5_file.create_dataset(
                        direc, dtype=dtype, shape=(n_dir_files,) + data_shape)
                else:
                    if current_data.shape != original_shape:
                        raise ValueError(
                            'Data shapes do not match in directory %s'
                            'First mismatch: %s.' % (direc, filename))


                #the reshape appends a channel axis for conv2d_format if necessary
                dataset[i_file, ...] = current_data.reshape(data_shape)

            dataset.attrs['filenames'] = filenames
            for key in dataset_attrs:
                dataset.attrs[key] = np.array(dataset_attrs[key])




def make_hdf5_from_dataset(dataset, h5_path, iter_batch_size=100,
                           iter_mode='sorted_shuffled_sequential',
                           data_specs=None, dtypes=None):
    """
    Make a h5 file from a dataset, iterating over the dataset with a given
    iteration mode and data_specs. Providing latter allows for changing the
    ordering and format of how the data is stored, respectively.

    If no data_specs are provided, the default data_specs of the dataset are used
    [I think this should be the default behaviour of the iterator but I'm not
    sure if it is. It's also not clear whether datasets always have a (default)
    data_specs attribute.]
    Unfortunately, pylearn2 iteration assumes that the data is stored with the batch
    axis first, so there is no point in storing data with a different axis ordering.

    The data_specs are required to be flat (i.e. the space needs to be an elementary
    space or a non-nested composite space).

    dtypes sets the dtype of the h5 datasets (pylearn2 dataset iterators will return
    config.floatX).

    [remeber to copy attrs]
    """


    if data_specs is None:
        data_specs = dataset.data_specs

    space, source = data_specs

    if not is_flat_space(space):
        raise TypeError('Space in data_specs needs to be flat.')

    if not isinstance(space, CompositeSpace):
        assert not isinstance(source, tuple)
        space = CompositeSpace([space])
        source = (source,)
        if not dtypes is None:
            if not isinstance(dtypes, list) and not isinstance(dtypes, tuple):
                dtypes = [dtypes]


    it = dataset.iterator(batch_size=iter_batch_size, mode=iter_mode,
                          data_specs=data_specs, return_tuple=True)


    h5_file = h5py.File(h5_path, 'w')
    first_batch = True
    batch_axes = []
    h5_datasets = []
    examples_seen = 0
    for batch in it:
        for i_component, component_source, component_space, component_batch in \
            safe_zip(range(len(batch)), source, space.components, batch):
            if dtypes is None:
                dtype = batch.dtype
            else:
                dtype = dtypes[i_component]

            if first_batch:
                num_examples = it.num_examples
                if isinstance(component_space, Conv2DSpace):
                    topo_shape = component_space.shape
                    num_channels = component_space.num_channels
                    # b,0,1,c
                    shape = [num_examples] + list(topo_shape) + [num_channels]
                    # putting in desired axis order
                    shape = [shape[('b', 0, 1, 'c').index(axis)]
                             for axis in component_space.axes]
                    batch_axes.append(component_space.axes.index('b'))
                elif isinstance(component_space, VectorSpace):
                    shape = (num_examples, component_space.dim)
                    batch_axes.append(0)
                else:
                    raise TypeError('make_hdf5_from_dataset does not know '
                                    'how to infer the data format (axes) for a '
                                    'space of type '
                                    '%s' % component_space.__class__)
                h5_datasets.append(h5_file.create_dataset(
                    component_source, dtype=dtype, shape=shape))

            # This unfortunately does not work:
            # "TypeError: Broadcasting is not supported for complex selections"
            # Have to do things by hand :( -- better way?
            """
            slices = [slice(0, None) for _ in range(len(component_batch.shape))]
            slices[batch_axes[i_component]] = slice(
                examples_seen, examples_seen + iter_batch_size)
            h5_datasets[i_component][slices] = component_batch
            """
            if batch_axes[i_component] == 0:
                h5_datasets[i_component][
                    examples_seen : examples_seen + iter_batch_size, ...] = \
                    component_batch
            elif batch_axes[i_component] == 1:
                h5_datasets[i_component][
                    :, examples_seen : examples_seen + iter_batch_size, ...] = \
                    component_batch
            elif batch_axes[i_component] == 2:
                h5_datasets[i_component][
                    :, :, examples_seen : examples_seen + iter_batch_size, :] = \
                    component_batch
            elif batch_axes[i_component] == 3:
                h5_datasets[i_component][
                    :, :, :, examples_seen : examples_seen + iter_batch_size] = \
                    component_batch
            else:
                assert False

        examples_seen += iter_batch_size
        print examples_seen
        first_batch = False
    # Last batch might not be full
    assert it.num_examples <= examples_seen and \
           examples_seen <= it.num_examples + iter_batch_size





class VectorSpacesDatasetNoCheck(Dataset):
    """
    This is mostly like the VectorSpacesDataset. The setup is changed slightly to avoid any type
    checking on the data (e.g. to work with h5 as underlying storage).

    This also fixes some currently incomplete implementation of VectorSpacesDataset.

    Explanation: the VectorSpacesDataset iterator gets the dataset size by calling the batch_size
    method of a space with the full data as argument (which does not seem very general; not
    sure if the space in general should be required to be able to identify the dataset size);
    batch_size calls validate. Instead, here the dataset_size needs to be provided during
    construction (better alternative?).

    VectorSpacesDataset:

    A class representing datasets being stored as a number of VectorSpaces.

    This can be seen as a generalization of DenseDesignMatrix where
    there can be any number of sources, not just X and possibly y.

    TODO: implement saving somehow, also in transformers etc. (don't pickle h5py, mlps, etc.)
    """
    _default_seed = (17, 2, 946)

    def __init__(self, dataset_size, data=None, data_specs=None, rng=_default_seed,
                 preprocessor=None, fit_preprocessor=False):
        """
        Parameters
        ----------

        dataset_size: total number of examples (no checking is done on provided data).

        data: ndarray (or similar), or tuple of ndarrays, containing the data.
            It is formatted as specified in `data_specs`.
            For instance, if `data_specs` is (VectorSpace(nfeat), 'features'),
            then `data` has to be a 2-d ndarray, of shape (nb examples,
            nfeat), that defines an unlabeled dataset. If `data_specs`
            is (CompositeSpace(Conv2DSpace(...), VectorSpace(1)),
            ('features', 'target')), then `data` has to be an (X, y) pair,
            with X being an ndarray containing images stored in the topological
            view specified by the `Conv2DSpace`, and y being a 2-D ndarray
            of width 1, containing the labels or targets for each example.

        data_specs: A (space, source) pair, where space is an instance of
            `Space` (possibly a `CompositeSpace`), and `source` is a
            string (or tuple of strings, if `space` is a `CompositeSpace`),
            defining the format and labels associated to `data`.

        rng : object, optional
            A random number generator used for picking random
            indices into the design matrix when choosing minibatches.
            [shuffled iteration not currently supported for h5py datasets]

        preprocessor: WRITEME

        fit_preprocessor: WRITEME
        """
        self.dataset_size = dataset_size

        # data_specs should be flat, and there should be no
        # duplicates in source, as we keep only one version
        assert is_flat_specs(data_specs)
        if isinstance(data_specs[1], tuple):
            assert sorted(set(data_specs[1])) == sorted(data_specs[1])
        self.data = data
        self.data_specs = data_specs

        self.compress = False
        self.design_loc = None
        if hasattr(rng, 'random_integers'):
            self.rng = rng
        else:
            self.rng = np.random.RandomState(rng)
        # Defaults for iterators
        self._iter_mode = resolve_iterator_class('sequential')

        if preprocessor:
            preprocessor.apply(self, can_fit=fit_preprocessor)
        self.preprocessor = preprocessor

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None, return_tuple=False):

        if topo is not None or targets is not None:
            raise ValueError("You should use the new interface iterator")

        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if rng is None and mode.stochastic:
            rng = self.rng
        if data_specs is None:
            data_specs = self.data_specs
        return FiniteDatasetIterator(
                self,
                mode(self.dataset_size,
                     batch_size, num_batches, rng),
                data_specs=data_specs, return_tuple=return_tuple)

    def get_data(self):
        return self.data

    def set_data(self, data, data_specs):
        # data is organized as data_specs
        # keep self.data_specs, and convert data
        data_specs[0].validate(data)
        assert not [np.any(np.isnan(X)) for X in data]
        raise NotImplementedError()

    def get_source(self, name):
        raise NotImplementedError()

    # Uncommenting this because it might not work with h5py dataset
    # -- also isn't num examples etc. handled by iterators?
    #@property
    #def num_examples(self):
    #    return self.data_specs[0].get_batch_size(self.data)
    #

    def get_batch(self, batch_size, data_specs=None):
        raise NotImplementedError()

    def get_data_specs(self):
        """
        Returns the data_specs specifying how the data is internally stored.

        This is the format the data returned by `self.get_data()` will be.
        """
        return self.data_specs




def make_vector_spaces_dataset_from_h5(
    h5_path, axes=('b', 0, 1, 'c'), load_into_memory=False, num_examples=None):
    """
    Generate a VectorSpacesDatasetNoCheck on top of a h5 file, making some assumptions about
    the underlying spaces.

    With load_into_memory==True, the data is fully loaded into memory instead of residing
    on disk in the h5 file.

    num_examples: only take the first num_examples from the h5 datasets. Currently
    requires load_into_memory == True.
    """
    if num_examples is not None:
        if not load_into_memory:
            raise ValueError('Setting num_examples requires load_into_memory.')

    if axes != ('b', 0, 1, 'c'):
        # Todo: find a solution
        print('Warning: pylearn2 iteration assumes the underlying data is stored '
              'with the batch axis first.')

    h5_file = h5py.File(h5_path, 'r')
    component_spaces = []
    last_num_examples = None
    for component_data in h5_file.values():
        if len(component_data.shape) == 2:
            component_space = VectorSpace(component_data.shape[1])
            batch_axis = 0
        elif len(component_data.shape) == 4:
            batch_axis = axes.index('b')
            channel_axis = axes.index('c')
            shape = (component_data.shape[axes.index(0)],
                     component_data.shape[axes.index(1)])
            component_space = Conv2DSpace(shape,
                                    num_channels=component_data.shape[channel_axis],
                                    axes=axes)

        else:
            raise RuntimeError('make_vector_spaces_dataset_from_h5 does not know '
                               'what space to pick for h5 dataset of shape %s.'
                               % (component_data.shape,))
        if last_num_examples is None:
            last_num_examples = component_data.shape[batch_axis]
        else:
            if last_num_examples != component_data.shape[batch_axis]:
                raise ValueError('h5 file datasets batch axis dimensions'
                                 ' do not match.')

        component_spaces.append(component_space)

    space = CompositeSpace(component_spaces)
    source = h5_file.keys()
    source = tuple([str(s) for s in source]) #converting from unicode

    data_attrs = OrderedDict()
    for key in h5_file:
        data_attrs[key] = h5_file[key].attrs

    data = tuple(h5_file.values())
    if load_into_memory:

        if num_examples is None:
            data = tuple([dat.value for dat in data])
        else:
            restricted_data = []
            for component_data in data:
                if len(component_data.shape) == 2:
                    restricted_data.append(component_data[:num_examples,:])
                elif len(component_data.shape) == 4:
                    batch_axis = axes.index('b')
                    slices = [slice(0, None) for _ in range(4)]
                    slices[batch_axis] = slice(0, num_examples)
                    # h5py throws error if not tuple.
                    slices = tuple(slices)
                    restricted_data.append(component_data[slices])
                else:
                    assert False
            data = tuple(restricted_data)

    if num_examples is None:
        num_examples = last_num_examples

    rval = VectorSpacesDatasetNoCheck(
        dataset_size=num_examples,
        data=data,
        data_specs=(space, source))
    rval.attrs = h5_file.attrs
    rval.data_attrs = data_attrs
    return rval


def restrict_specs_to_sources(data_specs, restricted_sources, return_indices=False):
    """
    Flattens data_specs and returns a composite_space or elementary space
    that is restricted to the provided sources (the ordering will be
    determined by the ordering in the original data_specs).

    [suggest to add this to the main library]
    """
    if not isinstance(restricted_sources, list) and not isinstance(restricted_sources, tuple):
        raise TypeError('restrict_to_sources must be list or tuple.')
    if len(set(restricted_sources)) != len(restricted_sources):
        raise ValueError('Source strings in restricted_sources must be unique.')

    mapping = DataSpecsMapping(data_specs)
    flat_space = mapping.flatten(data_specs[0])
    assert isinstance(flat_space, CompositeSpace)
    # I think the above always returns composite space, so the sources should be a tuple
    flat_source = mapping.flatten(data_specs[1], return_tuple=True)

    assert is_flat_specs((flat_space, flat_source))

    restricted_indices = []
    for source in restricted_sources:
        if source not in flat_source:
            raise ValueError('Mismatch between restricted_sources '
                             'and available sources in data_specs.')
        restricted_indices.append(flat_source.index(source))
    restricted_indices.sort()
    restricted_space = flat_space.restrict(restricted_indices)
    if len(restricted_sources) == 1:
        restricted_sources = restricted_sources[0]
    if return_indices:
        return restricted_indices, (restricted_space, restricted_sources)
    else:
        return (restricted_space, restricted_sources)

def restrict_tuple_to_sources(the_tuple, sources, restricted_sources,
                              return_indices=False):
    """
    Takes (flat) tuple, e.g. containing batches from multiple sources as specified
    by sources, and returns a subset of the batches as specified by restricted_sources
    (the ordering will be determined by the ordering in the original data_specs).

    [suggest to add this to the main library]
    """
    if not isinstance(restricted_sources, list) and not isinstance(restricted_sources, tuple):
        raise TypeError('restrict_to_sources must be list or tuple.')
    if len(set(restricted_sources)) != len(restricted_sources):
        raise ValueError('Source strings in restricted_sources must be unique.')
    if not isinstance(the_tuple, tuple):
        raise TypeError
    if len(the_tuple) != len(sources):
        raise ValueError('Lengths of the_tuple and sources need to match.')
    for member in the_tuple:
        if isinstance(member, tuple):
            raise TypeError('the_tuple must be flat (not contain tuples).')
    restricted_indices = []
    for restricted_source in restricted_sources:
        if restricted_source not in sources:
            raise ValueError('restricted source %s not in sources' % restricted_source)
        restricted_indices.append(sources.index(restricted_source))
    restricted_indices.sort()
    restricted_tuple = []
    for index in restricted_indices:
        restricted_tuple.append(the_tuple[index])
    if return_indices:
        return restricted_indices, tuple(restricted_tuple)
    else:
        return tuple(restricted_tuple)



class VectorSpacesTransformerDataset(Dataset):
    """
    Applies a transformation to data coming from a VectorSpacesDataset
    on the fly (when iterating).

    It looks like the standard transformer dataset makes assumptions about
    the data being in X,y format (e.g., the iterator only applies the transformation on
    the first component of composite space data), and might more generally not have been
    fully adapted to the new dataspecs interface yet (only patched for compatibility)...
    also not sure about the underlying block class used for the transformer. Hence, I need
    to implement my own version!
    """
    def __init__(self, raw, transformer, restrict_to_sources=None):
        """
        raw: the raw dataset to be iterated over.
        transformer: an object providing the pylearn2 Block interace, in particular,
        a perform function. Required to work on flat spaces (i.e.,
        either tuples of data or individual batches), for now.
        restrict_to_sources: list of sources that the transformer should be applied to
        (ordering does not matter).
        """
        if restrict_to_sources is not None:
            if not isinstance(restrict_to_sources, list) and not \
               isinstance(restrict_to_sources, tuple):
                raise TypeError('restrict_to_sources must be list or tuple.')
        self.raw = raw
        self.transformer = transformer
        raw_space, raw_source = self.raw.data_specs
        assert is_flat_specs(self.raw.data_specs)
        self.restrict_to_sources = restrict_to_sources

        if restrict_to_sources is not None and (not isinstance(raw_space, CompositeSpace) or
                                                len(restrict_to_sources) == len(raw_source)):
            if set(restrict_to_sources) != set(raw_source):
                raise ValueError('Mismatch between available and restricted sources.')

            transformer.set_input_space(raw_space)
            # canonical data_specs
            self.data_specs = (transformer.get_output_space(), raw_source)

        elif restrict_to_sources is not None:
            #this is mostly to make my life easier right now, but in general
            #I'm not sure whether datasets need nested canonical specs
            assert is_flat_specs(self.raw.data_specs)

            restricted_indices, restricted_specs = restrict_specs_to_sources(
                self.raw.data_specs, restrict_to_sources, return_indices=True)
            transformer.set_input_space(restricted_specs[0])

            # the canonical outspace will correspond to the canonical raw space
            # but with the transformed components modified
            transformer_out_space = transformer.get_output_space()
            if isinstance(transformer_out_space, CompositeSpace):
                transformed_components = copy.deepcopy(transformer_out_space.components)
            else:
                transformed_components = [transformer_out_space]

            new_components = []
            for i_component, component in enumerate(raw_space.components):
                if i_component in restricted_indices:
                    new_components.append(transformed_components.pop(0))
                else:
                    new_components.append(component)
            assert len(transformed_components) == 0

            new_space = CompositeSpace(new_components)
            self.data_specs = (new_space, raw_source)

        else:
            transformer.set_input_space(raw_space)

            # canonical data_specs
            self.data_specs = (transformer.get_output_space(), raw_source)



    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):
        # Not clear: is data_specs normally flat? What about the ordering
        # of the sources? If flat and the ordering is fixed, why have
        # source argument at all?

        isinstance(self.transformer, Block)

        if data_specs is not None:
            #dpr: apparently using flat specs for the iterator is default?
            #Not very general though...
            assert is_flat_specs(data_specs)
            #Is there a use case where I only request a subset of the sources? (why
            #not make a separate dataset then). Because if not, and data_specs are
            #always flat, why pass sources as argument at all (instead of just the
            #desired output spaces)...

            out_space, out_source = data_specs

            if set(out_source) != set(self.data_specs[1]):
                #these checks should be done more in the main library. Right now
                #it just throws .index errors without messages.
                raise NotImplementedError('Mismatch between available and requested sources'
                                          '(note that requesting a subset of sources is not '
                                          'implemented -- should it be?).')
        else:
            #TODO: figure out how to deal with this case. The implemented TransformerDataset
            #accepts None but the underlying transformer doesn't appear to handle None.
            data_specs = self.data_specs #

        # I'm requesting raw data in its canonical space, let the transformer iterator
        # handle any reformatting.
        raw_iterator = self.raw.iterator(mode=mode, batch_size=batch_size,
                num_batches=num_batches, topo=topo, targets=targets, rng=rng,
                data_specs=None, return_tuple=return_tuple)

        final_iterator = VectorSpacesTransformerIterator(
            raw_iterator, self, data_specs=data_specs,
            restrict_to_sources=self.restrict_to_sources)

        return final_iterator


class VectorSpacesTransformerIterator(object):

    def __init__(self, raw_iterator, transformer_dataset, data_specs,
                 restrict_to_sources=None):
        self.raw_iterator = raw_iterator
        self.transformer_dataset = transformer_dataset
        self.stochastic = raw_iterator.stochastic
        self.uneven = raw_iterator.uneven
        self.data_specs = data_specs
        assert is_flat_specs(data_specs)
        assert is_flat_specs(self.transformer_dataset.raw.data_specs)
        self.restrict_to_sources = restrict_to_sources

    def __iter__(self):
        return self

    def next(self):
        raw_batch = self.raw_iterator.next()

        requested_space, requested_source = self.data_specs

        raw_space, raw_source = self.transformer_dataset.raw.data_specs

        assert set(raw_source) == set(requested_source)
        if raw_source != requested_source:
            # The raw iterator is set up to return batches in the
            # canonical sources ordering of the raw dataset.
            # If the requested order is different, an intermediate space
            # is needed to handle space reformatting in the raw order.
            # The transformed batches are put in the right order at the
            # end of the function.
            assert isinstance(requested_space, CompositeSpace)
            assert isinstance(raw_batch, tuple)
            reordered_components = [
                requested_space.components[requested_source.index(source)]
                for source in raw_source]
            raw_order_out_space = CompositeSpace(reordered_components)
        else:
            #raw_order_out_space = self.transformer_dataset.data_specs[0]
            raw_order_out_space = requested_space

        # Apply transformation on raw_batch, and format it
        # to the requested Space
        transformer = self.transformer_dataset.transformer
        if self.restrict_to_sources is None:
            rval = transformer.perform(raw_batch)
            rval = transformer.get_output_space().np_format_as(rval, raw_order_out_space)
        elif not isinstance(raw_source, tuple):
            #only one source to restrict to anyway
            if isinstance(self.restrict_to_sources, tuple) or \
               isinstance(self.restrict_to_sources, list):
                if len(self.restrict_to_sources) != 1:
                    raise ValueError('Mismatch between restrict_to_sources and '
                                     'available sources in self.data_specs.')
                else:
                    if self.restrict_to_sources[0] != requested_source:
                        raise ValueError('Mismatch between restrict_to_sources and '
                                         'available sources in self.data_specs.')
            else:
                if self.restrict_to_sources != requested_source:
                    raise ValueError('Mismatch between restrict_to_sources and '
                                     'available sources in self.data_specs.')
            rval = transformer.perform(raw_batch)
            rval = transformer.get_output_space().np_format_as(rval, raw_order_out_space)
        else:
            #restricting sources

            transformed_indices, batch_to_transform = restrict_tuple_to_sources(
                raw_batch, raw_source, self.restrict_to_sources,
                return_indices=True)
            restricted_specs = restrict_specs_to_sources((raw_order_out_space, raw_source),
                                                         self.restrict_to_sources)
            if len(batch_to_transform) == 1:
                batch_to_transform = batch_to_transform[0]
            transformed = transformer.perform(batch_to_transform)
            transformed = transformer.get_output_space().np_format_as(
                transformed, restricted_specs[0])

            # recomposing all batches back together
            if isinstance(transformed, tuple):
                transformed = list(transformed)
            elif not isinstance(transformed, list):
                transformed = [transformed]

            rval = []
            for i_batch, batch in enumerate(raw_batch):
                if i_batch in transformed_indices:
                    rval.append(transformed.pop(0))
                else:
                    # Non-transformed components have not been reformatted yet (if necessary).
                    batch = raw_space.components[i_batch].np_format_as(
                        batch, raw_order_out_space.components[i_batch])
                    rval.append(batch)
            assert len(transformed) == 0

            rval = tuple(rval)

        if not isinstance(rval, tuple):
            if isinstance(raw_batch, tuple):
                # raw iterator was instantiated with return_tuple == True
                # (dpr: not sure if that's how return_tuple is supposed to work)
                rval = (rval, )
        else:
            if raw_source != requested_source:
                # Reorder batches as requested
                rval = tuple([rval[raw_source.index(source)]
                        for source in requested_source])
        requested_space.np_validate(rval)
        return rval

    @property
    def num_examples(self):
        return self.raw_iterator.num_examples


class PatchExtractorDataset(Dataset):
    """
    Extracts patchs from a raw dataset on the fly. A configurable number of patches
    is extracted from a batch by applying the provided transformer to the batch
    multiple times (thus the transformer should be configured to extract patches
    at different positions when called repeatedly). Not designed to be efficient.

    The current implementation works as follows (probably not ideal, but I wasn't
    sure how to best deal with the iterators etc.):

    When requesting batches of size batch_size from the iterator, the batches
    are produced by extracting batch_size patches from raw batches of size
    (batch_size / num_patches_per_raw). The raw dataset is iterated through
    num_raw_iterations times, producing a total number of patches that is approx.
    num_raw_iterations & num raw examples * num_patches_per_raw. However,
    currently if num raw examples is not a multiple of (batch_size / num_patches_per_raw),
    any leftover examples will not be used in one raw iteration (and not all if the
    raw iterator is not stochastic).

    """

    def __init__(self, raw, transformer, num_raw_iterations, num_patches_per_raw):

        self.raw = raw
        self.transformer = transformer
        self.num_raw_iterations = num_raw_iterations
        self.num_patches_per_raw = num_patches_per_raw

        raw_space, raw_source = self.raw.data_specs
        # canonical data_specs
        transformer.set_input_space(raw_space)
        self.data_specs = (transformer.get_output_space(), raw_source)


    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):

        isinstance(self.transformer, Block)

        # Need to figure out how to handle these cases.
        if batch_size is None:
            raise NotImplementedError

        if batch_size % self.num_patches_per_raw != 0.:
            raise ValueError('batch_size needs to be divisible by num_patches_per_raw.')
        raw_batch_size = batch_size / self.num_patches_per_raw

        raw_iterator = self.raw.iterator(mode='sequential', batch_size=1)

        if raw_batch_size > raw_iterator.num_examples:
            raise RuntimeError('Raw dataset number of examples is smaller than '
                               'batch_size / num_patches_per_raw.')

        num_raw_used_per_raw_it = (raw_iterator.num_examples // raw_batch_size) * raw_batch_size

        if num_raw_used_per_raw_it != raw_iterator.num_examples:
            print('Patch extractor warning: raw dataset size is not divisible by '
                  'raw batch size (computed as batch_size / num_patches_per_raw). '
                  '%i raw examples will potentially not be used.'
                  % (raw_iterator.num_examples - num_raw_used_per_raw_it))
        num_patches = num_raw_used_per_raw_it * self.num_raw_iterations * self.num_patches_per_raw

        if data_specs is not None:
            #dpr: apparently using flat specs for the iterator is default?
            #Not very general though...
            assert is_flat_specs(data_specs)

            out_space, out_source = data_specs

            if set(out_source) != set(self.data_specs[1]):
                #these checks should be done more in the main library. Right now
                #it just throws .index errors without messages.
                raise NotImplementedError('Mismatch between available and requested sources'
                                          '(note that requesting a subset of sources is not '
                                          'implemented -- should it be?).')
        else:
            data_specs = self.data_specs #

        def make_raw_iterator(the_self):
            # I'm requesting raw data in its canonical space, let the transformer iterator
            # handle any reformatting.
            raw_it = the_self.raw.iterator(
                mode=mode, batch_size=raw_batch_size,
                num_batches=num_batches, topo=topo, targets=targets, rng=rng,
                data_specs=None, return_tuple=return_tuple)
            return raw_it

        # This dynamically adds the method reset_raw_iterator to this PatchExtractorDataset
        # instance (note that assigning it to self would apparenly add it to the class, not
        # the instance).
        self.make_raw_iterator = types.MethodType(
            make_raw_iterator, self, PatchExtractorDataset)

        final_iterator = PatchExtractorIterator(
            self, data_specs, batch_size, num_patches, raw_batch_size)

        return final_iterator


class PatchExtractorIterator(object):
    """
    Used by PatchExtractorDataset. Similar to the VectorSpacesTransformerIterator,
    but can apply the transformer multiple times per raw batch (e.g. to extract multiple
    patches on the fly), yielding larger batches in the process. The number of examples
    generated in this way can be larger than the number of examples in the raw
    dataset; the PatchExtractorDataset can iterate multiple times through the raw
    data, resetting the raw iterator on request of PatchExtractorIterator whenever
    necessary.

    Also does not support applying the transformer only to a subset of the batches,
    because the batch sizes would would then no longer necessarily match.

    See PatchExtractorDataset for more explanation.
    """

    def __init__(self, my_dataset, data_specs, batch_size, num_patches, raw_batch_size):
        """
        my_dataset should be a PatchExtractorDataset or similar.
        """
        self.my_dataset = my_dataset
        self.data_specs = data_specs
        assert is_flat_specs(data_specs)
        assert is_flat_specs(self.my_dataset.raw.data_specs)
        self.batch_size = batch_size
        self.num_patches = num_patches
        self.raw_batch_size = raw_batch_size

        self.raw_iterator = my_dataset.make_raw_iterator()
        self.stochastic = self.raw_iterator.stochastic
        self.uneven = self.raw_iterator.uneven

        self._num_examples_left = num_patches

        if isinstance(self.my_dataset.raw.data_specs[0], CompositeSpace):
            spaces_to_check = self.my_dataset.raw.data_specs[0].components
        else:
            spaces_to_check = [self.my_dataset.raw.data_specs[0]]
        for space in spaces_to_check:
            if not isinstance(space, VectorSpace) and not hasattr(space, 'axes'):
                raise NotImplementedError(
                    'PatchExtractorIterator does not know how to '
                    'deal with %s space (how to find batch axis?)' % space.__class__)

    def __iter__(self):
        return self

    def next(self):
        assert self._num_examples_left >= 0

        if self._num_examples_left == 0:
            raise StopIteration

        requested_space, requested_source = self.data_specs

        raw_space, raw_source = self.my_dataset.raw.data_specs


        # Request raw batch, start from the beginning of the raw data if necessary
        try:
            raw_batch = self.raw_iterator.next()
            if raw_space.np_batch_size(raw_batch) < self.raw_batch_size:
                self.raw_iterator = self.my_dataset.make_raw_iterator()
                raw_batch = self.raw_iterator.next()
        except StopIteration:
            self.raw_iterator = self.my_dataset.make_raw_iterator()
            raw_batch = self.raw_iterator.next()

        assert raw_space.np_batch_size(raw_batch) == self.raw_batch_size

        assert set(raw_source) == set(requested_source)
        if raw_source != requested_source:
            # The raw iterator is set up to return batches in the
            # canonical sources ordering of the raw dataset.
            # If the requested order is different, an intermediate space
            # is needed to handle space reformatting in the raw order.
            # The transformed batches are put in the right order at the
            # end of the function.
            assert isinstance(requested_space, CompositeSpace)
            assert isinstance(raw_batch, tuple)
            reordered_components = [
                requested_space.components[requested_source.index(source)]
                for source in raw_source]
            raw_order_out_space = CompositeSpace(reordered_components)
        else:
            #raw_order_out_space = self.my_dataset.data_specs[0]
            raw_order_out_space = requested_space


        transformer = self.my_dataset.transformer

        num_extractions = self.batch_size / self.raw_batch_size
        assert num_extractions == self.my_dataset.num_patches_per_raw

        patches_list = []
        for _ in range(num_extractions):
            patches_list.append(transformer.perform(raw_batch))

        # Concatenating batches
        assert is_flat_space(raw_space)

        transformer_out_space = transformer.get_output_space()
        def get_batch_axis(elementary_space):
            if isinstance(elementary_space, VectorSpace):
                batch_axis = 1
            elif hasattr(elementary_space, 'axes'):
                batch_axis = elementary_space.axes.index('b')
            else:
                assert False #__init__ should have checked spaces
            return batch_axis

        if isinstance(raw_batch, tuple):
            assert isinstance(transformer_out_space, CompositeSpace)
            out_batch = []
            for i_component, component_space in enumerate(transformer_out_space.components):
                batch_axis = get_batch_axis(component_space)
                component_patches = [patches_list[i_extraction][i_component]
                                     for i_extraction in range(num_extractions)]
                component_out_patch = np.concatenate(
                    component_patches, axis=batch_axis)

                out_batch.append(component_out_patch)

            out_batch = tuple(out_batch)
        else:
            assert not isinstance(transformer_out_space, CompositeSpace)
            batch_axis = get_batch_axis(transformer_out_space)

            out_batch = np.concatenate(patches_list, axis=batch_axis)

        out_batch = transformer.get_output_space().np_format_as(
            out_batch, raw_order_out_space)

        if not isinstance(out_batch, tuple):
            if isinstance(raw_batch, tuple):
                # raw iterator was instantiated with return_tuple == True
                # (dpr: not sure if that's how return_tuple is supposed to work)
                out_batch = (out_batch, )
        else:
            if raw_source != requested_source:
                # Reorder batches as requested
                out_batch = tuple([out_batch[raw_source.index(source)]
                        for source in requested_source])
        requested_space.np_validate(out_batch)
        assert requested_space.np_batch_size(out_batch) == self.batch_size

        self._num_examples_left -= self.batch_size
        return out_batch

    @property
    def num_examples(self):
        return self.num_patches


class WindowAndFlipTransformer(object):
    """
    Similarly to the WindowAndFlip extension, this transformer takes a batch
    (living in a Conv2DSpace or CompositeSpace of Conv2Dspaces) and
    returns a window/patch (centered or random position, same position across batch
    and spaces--spaces thus need to have same shape), and optionally flips the content
    horizontally (with 50% chance; whole batch).

    [I'm only slicing so I'm not bothering with going through theano/the Block class].
    """
    def __init__(self, window_shape, center=True, flip=True, rng_seed=11314):
        self.window_shape = window_shape
        self.center = center
        self.flip = flip
        self.rng_seed = rng_seed
        self.np_rng = np.random.RandomState(self.rng_seed)

    def set_input_space(self, space):
        if not (isinstance(space, Conv2DSpace) or isinstance(space, CompositeSpace)):
            raise TypeError('WindowAndFlipTransformer input space must be '
                            'Conv2DSpace or CompositeSpace containing '
                            'Conv2DSpaces')
        if isinstance(space, CompositeSpace):
            for component_space in space.components:
                if not isinstance(component_space, Conv2DSpace):
                    raise TypeError('WindowAndFlipTransformer input space must be '
                                    'Conv2DSpace or CompositeSpace containing '
                                    'Conv2DSpaces')
            spaces_to_check = space.components
        else:
            spaces_to_check = [space]

        last_shape = None
        for conv_space in spaces_to_check:
            shape = conv_space.shape
            if shape[0] < self.window_shape[0] or shape[1] < self.window_shape[1]:
                raise RuntimeError('Window shape larger than input space shape, '
                                   '%s vs %s' % (self.window_shape, shape))
            if last_shape is not None:
                if last_shape != shape:
                    raise RuntimeError('Conv2DSpaces need to have same shape.')
            last_shape = shape


        self.input_space = space

        if isinstance(space, CompositeSpace):
            output_components = [Conv2DSpace(self.window_shape,
                                             num_channels=component_space.num_channels,
                                             axes=component_space.axes)
                                 for component_space in space.components]
            self.output_space = CompositeSpace(output_components)
        else:
            self.output_space = Conv2DSpace(self.window_shape,
                                            num_channels=space.num_channels,
                                            axes=space.axes)

    def get_input_space(self):
        return self.input_space

    def get_output_space(self):
        return self.output_space

    def perform(self, batches):
        if not hasattr(self, 'input_space'):
            raise RuntimeError('The transformer input space needs to be set first'
                               'with set_input_space, before calling perform.')

        self.input_space.np_validate(batches)

        if isinstance(self.input_space, Conv2DSpace):
            batches = [batches]
            input_shape = self.input_space.shape
            all_axes = [self.input_space.axes]
        else:
            input_shape = self.input_space.components[0].shape
            all_axes = [component.axes for component in self.input_space.components]

        if self.center:
            row_start = input_shape[0] / 2 - self.window_shape[0] / 2
            col_start = input_shape[1] / 2 - self.window_shape[1] / 2
        else:
            row_start = self.np_rng.randint(0, high = input_shape[0] - self.window_shape[0] + 1)
            col_start = self.np_rng.randint(0, high = input_shape[1] - self.window_shape[1] + 1)

        row_stop = row_start + self.window_shape[0]
        col_stop = col_start + self.window_shape[1]

        if self.flip:
            flip_now = self.np_rng.choice([False,True])
        else:
            flip_now = False

        out_batches = []
        for batch, axes in safe_zip(batches, all_axes):
            slices = [slice(None) for _ in range(4)]
            slices[axes.index(0)] = slice(row_start, row_stop)
            slices[axes.index(1)] = slice(col_start, col_stop)
            window = batch[slices]
            if flip_now:
                slices = [slice(None) for _ in range(4)]
                slices[axes.index(1)] = slice(None, None, -1)
                window = window[slices]
            out_batches.append(window)

        if isinstance(self.output_space, CompositeSpace):
            out_batches = tuple(out_batches)
        else:
            out_batches = out_batches[0]

        self.output_space.np_validate(out_batches)

        return out_batches


class PadTransformer(object):
    """
    pad_sizes: [top, bottom, left, right]

    [need to check that padding does what I want]
    """
    def __init__(self, pad_sizes):
        self.pad_sizes = pad_sizes

    def set_input_space(self, space):
        if not (isinstance(space, Conv2DSpace) or isinstance(space, CompositeSpace)):
            raise TypeError('PadTransformer input space must be '
                            'Conv2DSpace or CompositeSpace containing '
                            'Conv2DSpaces')
        if isinstance(space, CompositeSpace):
            for component_space in space.components:
                if not isinstance(component_space, Conv2DSpace):
                    raise TypeError('PadTransformer input space must be '
                                    'Conv2DSpace or CompositeSpace containing '
                                    'Conv2DSpaces')

        self.input_space = space
        if isinstance(space, CompositeSpace):
            output_components = [Conv2DSpace(
                (component_space.shape[0] + self.pad_sizes[0] + self.pad_sizes[1],
                 component_space.shape[1] + self.pad_sizes[2] + self.pad_sizes[3]),
                num_channels=component_space.num_channels,
                axes=component_space.axes)
                                 for component_space in space.components]
            self.output_space = CompositeSpace(output_components)
        else:
            self.output_space = Conv2DSpace(
                (space.shape[0] + self.pad_sizes[0] + self.pad_sizes[1],
                 space.shape[1] + self.pad_sizes[2] + self.pad_sizes[3]),
                num_channels=space.num_channels,
                axes=space.axes)

    def get_input_space(self):
        return self.input_space

    def get_output_space(self):
        return self.output_space

    def perform(self, batches):
        if not hasattr(self, 'input_space'):
            raise RuntimeError('The transformer input space needs to be set first'
                               'with set_input_space, before calling perform.')
        batch_size = self.input_space.np_batch_size(batches)
        if isinstance(self.input_space, Conv2DSpace):
            batches = [batches]
            in_spaces = [self.input_space]
            out_spaces = [self.output_space]
        else:
            all_axes = [component.axes for component in self.input_space.components]
            in_spaces = self.input_space.components
            out_spaces = self.output_space.components

        out_batches = []

        for batch, in_space, out_space in safe_zip(batches, in_spaces, out_spaces):
            assert in_space.axes == out_space.axes
            out_batch = out_space.get_origin_batch(batch_size, dtype=batch.dtype)
            row_start = self.pad_sizes[0]
            row_stop = row_start + in_space.shape[0]
            col_start = self.pad_sizes[1]
            col_stop = col_start + in_space.shape[1]
            slices = [slice(None) for _ in range(4)]
            slices[in_space.axes.index(0)] = slice(row_start, row_stop)
            slices[in_space.axes.index(1)] = slice(col_start, col_stop)
            out_batch[slices] = batch
            out_batches.append(out_batch)

        if isinstance(self.output_space, CompositeSpace):
            out_batches = tuple(out_batches)
        else:
            out_batches = out_batches[0]

        self.output_space.np_validate(out_batches)

        return out_batches


class GCNTransformer(object):
    """
    This transformer takes a batch and applies (in place) global contrast normalization
    operations on a per example basis (preprocessing and saving the data is
    of course generally more efficient).

    The operations are mostly taken from expr.preprocessing.global_contrast_normalize,
    but extended to batches not in design matrix format (can be conv2d or nested as well).

    [I'm not bothering with going through theano/the Block class but am doing things in np
    instead.].
    """
    def __init__(self, scale=1., subtract_mean=True, divide_by='std',
                 sqrt_bias=0., min_divisor=1e-8):
        """
        scale : float, optional
        Multiply features by this const.

        subtract_mean : bool, optional
        Remove the mean across features/pixels before normalizing.
        Defaults to `True`.

        divide_by : one of 'std', 'norm', None; optional
        Normalize by the per-example standard deviation across features,
        the vector norm, or not at all. Defaults to 'std'.

        sqrt_bias : float, optional
        Fudge factor added inside the square root. Defaults to 0.

        min_divisor : float, optional
        If the divisor for an example is less than this value,
        do not apply it. Defaults to `1e-8`.
        """
        if divide_by not in ['std', 'norm', None]:
            raise ValueError("divide_by must be one of 'std', 'norm', None")
        self.scale = scale
        self.subtract_mean = subtract_mean
        self.divide_by = divide_by
        self.sqrt_bias = sqrt_bias
        self.min_divisor = min_divisor

    def set_input_space(self, space):

        def recursive_check_space(space):
            if isinstance(space, CompositeSpace):
                if len(space.components) == 0:
                    raise RuntimeError("GCNTransformer input composite space has no component"
                                       " spaces.")
                for component_space in space.components:
                    recursive_check_space(component_space)
            elif not isinstance(space, VectorSpace) and not hasattr(space, 'axes'):
                raise NotImplementedError('GCNTransformer does not know how to deal with %s '
                                          'space (how to find batch axis?)' % space.__class__)

        recursive_check_space(space)
        self.input_space = space
        self.output_space = self.input_space

    def get_input_space(self):
        return self.input_space

    def get_output_space(self):
        return self.output_space

    def perform(self, batches):
        if not hasattr(self, 'input_space'):
            raise RuntimeError('The transformer input space needs to be set first '
                               'with set_input_space, before calling perform.')

        self.input_space.np_validate(batches)

        def gcn(batch, axes_to_gcn):
            # yeah that actually happened, and mean() doesn't complain
            assert axes_to_gcn is not None

            broadcast = [slice(None) for _ in range(len(batch.shape))]
            for axis in axes_to_gcn:
                broadcast[axis] = np.newaxis

            #doing things in place so I don't need to know where I am in the nested tuples
            if self.subtract_mean:
                mean = np.mean(batch, axis=axes_to_gcn)

                batch[...] = batch - mean[broadcast]
            if self.divide_by is None:
                batch *= self.scale
                return
            if self.divide_by == 'std':
                normalizers = np.sqrt(
                    self.sqrt_bias + batch.var(axis=axes_to_gcn, ddof=1)) / self.scale
            elif self.divide_by == 'norm':
                normalizers = np.sqrt(
                    self.sqrt_bias + np.square(batch).sum(axis=axes_to_gcn)) / self.scale
            else:
                assert False
            batch /= normalizers[broadcast]


        # I guess the recursive bit is unnecessary as iterators normally work with
        # flat specs?

        def recursive_gcn(space, batch):
            if isinstance(space, VectorSpace):
                assert not isinstance(batch, tuple)
                axes_to_gcn = 1
                gcn(batch, axes_to_gcn)

            elif hasattr(space, 'axes'):
                assert not isinstance(batch, tuple)
                batch_axis = space.axes.index('b')
                axes_to_gcn = range(len(space.axes))
                axes_to_gcn.remove(batch_axis)
                axes_to_gcn = tuple(axes_to_gcn)
                gcn(batch, axes_to_gcn)

            elif isinstance(space, CompositeSpace):
                assert isinstance(batch, tuple)
                for component_space, component_batch in safe_zip(space.components, batch):
                    recursive_gcn(component_space, component_batch)
            else:
                assert False #set_input_space should have checked the spaces

        recursive_gcn(self.input_space, batches)
        self.output_space.np_validate(batches)

        return batches


class FunctionTransformer(object):
    """
    Applies provided (non-theano, space-preserving) function to data batches (in place).
    Suitable for simple element-wise transformers that don't need to know anything
    about the underlying spaces or batch axes.
    """
    def __init__(self, function):
        self.function = function

    def set_input_space(self, space):
        self.input_space = space
        self.output_space = space

    def get_input_space(self):
        return self.input_space

    def get_output_space(self):
        return self.output_space

    def perform(self, batch):

        #not sure if working on nested tuples is even needed
        def recursive_apply(b):
            if isinstance(b, tuple) or isinstance(b, list):
                for x in b:
                    recursive_apply(x)
            else:
                b[...] = self.function(b)

        recursive_apply(batch)
        return batch



class SlidingWindowTransformer(object):

    # could be more general than mlp (block once updated?)

    def __init__(self, window_mlp):
        """
        If the input space is a CompositeSpace, the window_mlp is either applied
        to each component independently (if the mlp input space is a Conv2Space),
        or to the CompositeSpace as a whole (if the mlp input is itself a
        compatible CompositeSpace).

        The window_mlp output needs to be a vector space of dim 1. Currently,
        this also is an requirement if the mlp uses the whole composite input.
        This implies that the transformer actually combines sources to compute something
        else. As a workaround, I return the output multiple times to keep the input
        structure in tact. What is needed in general for the transformer dataset is
        the possibility to map input to output sources. (for simplicity, I furthermore
        require that the mlp input component spaces are all of the same form).
        """

        self.window_mlp = window_mlp
        mlp_input_space = self.window_mlp.get_input_space()
        mlp_output_space = self.window_mlp.get_output_space()
        if not isinstance(mlp_output_space, VectorSpace) or mlp_output_space.dim != 1:
            raise TypeError('window_mlp output space must be VectorSpace with dim 1.')
        if not self.check_input_space(mlp_input_space):
            raise TypeError('SlidingWindowTransformer window_mlp input space must be '
                                       'Conv2DSpace or CompositeSpace containing '
                                       'Conv2DSpaces')

    def check_input_space(self, space):
        if not (isinstance(space, Conv2DSpace) or isinstance(space, CompositeSpace)):
            return False
        if isinstance(space, CompositeSpace):
            for component_space in space.components:
                if not isinstance(component_space, Conv2DSpace):
                    return False
        return True

    def set_input_space(self, space):
        if not self.check_input_space(space):
            raise TypeError('SlidingWindowTransformer input space must be '
                            'Conv2DSpace or CompositeSpace containing '
                            'Conv2DSpaces')

        self.input_space = space

        mlp_input_space = self.window_mlp.get_input_space()
        if isinstance(mlp_input_space, CompositeSpace):
            if not isinstance(space, CompositeSpace):
                raise TypeError('window_mlp has composite input space but'
                                ' transformer input space is not composite.')
            if len(mlp_input_space.components) != len(space.components):
                raise RuntimeError('window_mlp composite input space and '
                                   'transformer input space do not have the same '
                                   'number components.')

        self.compile_mlp_theano()

        def get_conv2d_out_space(in_conv2d_space, mlp_in_conv2d_space):
            assert isinstance(in_conv2d_space, Conv2DSpace)
            in_shape = in_conv2d_space.shape
            mlp_shape = mlp_in_conv2d_space.shape

            if in_shape[0] < mlp_shape[0] or in_shape[1] < mlp_shape[1]:
                raise ValueError("Input space 2D shape too small for window mlp. "
                                 "%s vs %s" % (in_shape, mlp_shape))
            if mlp_in_conv2d_space.num_channels != in_conv2d_space.num_channels:
                raise ValueError(
                    "Channels of input space different from window "
                    "mlp input space, %i vs. %i" %
                    (in_conv2d_space.num_channels, mlp_in_conv2d_space.num_channels))

            out_shape = (in_shape[0] - mlp_shape[0] + 1, in_shape[1] - mlp_shape[1] + 1)
            out_space = Conv2DSpace(out_shape, num_channels=1,
                                    axes=mlp_in_conv2d_space.axes)
            return out_space


        if isinstance(space, CompositeSpace):
            if not isinstance(mlp_input_space, CompositeSpace):
                output_components = [get_conv2d_out_space(component_space, mlp_input_space)
                                     for component_space in space.components]
                self.output_space = CompositeSpace(output_components)
            else:
                # Workaround, see __init__ docu.
                first_mlp_in_component = mlp_input_space.components[0]
                first_in_component = space.components[0]

                for mlp_in_component, in_component in safe_zip(
                    mlp_input_space.components[1:], space.components[1:]):
                    if first_mlp_in_component != mlp_in_component or \
                       first_in_component != in_component:
                        raise NotImplementedError

                out_component = get_conv2d_out_space(in_component, mlp_in_component)
                output_components = [out_component for _ in space.components]
            self.output_space = CompositeSpace(output_components)
        else:
            self.output_space = get_conv2d_out_space(space, mlp_input_space)



    def compile_mlp_theano(self):
        print("SlidingWindowTransformer: compiling window mlp fprop.")
        assert is_flat_space(self.get_input_space())
        inputs = self.get_input_space().make_theano_batch()
        if isinstance(self.get_input_space(), CompositeSpace):
            theano_inputs = inputs
        else:
            theano_inputs = [inputs]


        self._mlp_fprop = theano.function(theano_inputs, self.window_mlp.fprop(inputs),
                                          name='SlidingWindowTransformer mlp fprop')


    def get_input_space(self):
        return self.input_space

    def get_output_space(self):
        return self.output_space


    def perform(self, batches):
        batch_size = self.input_space.np_batch_size(batches)
        mlp_input_space = self.window_mlp.get_input_space()


        # Temp. workaround. See __init__ docu. Also needs cleanup.
        if isinstance(mlp_input_space, CompositeSpace):
            assert isinstance(self.input_space, CompositeSpace)
            assert len(self.input_space.components) == len(mlp_input_space.components)
            assert isinstance(self.output_space, CompositeSpace)

            batches = self.input_space.np_format_as(batches, mlp_input_space)

            mlp_in_component = mlp_input_space.components[0]
            out_component = self.output_space.components[0]

            out_shape = [batch_size, out_component.shape[0],
                         out_component.shape[1], 1]

            # To put axes in the right order (as specified by the window mlp input space).
            def put_in_mlp_axes_order(b01c_list):
                return [b01c_list[('b', 0, 1, 'c').index(axis)]
                        for axis in mlp_in_component.axes]

            out_shape = put_in_mlp_axes_order(out_shape)

            out = np.zeros(out_shape, dtype=config.floatX)

            batch_slice = slice(0, batch_size)
            channel_slice = slice(0, mlp_in_component.num_channels)

            for i_row in range(out_component.shape[0]):
                for j_col in range(out_component.shape[1]):
                    row_slice = slice(i_row, i_row + mlp_in_component.shape[0])
                    col_slice = slice(j_col, j_col + mlp_in_component.shape[1])
                    in_slices = [batch_slice, row_slice, col_slice, channel_slice]
                    in_slices = put_in_mlp_axes_order(in_slices)

                    window = [batch[in_slices] for batch in batches]

                    out_slices = [batch_slice, slice(i_row, i_row + 1),
                                  slice(j_col, j_col + 1), slice(0, None)]
                    out_slices = put_in_mlp_axes_order(out_slices)

                    # mlp output is single number per batch member (nx1). Extending axes.
                    mlp_out_slices = [batch_slice, slice(0, None), np.newaxis, np.newaxis]
                    mlp_out = self._mlp_fprop(*window)[mlp_out_slices]
                    # Axes need to match the out array.
                    mlp_out = Conv2DSpace.convert_numpy(
                        mlp_out, ('b', 0, 1, 'c'), mlp_in_component.axes)
                    out[out_slices] = mlp_out
            out_batches = tuple([out for _ in self.output_space.components])


        else:
            def apply_to_component(in_space, out_space, batch):
                assert isinstance(in_space, Conv2DSpace)
                assert isinstance(out_space, Conv2DSpace)

                if in_space.axes != mlp_input_space.axes:
                    batch = in_space.convert_numpy(batch, in_space.axes, mlp_input_space.axes)

                out_shape = [batch_size, out_space.shape[0],
                             out_space.shape[1], 1]

                # To put axes in the right order (as specified by the window mlp input space).
                def put_in_mlp_axes_order(b01c_list):
                    return [b01c_list[('b', 0, 1, 'c').index(axis)]
                            for axis in mlp_input_space.axes]

                out_shape = put_in_mlp_axes_order(out_shape)

                out = np.zeros(out_shape, dtype=config.floatX)

                batch_slice = slice(0, batch_size)
                channel_slice = slice(0, mlp_input_space.num_channels)

                for i_row in range(out_space.shape[0]):
                    for j_col in range(out_space.shape[1]):
                        row_slice = slice(i_row, i_row + mlp_input_space.shape[0])
                        col_slice = slice(j_col, j_col + mlp_input_space.shape[1])
                        in_slices = [batch_slice, row_slice, col_slice, channel_slice]
                        in_slices = put_in_mlp_axes_order(in_slices)

                        window = batch[in_slices]

                        out_slices = [batch_slice, slice(i_row, i_row + 1),
                                      slice(j_col, j_col + 1), slice(0, None)]
                        out_slices = put_in_mlp_axes_order(out_slices)

                        # mlp output is single number per batch member (nx1). Extending axes.
                        mlp_out_slices = [batch_slice, slice(0, None), np.newaxis, np.newaxis]
                        mlp_out = self._mlp_fprop(window)[mlp_out_slices]
                        # Axes need to match the out array.
                        mlp_out = Conv2DSpace.convert_numpy(
                            mlp_out, ('b', 0, 1, 'c'), mlp_input_space.axes)
                        out[out_slices] = mlp_out
                return out


            if isinstance(self.input_space, CompositeSpace):
                out_batches = []
                assert isinstance(self.output_space, CompositeSpace)
                for in_component, out_component, batch in safe_zip(
                    self.input_space.components, self.output_space.components, batches):
                    out_batches.append(
                        apply_to_component(in_component, out_component, batch))
                out_batches = tuple(out_batches)
            else:
                out_batches = apply_to_component(self.input_space, self.output_space, batches)

        self.output_space.np_validate(out_batches)

        return out_batches





if __name__ == "__main__":

    #make a proper test suite somehow

    #'/home/dreiche2/Data/temp/scene4'


    """
    prepare_hdf5_data(os.path.join(paths.mc_data_path, 'scene4'),
                      h5_filename='scene4_train',
                      dirs_to_use=['images_16','contour_groundTruth'],
                      conv2d_format=True, shuffle=True,
                      start=0, stop=100)
    prepare_hdf5_data(os.path.join(paths.mc_data_path, 'scene4'),
                      h5_filename='scene4_test',
                      dirs_to_use=['images_16','contour_groundTruth'],
                      conv2d_format=True, shuffle=True,
                      start=100)

    prepare_hdf5_data(os.path.join(paths.mc_data_path, 'scene4'),
                      h5_filename='scene4_stereo_train',
                      dirs_to_use=['stereoLeft_groundTruth', 'stereoRight_groundTruth',
                                   'contour_groundTruth'],
                      conv2d_format=True, shuffle=True,
                      start=0, stop=100)
    prepare_hdf5_data(os.path.join(paths.mc_data_path, 'scene4'),
                      h5_filename='scene4_stereo_test',
                      dirs_to_use=['stereoLeft_groundTruth', 'stereoRight_groundTruth',
                                   'contour_groundTruth'],
                      conv2d_format=True, shuffle=True,
                      start=100)
    lalala
    """



    h5_train = h5py.File(os.path.join(paths.mc_data_path, 'scene4/scene4_train.h5'))

    space = CompositeSpace([Conv2DSpace(component_data.shape[1:3],
                                        num_channels=component_data.shape[3],
                                        axes=('b', 0, 1, 'c'))
                            for component_data in h5_train.values()])
    source = h5_train.keys()
    source = tuple([str(s) for s in source]) #converting from unicode
    train_data = VectorSpacesDatasetNoCheck(
        dataset_size=h5_train.values()[0].shape[0],
        data=tuple(h5_train.values()),
        data_specs=(space, source))

    h5_test = h5py.File(os.path.join(paths.mc_data_path, 'scene4/scene4_test.h5'))

    space = CompositeSpace([Conv2DSpace(component_data.shape[1:3],
                                        num_channels=component_data.shape[3],
                                        axes=('b', 0, 1, 'c'))
                            for component_data in h5_test.values()])
    source = h5_test.keys()
    source = tuple([str(s) for s in source]) #converting from unicode
    test_data = VectorSpacesDatasetNoCheck(
        dataset_size=h5_test.values()[0].shape[0],
        data=tuple(h5_test.values()),
        data_specs=(space, source))

    assert len(h5_train.attrs['filestems']) == 100
    assert len(h5_test.attrs['filestems']) == 21
    assert len(set(list(h5_train.attrs['filestems']) + list(h5_test.attrs['filestems']))) == 121
    #converting makes the file bigger by factor 4... I *think* the iterator will convert to
    #config.floatX anyway!
    #assert h5_train.values()[0][...].dtype == 'float32'
    #assert h5_train.values()[1][...].dtype == 'float32'

    it = train_data.iterator(mode='sequential', batch_size=2)
    a,b = it.next()
    assert a.dtype == b.dtype and a.dtype == config.floatX

    transformer = WindowAndFlipTransformer((200,200), center=False, flip=True)
    #transformer.set_input_space(space)
    windowed = VectorSpacesTransformerDataset(train_data, transformer=transformer)

    it2 = windowed.iterator(mode='sequential', batch_size=2, data_specs=(transformer.output_space,source))
    a2,b2 = it2.next()
    it2 = windowed.iterator(mode='sequential', batch_size=2)
    a2,b2 = it2.next()
    import matplotlib.pyplot as pp
    #pp.figure(); pp.imshow(a[1,...,0])
    #pp.figure(); pp.imshow(b[1,...,:]/255.)
    pp.figure(); pp.imshow(a2[1,...,0])
    pp.figure(); pp.imshow(b2[1,...,:]/255.)


    transformer2 = GCNTransformer(scale=1., subtract_mean=True,
                                 divide_by='std',
                                 sqrt_bias=0.,
                                 min_divisor=1e-8)
    #transformer2.set_input_space(transformer.get_output_space())
    gcn_ed = VectorSpacesTransformerDataset(windowed, transformer=transformer2)
    it3 = gcn_ed.iterator(mode='sequential', batch_size=2)
    a3, b3 = it3.next()
    #import matplotlib.pyplot as pp
    #pp.figure(); pp.imshow(a3[1,...,0])
    #pp.figure(); pp.imshow(b3[1,...,:])

    gcn_ed_restricted = VectorSpacesTransformerDataset(
        windowed, transformer=transformer2,
        restrict_to_sources=['images_16', 'contour_groundTruth'])
    it4 = gcn_ed_restricted.iterator(mode='sequential', batch_size=2)
    a4, b4 = it4.next()
    import matplotlib.pyplot as pp
    pp.figure(); pp.imshow(a4[1,...,0])
    pp.figure(); pp.imshow(b4[1,...,:])

    gcn_ed_restricted = VectorSpacesTransformerDataset(
        windowed, transformer=transformer2,
        restrict_to_sources=['images_16'])
    it5 = gcn_ed_restricted.iterator(mode='sequential', batch_size=2)
    a5, b5 = it5.next()
    import matplotlib.pyplot as pp
    pp.figure(); pp.imshow(a5[1,...,0]/ 255)
    pp.figure(); pp.imshow(b5[1,...,:])


    patch_transformer = WindowAndFlipTransformer((128,128), center=False, flip=True)
    patches = PatchExtractorDataset(test_data, patch_transformer, 2, 5)
    it6 = patches.iterator(mode='sequential', batch_size=100)
    a6, b6 = it6.next()
    pp.figure(); pp.imshow(a6[1,...,0]/ 255)
    pp.figure(); pp.imshow(b6[1,...,:] / 255)

    target_transformer = WindowAndFlipTransformer((1,1), center=True, flip=False)

    one_zero_transformer = FunctionTransformer(lambda x: x / 255.)
    one_zero = VectorSpacesTransformerDataset(gcn_ed_restricted, one_zero_transformer,
                                              restrict_to_sources=['contour_groundTruth'])

    it7 = one_zero.iterator(mode='sequential', batch_size=2)
    a7, b7 = it7.next()


    with_targets = VectorSpacesTransformerDataset(
        one_zero, transformer=target_transformer,
        restrict_to_sources=['contour_groundTruth'])
    it8 = with_targets.iterator(mode='sequential', batch_size=2)
    a8, b8 = it8.next()

    def rescale(arr):
        rval = arr - arr.min()
        rval = rval / (rval.max())
        return rval

    pad_transformer = PadTransformer([2,4,6,8])
    padded = VectorSpacesTransformerDataset(one_zero, pad_transformer)
    it9 = padded.iterator(mode='sequential', batch_size=2)
    a9, b9 = it9.next()
    pp.figure(); pp.imshow(rescale(a9[1,...,0]))
    pp.figure(); pp.imshow(rescale(b9[1,...,:]))



    #todo: should check that the targets are correct -- but how to retrieve both centered
    #targets and original ground truth?... maybe just hope for the best.
    #also, need to divide by 255... do I really need separate transformer?