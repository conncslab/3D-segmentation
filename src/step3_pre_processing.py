import numpy as np
import warnings
import cv2

import logging
transforms_logger = logging.getLogger(__name__)

import edt, fastremap


# needs to have a wider range to avoid weird effects with few cells in frame
# also turns out previous formulation can give negative numbers, messes up log operations etc.
def normalize99(Y,lower=0.01,upper=99.99,omni=False,value_range=None):
    """ normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile

    value_range: if given a (low, high) tuple, normalizes with this fixed value
    instead of computing the image's own percentile (when a small patch is cut out
    of a big volume, normalizing by the patch's own percentile makes even
    signal that would be negligible dataset-wide look relatively bright within that
    patch, so the global range is used as-is instead).
    """
    X = Y.copy()
    if value_range is not None:
        x01, x99 = value_range
    else:
        x01 = np.percentile(X, 1)
        x99 = np.percentile(X, 99)
    X = (X - x01) / (x99 - x01)
    return X

def normalize_img(img, axis=-1, invert=False, omni=False, percentile_range=None):
    """ normalize each channel of the image so that so that 0.0=1st percentile
    and 1.0=99th percentile of image intensities

    optional inversion

    Parameters
    ------------

    img: ND-array (at least 3 dimensions)

    axis: channel axis to loop over for normalization

    percentile_range: if given a (low, high) tuple, normalizes with this fixed range
    instead of each channel's own percentile (same purpose as normalize99's value_range).

    Returns
    ---------------

    img: ND-array, float32
        normalized image of same size

    """
    if img.ndim<3:
        error_message = 'Image needs to have at least 3 dimensions'
        transforms_logger.critical(error_message)
        raise ValueError(error_message)

    img = img.astype(np.float32)
    img = np.moveaxis(img, axis, 0)
    for k in range(img.shape[0]):
        if percentile_range is not None:
            i1, i99 = percentile_range
        else:
            # ptp can still give nan's with weird images
            i99 = np.percentile(img[k],99)
            i1 = np.percentile(img[k],1)
        if i99 - i1 > +1e-3: #np.ptp(img[k]) > 1e-3:
            img[k] = normalize99(img[k],omni=omni,value_range=percentile_range)
            if invert:
                img[k] = -1*img[k] + 1
        else:
            img[k] = 0
    img = np.moveaxis(img, 0, axis)
    return img

def reshape(data, channels=[0,0], chan_first=False):
    """ reshape data using channels

    Parameters
    ----------

    data : numpy array that's (Z,) Ly x Lx x nchan
        this pipeline always deals with 3D volumes: (Z, Ly, Lx) if single-channel,
        or (Z, Ly, Lx, nchan) if the channel axis is already attached

    channels : list of int of length 2 (optional, default [0,0])
        First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
        Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
        For instance, to train on grayscale images, input [0,0]. To train on images with cells
        in green and nuclei in blue, input [2,3].

    invert : bool
        invert intensities

    Returns
    -------
    data : numpy array that's (Z,) Ly x Lx x nchan (if chan_first==False)

    """
    data = data.astype(np.float32)
    if data.ndim == 3:
        data = data[..., np.newaxis]   # (Z,Ly,Lx) -> (Z,Ly,Lx,1)

    # if use grayscale image
    if data.shape[-1]==1:
        # add zeros channel(first channel = grayscale image data, second channel = all zero data)
        # data = np.concatenate((data, np.zeros_like(data)), axis=-1)

        # leave it as one channel instead of forcing it to 2, under the assumption that only grayscale images are handled
        data = data

    # no grayscale image
    # the branch below is unused since only grayscale images are handled right now
    else:
        if channels[0]==0:
            data = data.mean(axis=-1, keepdims=True)
            data = np.concatenate((data, np.zeros_like(data)), axis=-1)
        else:
            chanid = [channels[0]-1]
            if channels[1] > 0:
                chanid.append(channels[1]-1)
            data = data[...,chanid]
            for i in range(data.shape[-1]):
                if np.ptp(data[...,i]) == 0.0:
                    if i==0:
                        warnings.warn("chan to seg' has value range of ZERO")
                    else:
                        warnings.warn("'chan2 (opt)' has value range of ZERO, can instead set chan2 to 0")
            if data.shape[-1]==1:
                data = np.concatenate((data, np.zeros_like(data)), axis=-1)
    if chan_first:
        if data.ndim==4:
            data = np.transpose(data, (3,0,1,2)) # (Z,Ly,Lx,C) -> (C, Z,Ly,Lx)
        else:
            data = np.transpose(data, (2,0,1))
    return data

def reshape_and_normalize_data(train_data, test_data=None, channels=None, normalize=True, omni=False,
                               percentile_range=None):
    """ inputs converted to correct shapes for *training* and rescaled so that 0.0=1st percentile
    and 1.0=99th percentile of image intensities in each channel

    Parameters
    --------------

    train_data: list of ND-arrays, float
        list of training volumes of size [Z x Ly x Lx] or [Z x Ly x Lx x nchan]

    test_data: list of ND-arrays, float (optional, default None)
        list of testing volumes of size [Z x Ly x Lx] or [Z x Ly x Lx x nchan]

    channels: list of int of length 2 (optional, default None)
        First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
        Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
        For instance, to train on grayscale images, input [0,0]. To train on images with cells
        in green and nuclei in blue, input [2,3].

    normalize: bool (optional, True)
        normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

    percentile_range: (low, high) tuple (optional, default None)
        if given, normalizes with this fixed global range instead of each image's own percentile (see normalize_img).

    Returns
    -------------

    train_data: list of ND-arrays, float
        list of training volumes of size [2 x Z x Ly x Lx]

    test_data: list of ND-arrays, float (optional, default None)
        list of testing volumes of size [2 x Z x Ly x Lx]

    run_test: bool
        whether or not test_data was correct size and is useable during training

    """

    # if training data is less than 2D
    run_test = False
    for test, data in enumerate([train_data, test_data]):
        if data is None:
            return train_data, test_data, run_test
        nimg = len(data)
        for i in range(nimg):
            if channels is not None:
                data[i] = reshape(data[i], channels=channels, chan_first=True)
            if data[i].ndim < 3:
                data[i] = data[i][np.newaxis,:,:]
            if normalize:
                data[i] = normalize_img(data[i], axis=0, omni=omni, percentile_range=percentile_range)
        nchan = [data[i].shape[0] for i in range(nimg)]
    run_test = True
    return train_data, test_data, run_test

def reshape_train_test(train_data, train_labels, test_data, test_labels, channels, normalize=True, omni=False,
                       percentile_range=None):

    """ check sizes and reshape train and test data for training (always 3D volumes in this pipeline) """

    nimg = len(train_data)
    #=====================Data check(start)====================

    # check that arrays are correct size
    if nimg != len(train_labels):
        error_message = 'train data and labels not same length'
        transforms_logger.critical(error_message)
        raise ValueError(error_message)
        return
    if train_labels[0].ndim < 2 or train_data[0].ndim < 2:
        error_message = 'training data or labels are not at least two-dimensional'
        transforms_logger.critical(error_message)
        raise ValueError(error_message)
        return

    if train_data[0].ndim > 4:
        error_message = 'training data is more than four-dimensional (should be Z x Ly x Lx (x nchan))'
        transforms_logger.critical(error_message)
        raise ValueError(error_message)
        return

    # check if test_data correct length
    if not (test_data is not None and test_labels is not None and
            len(test_data) > 0 and len(test_data)==len(test_labels)):
        test_data = None

    #=====================Data check(end)====================

    #=====================Main code==========================
    # make data correct shape and normalize it so that 0 and 1 are 1st and 99th percentile of data
    train_data, test_data, run_test = reshape_and_normalize_data(train_data, test_data=test_data,
                                                                 channels=channels, normalize=normalize, omni=omni,
                                                                 percentile_range=percentile_range)
    #=====================Main code==========================

    # exception handling
    if train_data is None:
        error_message = 'training data do not all have the same number of channels'
        transforms_logger.critical(error_message)
        raise ValueError(error_message)
        return

    if not run_test:
        test_data, test_labels = None, None

    return train_data, train_labels, test_data, test_labels, run_test


def preprocess(train_data, train_labels, test_data, test_labels, channels=[2, 1], percentile_range=None):
    """ channels = [cell channel, nucleus channel]; cellpose cyto dataset => channels = [2, 1]
    Currently only grayscale images are handled, so the channels value itself isn't actually used. Left in for future extension.
    Always deals with (Z,Y,X) 3D volumes only, so the reshape family of functions defaults to handling 3D.

    percentile_range: if given a (low, high) tuple, normalizes with this fixed global
    range instead of each image's own percentile (the HITL pipeline passes in the
    global range it pre-scanned in step1).
    """
    return reshape_train_test(train_data, train_labels, test_data, test_labels, channels,
                              percentile_range=percentile_range)


def pad_to_3channel(data):
    """CPSAM (SAM ViT-L backbone) requires 3-channel input because patch_embed is
    fixed as nn.Conv2d(3, ...). The pipeline handles grayscale (1-channel) data, so the
    missing 2 channels are filled with 0 -- zero-padding rather than duplication, to follow
    cellpose's own channel-handling convention (cellpose.transforms.convert_image) as-is.

    data: list of [1, Z, Y, X] (or [C,Z,Y,X], C<3) normalized volumes
    returns: list of [3, Z, Y, X]
    """
    out = []
    for d in data:
        c = d.shape[0]
        if c >= 3:
            out.append(d[:3])
            continue
        pad = np.zeros((3 - c,) + d.shape[1:], dtype=d.dtype)
        out.append(np.concatenate([d, pad], axis=0))
    return out
