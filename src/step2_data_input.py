import os
import cv2
import tifffile


def imread(filename):
    ext = os.path.splitext(filename)[-1]
    if ext== '.tif' or ext=='tiff':
        img = tifffile.imread(filename)
        return img
    else:
        try:
            img = cv2.imread(filename, -1) #cv2.LOAD_IMAGE_ANYDEPTH
            if img.ndim > 2:
                img = img[..., [2,1,0]]
            return img
        except Exception as e:
            print('ERROR: could not read file, %s'%e)
            return None

def get_label_files(image_names, mask_filter, imf=None):
    nimg = len(image_names)
    label_names0 = [os.path.splitext(image_names[n])[0] for n in range(nimg)]

    if imf is not None and len(imf) > 0:
        label_names = [label_names0[n][:-len(imf)] for n in range(nimg)]
    else:
        label_names = label_names0

    # check for flows
    if os.path.exists(label_names0[0] + '_flows.tif'):
        flow_names = [label_names0[n] + '_flows.tif' for n in range(nimg)]
    else:
        flow_names = [label_names[n] + '_flows.tif' for n in range(nimg)]
    if not all([os.path.exists(flow) for flow in flow_names]):
        print('not all flows are present, running flow generation for all images')
        flow_names = None

    # check for masks
    if os.path.exists(label_names[0] + mask_filter + '.tif'):
        label_names = [label_names[n] + mask_filter + '.tif' for n in range(nimg)]
    elif os.path.exists(label_names[0] + mask_filter + '.tiff'):
        label_names = [label_names[n] + mask_filter + '.tiff' for n in range(nimg)]
    elif os.path.exists(label_names[0] + mask_filter + '.png'):
        label_names = [label_names[n] + mask_filter + '.png' for n in range(nimg)]
    # todo, allow _seg.npy
    #elif os.path.exists(label_names[0] + '_seg.npy'):
    #    io_logger.info('labels found as _seg.npy files, converting to tif')
    else:
        raise ValueError('labels not provided with correct --mask_filter')
    if not all([os.path.exists(label) for label in label_names]):
        raise ValueError('labels not provided for all images in train and/or test set')

    return label_names, flow_names
