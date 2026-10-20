import os
import torch
import scipy.ndimage
import numpy as np
import tifffile
from tqdm import trange
import fastremap

import logging
dynamics_logger = logging.getLogger(__name__)


def _extend_centers_gpu(neighbors, centers, isneighbor, Ly, Lx, n_iter=200, device=torch.device('cuda')):
    """ runs diffusion on GPU to generate flows for training images or quality control
    
    neighbors is 9 x pixels in masks, 
    centers are mask centers, 
    isneighbor is valid neighbor boolean 9 x pixels
    
    """
    if device is not None:
        device = device
    nimg = neighbors.shape[0] // 9
    pt = torch.from_numpy(neighbors).to(device)
    
    T = torch.zeros((nimg,Ly,Lx), dtype=torch.double, device=device)
    meds = torch.from_numpy(centers.astype(int)).to(device).long() 
    isneigh = torch.from_numpy(isneighbor).to(device)
    for i in range(n_iter):
        T[:, meds[:,0], meds[:,1]] +=1 # 세포 중심에 열 주입
        Tneigh = T[:, pt[:,:,0], pt[:,:,1]]
        Tneigh *= isneigh # 다른 마스크로 새는 열 차단
        T[:, pt[0,:,0], pt[0,:,1]] = Tneigh.mean(axis=1) # 이웃으로 확산
    
    T = torch.log(1.+ T)
    # gradient positions
    grads = T[:, pt[[2,1,4,3],:,0], pt[[2,1,4,3],:,1]] # 상,하,우,좌
    dy = grads[:,0] - grads[:,1]
    dx = grads[:,2] - grads[:,3]

    mu_torch = np.stack((dy.cpu().squeeze(), dx.cpu().squeeze()), axis=-2)
    return mu_torch


def masks_to_flows_gpu(masks, device=None):
    """ convert masks to flows using diffusion from center pixel
    Center of masks where diffusion starts is defined using COM
    Parameters
    -------------
    masks: int, 2D or 3D array
        labelled masks 0=NO masks; 1,2,...=mask labels
    Returns
    -------------
    mu: float, 3D or 4D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].
    mu_c: float, 2D or 3D array
        for each pixel, the distance to the center of the mask 
        in which it resides 
    """
    if device is None:
        device = torch.device('cuda')

    
    Ly0,Lx0 = masks.shape
    Ly, Lx = Ly0+2, Lx0+2

    masks_padded = np.zeros((Ly, Lx), np.int64)
    masks_padded[1:-1, 1:-1] = masks

    # get mask pixel neighbors
    y, x = np.nonzero(masks_padded)
    neighborsY = np.stack((y, y-1, y+1, 
                           y, y, y-1, 
                           y-1, y+1, y+1), axis=0)
    neighborsX = np.stack((x, x, x, 
                           x-1, x+1, x-1, 
                           x+1, x-1, x+1), axis=0)
    neighbors = np.stack((neighborsY, neighborsX), axis=-1)

    # get mask centers
    slices = scipy.ndimage.find_objects(masks)
    
    centers = np.zeros((masks.max(), 2), 'int')
    for i,si in enumerate(slices):
        if si is not None:
            sr,sc = si
            ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
            yi,xi = np.nonzero(masks[sr, sc] == (i+1))
            yi = yi.astype(np.int32) + 1 # add padding
            xi = xi.astype(np.int32) + 1 # add padding
            ymed = np.median(yi)
            xmed = np.median(xi)
            imin = np.argmin((xi-xmed)**2 + (yi-ymed)**2)
            xmed = xi[imin]
            ymed = yi[imin]
            centers[i,0] = ymed + sr.start 
            centers[i,1] = xmed + sc.start

    # get neighbor validator (not all neighbors are in same mask)
    neighbor_masks = masks_padded[neighbors[:,:,0], neighbors[:,:,1]]
    isneighbor = neighbor_masks == neighbor_masks[0]
    ext = np.array([[sr.stop - sr.start + 1, sc.stop - sc.start + 1] for sr, sc in slices])
    n_iter = 2 * (ext.sum(axis=1)).max()
    # run diffusion
    mu = _extend_centers_gpu(neighbors, centers, isneighbor, Ly, Lx, 
                             n_iter=n_iter, device=device)

    # normalize
    mu /= (1e-20 + (mu**2).sum(axis=0)**0.5)

    # put into original image
    mu0 = np.zeros((2, Ly0, Lx0))
    mu0[:, y-1, x-1] = mu
    mu_c = np.zeros_like(mu0)
    return mu0, mu_c

def masks_to_flows(masks, use_gpu=False, device=None):
    """ convert masks to flows using diffusion from center pixel

    Center of masks where diffusion starts is defined to be the 
    closest pixel to the median of all pixels that is inside the 
    mask. Result of diffusion is converted into flows by computing
    the gradients of the diffusion density map. 

    Parameters
    -------------

    masks: int, 2D array
        labelled masks 0=NO masks; 1,2,...=mask labels

    Returns
    -------------

    mu: float, 3D array
        flows in Y = mu[0], flows in X = mu[1].

    mu_c: float, 2D array
        for each pixel, the distance to the center of the mask
        in which it resides

    """
    # exception handling
    if masks.max() == 0:
        dynamics_logger.warning('empty masks!')
        return np.zeros((2, *masks.shape), 'float32')

    mu, mu_c = masks_to_flows_gpu(masks, device=device)
    return mu

def labels_to_flows(labels, files=None, use_gpu=False, device=None, redo_flows=False):
    """ convert labels (list of masks or flows) to flows for training model 

    if files is not None, flows are saved to files to be reused

    Parameters
    --------------

    labels: list of ND-arrays
        labels[k] is a raw 2D (Ly,Lx) mask, or an already-computed [4 x Ly x Lx]
        flow array ([mask, cellprob, dY, dX]) if flows were precomputed.

    Returns
    --------------

    flows: list of [4 x Ly x Lx] arrays
        flows[k][0] is labels[k], flows[k][1] is cell probability,
        flows[k][2] is Y flow, flows[k][3] is X flow

    """
    nimg = len(labels)
    if labels[0].ndim < 3:
        labels = [labels[n][np.newaxis,:,:] for n in range(nimg)]

    if labels[0].shape[0] == 1 or labels[0].ndim < 3 or redo_flows: # flows need to be recomputed
        
        dynamics_logger.info('computing flows for labels')
        
        # compute flows; labels are fixed here to be unique, so they need to be passed back
        # make sure labels are unique!
        labels = [fastremap.renumber(label, in_place=True)[0] for label in labels]
        veci = [masks_to_flows(labels[n][0],use_gpu=use_gpu, device=device) for n in trange(nimg)]
        
        # concatenate labels, distance transform, vector flows, heat (boundary and mask are computed in augmentations)
        flows = [np.concatenate((labels[n], labels[n]>0.5, veci[n]), axis=0).astype(np.float32)
                    for n in range(nimg)]
        if files is not None: #caching
            for flow, file in zip(flows, files):
                file_name = os.path.splitext(file)[0]
                tifffile.imwrite(file_name+'_flows.tif', flow)
    else:
        dynamics_logger.info('flows precomputed')
        flows = [labels[n].astype(np.float32) for n in range(nimg)]
    return flows


def labels_to_flows_3d(labels, files=None, use_gpu=False, device=None, redo_flows=False):
    """ project-specific wrapper (not from cellpose_v1) -- applies labels_to_flows'
    per-slice 2D diffusion (no dZ, no cross-z coupling) independently to every
    z-slice of a 3D mask volume, then stacks the results back into one 3D flow
    volume per patch. This is how a 3D U-Net that only refines (dY,dX,cellprob)
    using cross-z image context (but never predicts dZ) gets its training targets --
    z-linking itself is left to IoU-based stitching at inference time, not to
    anything learned here.

    labels: list of ND-arrays
        labels[k] is a raw 3D (Z,Ly,Lx) mask volume, or an already-computed
        (4,Z,Ly,Lx) flow volume ([mask,cellprob,dY,dX]) if flows were precomputed.
        (ndim 3 vs 4 is what tells the two apart -- unambiguous, unlike the 2D
        case's shape[0]==1 heuristic.)

    Returns
    --------------

    flows: list of [4 x Z x Ly x Lx] arrays
        flows[k][0] is labels[k], flows[k][1] is cell probability,
        flows[k][2] is Y flow, flows[k][3] is X flow -- each z-slice computed and
        renumbered independently, so mask ids are NOT consistent across z.
    """
    nimg = len(labels)

    if labels[0].ndim == 4 and not redo_flows:
        dynamics_logger.info('flows precomputed')
        return [labels[n].astype(np.float32) for n in range(nimg)]

    flows = []
    for n in trange(nimg):
        vol = labels[n]   # (Z,Ly,Lx)
        Lz = vol.shape[0]
        per_slice = labels_to_flows([vol[z] for z in range(Lz)], files=None,
                                     use_gpu=use_gpu, device=device, redo_flows=redo_flows)
        flow_vol = np.stack(per_slice, axis=1)   # (4,Z,Ly,Lx)
        flows.append(flow_vol)
        if files is not None:   # caching
            file_name = os.path.splitext(files[n])[0]
            tifffile.imwrite(file_name + '_flows.tif', flow_vol)

    return flows
