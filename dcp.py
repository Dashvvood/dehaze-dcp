import numpy as np
from typing import List, Optional, Union
import scipy.ndimage
import scipy.sparse
from scipy.sparse.linalg import spsolve

def get_dark_channel(img: np.ndarray, patch_size: tuple[int, int]=(15,15)) -> np.ndarray:
    if len(img.shape) == 3 and img.shape[-1] == 3:
        img_min = np.min(img, axis=-1)
    elif len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[-1] == 1):
        img_min = img.copy()
    else:
        raise NotImplementedError
    
    # ## keep the same output shape
    # img_padding = np.pad(img_min, 
    #                  ((patch_size // 2, patch_size // 2),
    #                  (patch_size // 2, patch_size // 2)),
    #                  mode='edge')
    
    # ## window min filter
    # dc = np.empty_like(img_min)
    # for i, j in np.ndindex(img_min.shape):
    #     dc[i, j, ...] = np.min(img_padding[i:i+patch_size, j:j+patch_size, ...])
    
    return scipy.ndimage.minimum_filter(img_min, patch_size, mode='nearest')


def get_atmos_light(im, dc, top_ratio:float=1e-3) -> Union[float, np.ndarray]:
    """
    average of the top-intensity pixels
    """
    numpix = max(int(dc.shape[0] * dc.shape[1] * top_ratio), 1)
    
    dc_flatten = dc.flatten()
    indices = np.argsort(dc_flatten)[-numpix:]
    mask = np.full_like(dc_flatten, False)
    mask[indices] = True

    if len(im.shape) == 3:
        mask = np.reshape(mask, dc.shape)[:, :, np.newaxis]
    elif len(im.shape) == 2:
        mask = np.reshape(mask, dc.shape)
    else:
        raise NotImplementedError
    
    res = mask * im
    res = np.sum(res, axis=(0,1)) / numpix

    return res


def get_tilde_t(im, A, omega=0.95, **kwarg):
    while len(A.shape) < len(im.shape):
        A = A[np.newaxis, :]
    return 1 - omega * get_dark_channel(im / A, **kwarg)



def get_laplace_matting_matrix(I:np.ndarray, consts:np.ndarray=None, eps=1e-7, win_size:int=1):
    """
    The original version is offered by Levin matlab code
    """
    h, w, c = I.shape
    img_size = h * w
    neb_size = (win_size * 2 + 1) ** 2

    ## the verse of "mask"
    if consts is not None:
        consts = scipy.ndimage.binary_erosion(consts, structure=np.ones((win_size * 2 + 1, win_size * 2 + 1)))
        tlen = np.sum(1 - consts[win_size:h-win_size, win_size:w-win_size]) * (neb_size ** 2)
    else:
        tlen = (h-2*win_size) * (w-2*win_size) * neb_size ** 2

    indsM = np.arange(0, img_size).reshape(h, w)
    row_inds = np.zeros(tlen, dtype=int)
    col_inds = np.zeros(tlen, dtype=int)
    vals = np.zeros(tlen)
    LEN = 0

    for j in range(win_size, w - win_size):
        for i in range(win_size, h - win_size):
            if consts is not None and consts[i, j]:
                continue

            win_inds = indsM[i - win_size:i + win_size + 1, j - win_size: j + win_size + 1].flatten()
            winI = I[i - win_size: i + win_size + 1, j - win_size: j + win_size + 1].reshape(neb_size, c)
            win_mu = np.mean(winI, axis=0).reshape(3, 1)
            win_var = np.linalg.inv(winI.T @ winI / neb_size - (win_mu@win_mu.T) + eps / neb_size * np.eye(c))
            winI = winI - np.tile(win_mu.T, (neb_size, 1))
            tvals = (1 + (winI @ win_var) @ winI.T) / neb_size

            row_inds[LEN:LEN + neb_size**2] = np.tile(win_inds, (neb_size, 1)).flatten()
            col_inds[LEN:LEN + neb_size**2] = np.repeat(win_inds, neb_size).flatten()
            vals[LEN:LEN + neb_size**2] = tvals.flatten()

            LEN += neb_size**2   

    vals = vals[:LEN]

    row_inds = row_inds[:LEN]
    col_inds = col_inds[:LEN]

    A = scipy.sparse.coo_matrix((vals, (row_inds, col_inds)), shape=(img_size, img_size))
    
    sumA = np.array(np.sum(A, axis=1)).squeeze()

    return scipy.sparse.diags(sumA, 0, (img_size, img_size)) - A


def guided_filter(I, p, ks:tuple[int, int]=(3,3), eps=1e-7):
    filter_mean = np.ones(ks)
    filter_mean /= np.sum(filter_mean)
    
    mean_I = np.convolve(I, filter_mean, mode="same")
    mean_p = np.convolve(p, filter_mean, mode="same")
    corr_Ip = np.convolve(I * p , filter_mean, mode="same")
    corr_I = np.convolve(I * I , filter_mean, mode="same")

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = np.convolve(a, filter_mean, mode="same")
    mean_b = np.convolve(b, filter_mean, mode="same")

    return mean_a * I + mean_b


def get_t(L, tilde_t, lam=1e-4, ):
    t = spsolve(L + lam * scipy.sparse.diags([1] * L.shape[0], 0), lam * tilde_t.flatten())
    return t.reshape(tilde_t.shape)

def get_J(I, A, t, t0=0.1):
    while len(A.shape) < len(I.shape):
        A = A[np.newaxis, ...]
    t = np.clip(t, a_min=t0, a_max=1)
    while len(t.shape) < len(I.shape):
        t = t[..., np.newaxis]
    return (I - A) / t + A


def dehaze():
    pass