import numpy as np
import numba
from sklearn.neighbors import NearestNeighbors
from skimage.color import rgb2lab
from skimage.segmentation import felzenszwalb, slic
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage.measurements import center_of_mass


def quantize(img, k=12):
    img = np.asarray(img)
    img = (img * (k/256)).astype(np.uint32)
    colors = (img * [k**2, k, 1]).sum(2)
    lut = np.stack([np.arange(k**3)//k**2,
                   (np.arange(k**3)//k)%k,
                   np.arange(k**3)%k], 1)
    hst, bins = np.histogram(colors, np.arange(k**3+1))
    ind = np.argsort(hst)[::-1]
    cdf = hst[ind].cumsum() / hst.sum()
    idx = np.searchsorted(cdf, 0.95, side='right') + 1
    ind = ind[:idx]
    colors = lut[bins[ind]].reshape(-1, 3)
    probs = hst[ind] / hst[ind].sum()
    ind = cdist(img.reshape(-1, 3), colors, 'sqeuclidean').argmin(1)
    ind = ind.reshape(img.shape[:2])
    colors = colors * int(256/k)
    colors = colors.astype(np.uint8)

    return ind, colors, probs


def smoothed_saliency(ind, colors, probs):
    lab = rgb2lab(colors[None].astype(np.uint8)).squeeze()
    #lab_dist = np.square(lab[...,None] - lab.T).sum(1)
    lab_dist = squareform(pdist(lab, 'sqeuclidean'))
    s = (lab_dist * probs).sum(1)
    s = (s - s.min()) / (s.max() - s.min())
    m = lab.shape[0] // 4
    dist, nn = NearestNeighbors(m).fit(lab).kneighbors()
    T = dist.sum(1)
    sp = ((T[:,None] - dist) * s[nn]).sum(1) / ((m-1)*T)

    return sp


def hc_saliency(img):
    ind, colors, probs = quantize(img)
    sal = smoothed_saliency(ind, colors, probs)
    sal = (sal - sal.min()) / (sal.max() - sal.min())
    sal_img = sal[ind]
    return sal_img


@numba.njit(parallel=True)
def region_saliency(lab_dist, histo, reg_sizes, reg_dist, v=0.4):
    n, k = histo.shape
    S = np.zeros(n)
    for i in numba.prange(n):
        for j in range(i+1, n):
            d = 0
            for c1 in range(k):
                for c2 in range(k):
                    d += histo[i, c1] * histo[j, c2] * lab_dist[c1, c2]
            w = np.exp(-reg_dist[i, j] / v)
            S[i] += (w * reg_sizes[j] * d)
            S[j] += (w * reg_sizes[i] * d)
    return S


def rc_saliency(img):
    ind, colors, probs = quantize(img)
    lab = rgb2lab(colors[None].astype(np.uint8)).squeeze()
    lab_dist = squareform(pdist(lab, 'sqeuclidean'))
    # region segmentation
    #regions = felzenszwalb(np.asarray(img), max(img.size)/2)
    regions = slic(np.asarray(img), 200, slic_zero=True)
    # region histograms
    histo = np.zeros((regions.max()+1, colors.shape[0]))
    np.add.at(histo, (regions, ind), 1)
    reg_sizes = np.bincount(regions.ravel())
    # region centroid distances
    centroids = center_of_mass(regions+1, regions, np.arange(regions.max()+1))
    centroids = np.array(centroids) / ind.shape
    reg_dist = squareform(pdist(centroids, 'euclidean'))

    reg_sal = region_saliency(lab_dist, histo, reg_sizes, reg_dist)
    reg_sal /= reg_sal.max()
    sal_img = reg_sal[regions]

    return sal_img

