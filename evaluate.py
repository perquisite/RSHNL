import numpy as np
import scipy
import scipy.spatial
def fx_calc_map_label(image, text, label, k = 0, dist_method='COS'):
  if dist_method == 'L2':
    dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  ord = dist.argsort()
  numcases = dist.shape[0]
  if k == 0:
    k = numcases
  res = []
  for i in range(numcases):
    order = ord[i]
    p = 0.0
    r = 0.0
    for j in range(k):
      if label[i] == label[order[j]]:
        r += 1
        p += (r / (j + 1))
    if r > 0:
      res += [p / r]
    else:
      res += [0]
  return np.mean(res)

def fx_calc_map_multilabel(image, text, label, k = 0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(image, text, metric)
    ord = dist.argsort()
    
    numcases = dist.shape[0]
    if k == 0:
      k = numcases
    res = []
    for i in range(dist.shape[0]):
        order = ord[i].reshape(-1)[0: dist.shape[0]]

        tmp_label = (np.dot(label[order], label[i]) > 0)
        if tmp_label.sum() > 0:
            prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
            total_pos = float(tmp_label.sum())
            if total_pos > 0:
                res += [np.dot(tmp_label, prec) / total_pos]
    return np.mean(res)