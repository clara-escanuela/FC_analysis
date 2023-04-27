import numpy as np

def bin(x,y,bins=10,_range=None):
  """
  Bins x and returns lists of the y-values inside each bin.

  Parameters
  ----------
  x: array-like
    Variable that is binned.
  y: array-like
    Variable that is sorted according to the binning of x.
  bins: integer or array-like
    Number of bins or array of lower bin edges + last high bin edge.
  range: tuple, lenght of 2 (optional)
    If range is set, only (x,y) pairs are used where x is inside the range.
    Ignored, if bins is an array.

  Returns
  -------
  yBins: list of lists
    List of y-values which correspond to the x-bins.
  xegdes: array of floats
    Lower bin edges. Has length len(yBins)+1.

  Authors
  -------
  Hans Dembinski <hans.dembinski@kit.edu>
  """

  ys = np.atleast_1d(y)
  xs = np.atleast_1d(x)

  if type(bins) is int:
    if _range is None:
      _range = (min(x),max(x)+np.finfo(float).eps)
    else:
      mask = (_range[0] <= xs) & (xs < _range[1])
      xs = xs[mask]
      ys = ys[mask]
    xedges = np.linspace(_range[0],_range[1],bins+1)
  else:
    xedges = bins
    bins = len(xedges)-1

  binnedys = []
  for i in range(bins):
    binnedys.append(ys[(xedges[i] <= xs) & (xs < xedges[i+1])])

  return binnedys, xedges


def profile(x, y, bins=10, _range=None, sigma_cut=None):
  """
  Computes the (robust) profile of a set of data points.

  Parameters
  ----------
  x,y : array-like
    Input data. The (x,y) pairs are binned according to the x-array,
    while the averages are computed from the y-values inside a x-bin.
  bins : int or array-like, optional
    Defines the number of equal width bins in the given range (10,
    by default). If bins is an array, it is used for the bin edges
    of the profile.
  range : (float,float), optional
    The lower and upper range of the bins.  If not provided, range
    is simply ``(a.min(), a.max())``.  Values outside the range are
    ignored.
  sigma_cut : float, optional
    If sigma_cut is set, outliers in the data are rejected before
    computing the profile. Outliers are detected based on the scaled
    MAD and the median of the distribution of the y's in each bin.
    All data points with |y - median| > sigma_cut x MAD are ignored
    in the computation.

  Returns
  -------
  yavg : array of dtype float
    Returns the averages of the y-values in each bin.
  ystd : array of dtype float
    Returns the standard deviation in each bin. If you want the
    uncertainty of ymean, calculate: yunc = ystd/numpy.sqrt(n-1).
  n : array of dtype int
    Returns the number of events in each bin.
  xedge : array of dtype float
    Returns the bin edges. Beware: it has length(yavg)+1.

  Examples
  --------
  >>> yavg,ystd,n,xedge = profile(np.array([0.,1.,2.,3.]), np.array([0.,1.,2.,3.]), 2)
  >>> print yavg, ystd, n, xedge
  [ 0.5  2.5] [ 0.5  0.5] [2 2] [ 0.   1.5  3. ]

  Authors
  -------
  Hans Dembinski <hans.dembinski@kit.edu>
  """

  y = np.atleast_1d(y)

  n,xedge = np.histogram(x,bins=bins,range=_range)

  if sigma_cut is None:
    ysum  = np.histogram(x,bins=bins,range=_range,weights=y)[0]
    yysum = np.histogram(x,bins=bins,range=_range,weights=y*y)[0]
  else:
    if sigma_cut <= 0:
      raise ValueError("sigma_cut <= 0 detected, has to be positive")
    # sort y into bins
    ybin = bin(x,y,bins,_range)[0]

    if type(bins) is int:
      nbins = bins
    else:
      nbins = len(bins) - 1

    # reject outliers in calculation of avg, std
    ysum  = np.zeros(nbins)
    yysum = np.zeros(nbins)
    for i in range(nbins):
      ymed = np.median(ybin[i])
      ymad = mad(ybin[i])
      for y in ybin[i]:
        if ymad == 0 or abs(y-ymed) < sigma_cut*ymad:
          ysum[i]  += y
          yysum[i] += y*y
        else:
          n[i] -= 1

  mask = n == 0
  n[mask] = 1
  yavg = ysum/n
  ystd = np.sqrt(yysum/n-yavg*yavg)
  yavg[mask] = np.nan
  ystd[mask] = np.nan
  n[mask] = 0

  return yavg,ystd,n,xedge

def mad(a, med=None):
  """
  Calculates the scaled median absolute deviation of a random distribution.

  Parameters
  ----------
  a : array-like
    1-d array of random numbers.

  Returns
  -------
  mad : float
    Scaled median absolute deviation of input sample. The scaling factor
    is chosen such that the MAD estimates the standard deviation of a
    normal distribution.

  Notes
  -----
  The MAD is a robust estimate of the true standard deviation of a random
  sample. It is robust in the sense that its output is not sensitive to
  outliers.

  The standard deviation is usually estimated by the square root of
  the sample variance. Note, that just one value in the sample has to be
  infinite for the sample variance to be also infinite. The MAD still
  provides the desired answer in such a case.

  In general, the sample variance is very sensitive to the tails of the
  distribution and will give undesired results if the sample distribution
  deviates even slightly from a true normal distribution. Many real world
  distributions are not exactly normal, so this is a serious issue.
  Fortunately, this is not the case for the MAD.

  Of course there is a price to pay for these nice features. If the sample is
  drawn from a normal distribution, the sample variance is the more
  efficient estimate of the true width of the Gaussian, i.e. its
  statistical uncertainty is smaller than that of the MAD.

  Examples
  --------
  >>> a = [1.,0.,5.,4.,2.,3.,1e99]
  >>> mad(a)
  2.9652044370112041

  Authors
  -------
  Hans Dembinski <hans.dembinski@kit.edu>
  """

  const = 1.482602218505602 # 1.0/inverse_cdf(3/4) of normal distribution
  if med is None:
    med = np.median(a)
  return const*np.median(np.abs(a-med))
