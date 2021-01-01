import copy
import pywt
import warnings
import numpy as np
from scipy import signal
from filter import QMFFilter


def MirrorFilt(x):
    '''
    Apply (-1)^t modulation
    :param x: 1-d signal. Type numpy.ndarray
    :return: (-1)^(t-1)  * x(t). Type numpy.ndarray
    '''
    t = len(x)
    return np.power(-1, t+1) * pywt.qmf(x)[::-1]


def upsample(x, s = 2):
    n = len(x) * s
    y = np.zeros(n)
    y[0:(n - s + 1):s] = x
    return y


def dyad(j):
    '''
    Index entire jth dyad of 1-d wavelet transform
    :param j: integer
    :return: list of all indices of wavelet coefficients at jth level
    '''
    i = np.arange(start=np.power(2, j),
                stop=np.power(2, j+1), step=1)
    return i


def dyad2HH(j):
    '''
    Index entire jth dyad of 2d wavelet xform in HH
    :param j: integer
    :return: [ix, iy] is the list of all indices of wavelet coefficients at jth level
    '''
    ix = np.arange(start=np.power(2, j),
                   stop=np.power(2, j+1), step=1)
    iy = copy.copy(ix)
    return ix, iy


def dyad2HL(j):
    '''
    Index entire j-th dyad of 2-d wavelet xform in HL(left-bottom corner)
    :param j:integer
    :return:list of all indices of wavelet coeffts at j-th level
    '''
    ix = np.arange(start=np.power(2, j),
                stop=np.power(2, j+1), step=1)
    iy = np.arange(start=0, stop=np.power(2, j), step=1)
    return ix, iy


def dyad2LH(j):
    '''
    Index entire j-th dyad of 2-d wavelet xform in LH(right-top corner)
    :param j:integer
    :return:list of all indices of wavelet coeffts at j-th level
    '''
    ix = np.arange(start=0, stop=2**j, step=1)
    iy = np.arange(start=2**j, stop=2**(j + 1), step=1)
    return ix, iy


def dyad2LL(j):
    '''
    Index entire j-th dyad of 2-d wavelet xform in LL
    :param j:integer
    :return:list of all indices of wavelet coeffts at j-th level
    '''
    ix = np.arange(start=0, stop=2**j, step=1)
    iy = copy.copy(ix)
    return ix, iy


def dyad2ix(j, k):
    '''
    Convert wavelet indexing into linear indexing
    :param j:Resolution Level. j >= 0.
    :param k:Spatial Position. 0 <= k < 2^j
    :return:index in linear 1-d wavelet transform array where
          the (j,k) wavelet coefficient is stored
    '''
    ix = 2**j + k
    return ix


def dyadlength(x):
    '''
    Find length and dyadic length of array
    :param x:1-d signal, array of length n = 2^J (hopefully)
    :return:
    n: length(x)
    J: least power of two greater than n
    '''
    n = len(x)
    J = int(np.ceil(np.log(n) / np.log(2)))
    if 2**J != n:
        warnings.warn("Warning in dyadlength: n != 2^J")
    return n, J


def lshift(x):
    n = len(x)
    y = np.zeros_like(x)
    y[0:(n-1)], y[n-1] = x[1:n], x[0]
    return y


def rshift(x):
    n = len(x)
    y = np.zeros_like(x)
    y[0], y[1:] = x[n - 1], x[0:(n - 1)]
    return y


def ReadSubMatrix(X, I1, subband):
    '''
    Wavelet subband extraction
    :param X: wavelet matrix of tensor type
    :param I1: a submatrix index
    :param subband: can be either one of 'a', 'h', 'v' or 'd'
    :return: a submatrix of X[I,I,n]
    '''
    I2 = 2 * I1
    if subband == 'a':
        D = X[0:I1, 0:I1, :]
    elif subband == 'h':
        D = X[I1:I2, I1:I2, :]
    elif subband == 'v':
        D = X[I1:I2, 0:I1, :]
    else:
        D = X[I1:I2, I1:I2, :]

    Xsub = np.reshape(D, -1)
    return Xsub


def DWT_1D(X, scale, wbase, nom):
    '''
    Forward 1D-wavelet transform
    :param X: a n by T matrix of 1D signals
    :param scale: scale parameter(should be integer)
    :param wbase: wavelet basis('Haar', 'Symmetric', 'Daubechies'...
    :param nom: number of moments
    :return: returns in W[n, T] the wavelet 1D transform of X[n, T]
    along the second dimension(each row reperesents one wavelet signal)
    '''
    qmf = QMFFilter(wbase, nom).generate_filter()
    resol = np.log2(X.shape[1]) - scale
    W = FWT_PO(X, scale, qmf)
    return W, resol, qmf


def FWT_PO(x, L, qmf):
    '''
    Forward wavelet transform(periodized, orthogonal)
    :param x: 1-d signal, length(x) = 2^J
    :param L: Coarsest Level of V_0; L << J
    :param qmf: quadrature mirror filter(orthonormal)
    :return:
    wcoef: 1-d wavelet tranform of x
    '''
    n, J = dyadlength(x)
    wcoef = np.zeros(n)
    beta = x
    for j in range(J - 1, L, -1):
        alpha = DownDyadHi(beta, qmf)
        wcoef[dyad(j)] = alpha
        beta = DownDyadLo(beta, qmf)
    wcoef[0 : 2**L] = beta
    return wcoef


def FWT2_PO(x, L, qmf):
    '''
    2-d MRA wavelet transform(periodized, orthogonal)
    :param x: 2-d image(n by n array, n dyadic)
    :param L: coarse level
    :param qmf: quadrature mirror filter
    :return:
    wc: 2-d wavelet transform
    '''
    n, J = quadlength(x)
    wc = x
    nc = n
    for j in range(J-1, L, -1):
        top = list(range(int(np.floor(nc / 2) + 1), nc))
        bot = list(range(0, int(np.floor(nc / 2))))

        for ix in range(0, nc):
            row = wc[ix, list(range(0, nc))]
            wc[ix, bot] = DownDyadLo(row, qmf)
            wc[ix, top] = DownDyadHi(row, qmf)

        for iy in range(0, nc):
            row = wc[list(range(0, nc)), iy]
            wc[top, iy] = DownDyadHi(row, qmf)
            wc[bot, iy] = DownDyadLo(row, qmf)

        nc /= 2

    return wc


def iconv(f, x):
    '''
    Convolution tool for two-scale transform
    filtering by periodic convolution of x with f
    :param f:filter
    :param x:1-d signal
    :return:filtered result, should be the same length as x
    '''
    n, p = len(x), len(f)
    if p <= n:
        xpadded = np.concatenate((x[(n - p) : n], x))
    else:
        z = np.zeros(p)
        for i in range(p):
            imod = 1 + np.fmod(p*n - p + i-1, n)
            z[i] = x[imod - 1]

        xpadded = np.concatenate((z, x))

    ypadded = signal.filtfilt(f, 1, xpadded)
    y = ypadded[p:(n + p + 1)]

    return y


def aconv(f, x):
    '''
    Convolution tool for two-scale transform.
    Filtering by periodic convolution of x with the time-reverse of f.
    :param f: filter
    :param x: 1-d signal
    :return: the filtered signal, should be the same length as x
    '''
    n, p = len(x), len(f)
    if p < n:
        xpadded = np.concatenate((x, x[0:p]))
    else:
        z = np.zeros(p)
        for i in range(p):
            imod = 1 + np.fmod(i - 1, n)
            z[i] = x[imod - 1]
        xpadded = np.concatenate((x, z))

    fflip = f[::-1]
    ypadded = signal.filtfilt(fflip, 1, xpadded)
    y = ypadded[p:(n + p)]

    return y


def DownDyadHi(x, qmf):
    '''
    High pass Downsampling operator
    :param x: 1-d signal
    :param qmf: filter
    :return: 1-d signal at coarse scale
    '''
    d = iconv(MirrorFilt(qmf), lshift(x))
    n = len(d)
    d = d[list(range(0, n, 2))]
    return d


def DownDyadLo(x, qmf):
    '''
    Low pass Downsampling operator
    :param x: 1-d signal
    :param qmf: filter
    :return: 1-d signal at coarse scale
    '''
    d = aconv(qmf, x)
    n = len(d)
    d = d[list(range(0, n, 2))]
    return d


def UpDyadHi(x, qmf):
    '''
    High-pass upsampling operator
    :param x: 1-d signal at coarser scale
    :param qmf: filter
    :return: 1-d signal at finer scale
    '''
    y = aconv(MirrorFilt(qmf), rshift(upsample(x)))
    return y


def UpDyadLo(x, qmf):
    '''
    Low-pass upsampling operator
    :param x: 1-d signal at coarser scale
    :param qmf: filter
    :return: 1-d signal at finer scale
    '''
    y = iconv(qmf, upsample(x))
    return y


def quadlength(x):
    '''
    Find length and dyadic length of square matrix
    :param x: 2-d image, of shape(n,n). Hopefully, n = 2^J
    :return: n: length(x); J: least power of two greater than n
    '''
    n = x.shape[0]
    p = x.shape[1]
    if p != n:
        warnings.warn("Warning in quadlength, row is not euqal to column")
    k = 1
    J = 0
    while k < n:
        k *= 2
        J += 1
    if k != n:
        warnings.warn("Warning in quadlength: n is not equal to 2^J")
    return n, J


def norm_noise_1d(x, qmf):
    '''
    Estimates noise level. Normalized signal to noise level 1
    :param x: 1-d signal
    :param qmf:quadrature mirror filter
    :return:
    y: 1-d signal, scaled so that wavelet coefficients at finest level
    have median absolute deviation 1
    coef: estimated of 1/sigma
    '''
    u = DownDyadHi(x, qmf)
    s = np.median(np.abs(u))
    if s != 0:
        y = 0.6745 * x / s
        coef = 0.6745 / s
    else:
        y = x
        coef = 1
    return y, coef


def norm_noise_2d(x, qmf):
    '''
    Estimates noise level. Normalize signal to noise level 1
    :param x: 2-d signal
    :param qmf: quadrature mirror filter
    :return:
    y: 2-d signal, scaled so wavelet coefficients
    at the finest level have median absolute deviation 1
    coef: estimation of 1/sigma
    '''
    n, J = quadlength(x)
    wc = x
    nc = n
    top = list(range(int(nc / 2), nc))
    bot = list(range(0, int(np.floor(nc / 2))))

    for ix in range(0, nc):
        row = wc[ix, list(range(0, nc))]
        wc[ix, bot] = DownDyadLo(row, qmf)
        wc[ix, top] = DownDyadHi(row, qmf)

    for iy in range(0, nc):
        row = wc[list(range(0, nc)), iy]
        wc[top, iy] = DownDyadHi(row, qmf)
        wc[bot, iy] = DownDyadLo(row, qmf)

    nc /= 2

    t1, t2 = dyad2HH(J-1)
    t1, t2 = t1 - 1, t2 - 1
    temp = wc[t1, t2]
    s = np.median(np.abs(temp))
    if s != 0:
        y = 0.6745 * x / s
        coef = 0.6745 / s
    else:
        y = x
        coef = 1

    return y, coef


def IDWT_1D(W, resol, qmf):
    '''
    Inverse 1D-wavelet transform
    :param W: n by T matrix of 1D-wavelet coefficients
    :param resol: resolution level
    :param qmf: quadratic mirror filters
    :return:
    X: n by T matrix of 1D signals
    scale: scale parameter, should be an integer
    '''
    scale = np.log2(X.shape[1]) - resol
    X = IWT_PO(W, scale, qmf)
    return X, scale


def IDWT_2D(W, resol, qmf):
    '''
    Inverse 2D wavelet transform
    :param W: n by T matrix of 2D-wavelet coefficients
    :param resol:resolution level
    :param qmf:quadratic mirror filters
    :return:
    X: n by T matrix of image
    scale: scale parameter, should be an integer
    '''
    scale = np.log2(W.shape[1]) - resol
    image = IWT2_PO(W,scale, qmf)
    X = np.transpose(image)
    return X, scale