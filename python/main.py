from filter import QMFFilter
from util import *
import numpy as np
import scipy as sp
from skimage.measure import compare_mse, compare_psnr


def metric(im1, im2):
    '''
    Calculate the PSNR performance of two images
    MSE = \frac{1}{MN} \sum_{i=1}^{M} \sum_{j=1}^{N} [f(i,j) - g(i,j)]^2
    metric = 10 * log_{10}(255^2 / MSE)
    :param im1: the first input image
    :param im2: the second input image
    :return: metric
    '''
    MSE = compare_mse(im1, im2)
    psnr = compare_psnr(im1, im2, 255)
    return MSE, psnr


def im2col(mtx, block_size):
    '''

    :param mtx: the input image
    :param block_size:
    :return:
    '''
    mtx_shape = mtx.shape
    sx = mtx_shape[0] - block_size[0] + 1
    sy = mtx_shape[1] - block_size[1] + 1
    result = np.empty((block_size[0] * block_size[1], sx * sy))
    for i in range(sy):
        for j in range(sx):
            result[:, i * sx + j] = mtx[j:j + block_size[0], i:i + block_size[1]].ravel(order='F')
    return result


def subband_denoise(x_in, noise_sigma, win_size):
    '''
    Subband denoising function
    :param x_in: the input image
    :param noise_sigma: the standard deviation of noise signal.
    The noise is often considered as coming from Gaussian distribution
    :param win_size: the window size
    :return: the image after denoising procedure
    '''
    AA = np.pad(x_in, pad_width=((win_size, win_size),
                                 (win_size, win_size)),
                mode="reflect")
    AA = im2col(AA, [win_size * 2 + 1, win_size * 2 + 1])
    x_in_1d = np.reshape(x_in, -1)
    result_1d = np.zeros_like(AA)

    var_gaussian = np.mean(AA**2, 0) - noise_sigma ** 2
    var_gaussian = np.maximum(var_gaussian, 0)
    sigma_gaussian = np.sqrt(var_gaussian)
    judge = sigma_gaussian != 0

    BB = np.mean(AA, 0)
    y = x_in_1d[judge]

    for i in range(0, AA.shape[0]):
        temp_x = AA[i,:]
        temp_A = temp_x[judge]
        temp_BB = BB[judge]
        temp_sigma = sigma_gaussian[judge]
        temp_result = (y * temp_sigma**2 + temp_A * noise_sigma**2 -
                       temp_BB * noise_sigma**2) / (temp_sigma**2 + noise_sigma**2)
        result_1d_temp = np.zeros_like(result_1d[i,:])
        result_1d_temp[judge] = temp_result
        result_1d[i,:] = result_1d_temp

    result_1d = np.mean(result_1d, 0)
    result = np.reshape(result_1d, [x_in.shape[0], x_in.shape[1]])

    return result


def denoise_kde(x_in, wbase, mom, dwt_level, win_size):
    '''

    :param x_in:
    :param wbase:
    :param mom:
    :param dwt_level:
    :param win_size:
    :return:
    '''
    [nrow, ncol] = x_in.shape
    L = np.log2(ncol) - dwt_level
    qmf = QMFFilter(wbase, mom)
    [temp, coef] = norm_noise_2d(x_in, qmf)
    noise_sigma = 1 / coef

    wx = FWT2_PO(x_in, L, qmf)
    [n, J] = dyadlength(wx)
    ws = wx

    for j in range(J-1, L-1, -1):
        [t1, t2] = dyad2HH(j)
        ws[t1, t2] = subband_denoise(wx[t1, t2], noise_sigma, win_size)
        [t1, t2] = dyad2HL(j)
        ws[t1, t2] = subband_denoise(wx[t1, t2], noise_sigma, win_size)
        [t1, t2] = dyad2LH(j)
        ws[t1, t2] = subband_denoise(wx[t1, t2], noise_sigma, win_size)

    x_out = iwt_po(ws, L, qmf)

    return x_out


if __name__ == '__main__':
