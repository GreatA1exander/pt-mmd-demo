import torch
#import scipy.stats import norm

eps_ = 1e-8

def mix_rbf_mmd2(X, Y, sigma_list, biased=True):

    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)

    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)

def mix_rbf_mmd2_and_ratio(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)

# relative mmd
def mmd_3_sample_test(X, Y, Z, sigma_list, computeMMDs=False):
    """

    :param X: true data, (Num Sample, Dimension of Data)
    :param Y: generated data 1 (full precision) (Num Sample, Dimension of Data)
    :param Z: generated data 2 (pruned / low precision) (Num Sample, Dimension of Data)
    :param sigma_list:
    :param SelectSigma:
    :param computeMMDs:
    :return:
    """
    assert X.size(0) == Y.size(0) == Z.size(0) #num samples is same
    m = X.size(0)

    K_XX, K_XY, K_YY, _ = _mix_rbf_kernel(X, Y, sigma_list)
    K_XX, K_XZ, K_ZZ, _ = _mix_rbf_kernel(X, Z, sigma_list)

    diag_X = torch.diag(K_XX)
    diag_Y = torch.diag(K_XY)
    diag_Z = torch.diag(K_ZZ)
    sum_diag_X = torch.sum(diag_X)
    sum_diag_Y = torch.sum(diag_Y)
    sum_diag_Z = torch.sum(diag_Z)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y
    Kt_ZZ_sums = K_ZZ.sum(dim=1) - diag_Z
    K_XY_sums_0 = K_XY.sum(dim=0)
    K_XZ_sums_0 = K_XY.sum(dim=0)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    Kt_ZZ_sum = Kt_ZZ_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()
    K_XZ_sum = K_XZ_sums_0.sum()

    u_yy = Kt_YY_sum / (m * (m - 1))
    u_zz = Kt_ZZ_sum / (m * (m - 1))
    u_xy = K_XY_sum / (m * m)
    u_xz = K_XZ_sum / (m * m)

    t = u_yy - 2 * u_xy - (u_zz - 2 * u_xz)
    diff_var, diff_var_z2,  data = _mmd_diff_var(K_YY, K_ZZ, K_XY, K_XZ)

    pvalue = _norm_cdf(-t/torch.sqrt(diff_var))
    # pvalue = _norm_cdf(-t / torch.sqrt(diff_var_z2))
    tstat = t/torch.sqrt(diff_var)

    if computeMMDs:
        u_xx = Kt_XX_sum / (m * (m - 1))
        MMD_XY = u_xx + u_yy - 2 * u_xy
        MMD_XZ = u_xx + u_yy - 2 * u_xz
    else:
        MMD_XY = None
        MMD_XZ = None

    return pvalue, tstat, sigma_list, MMD_XY, MMD_XZ




###############################################################################
# Helper Functions
###############################################################################




def _mix_rbf_kernel(X, Y, sigma_list):

    assert(X.size(0) == Y.size(0)) # make sure num samples is same

    m = X.size(0)
    Z = torch.cat((X,Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()
    
    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma ** 2)
        K += torch.exp(-gamma * exponent)
    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)  #KXX, KXY, KYY, num_sigmas

def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):

    m = K_XX.size(0)

    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)
        diag_Y = torch.diag(K_XY)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
                + Kt_YY_sum / (m * (m - 1))
                - 2.0 * K_XY_sum / (m * m))

    return mmd2

def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    mmd2, var_est = _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased)
    loss = mmd2 / torch.sqrt(torch.clamp(var_est, min=eps_))
    return loss, mmd2, var_est

def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e
    K_XY_sums_1 = K_XY.sum(dim=1)                     # K_{XY} * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X      # \| \tilde{K}_XX \|_F^2
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y      # \| \tilde{K}_YY \|_F^2
    K_XY_2_sum  = (K_XY ** 2).sum()                    # \| K_{XY} \|_F^2

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    var_est = (
        2.0 / (m**2 * (m - 1.0)**2) * (2 * Kt_XX_sums.dot(Kt_XX_sums) - Kt_XX_2_sum + 2 * Kt_YY_sums.dot(Kt_YY_sums) - Kt_YY_2_sum)
        - (4.0*m - 6.0) / (m**3 * (m - 1.0)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4.0*(m - 2.0) / (m**3 * (m - 1.0)**2) * (K_XY_sums_1.dot(K_XY_sums_1) + K_XY_sums_0.dot(K_XY_sums_0))
        - 4.0*(m - 3.0) / (m**3 * (m - 1.0)**2) * (K_XY_2_sum) - (8 * m - 12) / (m**5 * (m - 1)) * K_XY_sum**2
        + 8.0 / (m**3 * (m - 1.0)) * (
            1.0 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - Kt_XX_sums.dot(K_XY_sums_1)
            - Kt_YY_sums.dot(K_XY_sums_0))
        )
    return mmd2, var_est

def _norm_cdf(x):
    return norm.cdf(x.data.cpu().numpy())

def _mmd_diff_var(K_YY, K_ZZ, K_XY, K_XZ):

    assert K_YY.size(0) == K_ZZ.size(0) == K_XY.size(0) == K_XZ.size(0)
    m = K_YY.size(0)

    K_YY_nd = K_YY - torch.diag(K_YY)
    K_ZZ_nd = K_ZZ - torch.diag(K_ZZ)

    u_yy = torch.sum(K_YY_nd) / (m * (m - 1))
    u_zz = torch.sum(K_ZZ_nd) / (m * (m - 1))
    u_xy = torch.sum(K_XY) / (m * m)
    u_xz = torch.sum(K_XZ) / (m * m)

    t1=(1./m**3) * torch.sum(K_YY_nd.T.dot(K_YY_nd))-u_yy**2
    t2=(1./(m**2*m)) * torch.sum(K_XY.T.dot(K_XY))-u_xy**2
    t3=(1./(m*m**2)) * torch.sum(K_XY.dot(K_XY.T))-u_xy**2
    t4=(1./m**3) * torch.sum(K_ZZ_nd.T.dot(K_ZZ_nd))-u_zz**2
    t5=(1./(m*m**2)) * torch.sum(K_XZ.dot(K_XZ.T))-u_xz**2
    t6=(1./(m**2*m)) * torch.sum(K_XZ.T.dot(K_XZ))-u_xz**2
    t7=(1./(m**2*m)) * torch.sum(K_YY_nd.dot(K_XY.T))-u_yy*u_xy
    t8=(1./(m*m*m)) * torch.sum(K_XY.T.dot(K_XZ))-u_xz*u_xy
    t9=(1./(m**2*m)) * torch.sum(K_ZZ_nd.dot(K_XZ.T))-u_zz*u_xz

    zeta1 = (t1 + t2 + t3 + t4 + t5 + t6 - 2. * (t7 + t8 + t9))
    zeta2 = (1 / m / (m - 1)) * torch.sum((K_YY_nd - K_ZZ_nd - K_XY.T - K_XY + K_XZ + K_XZ.T) ** 2) - (u_yy - 2. * u_xy - (u_zz - 2. * u_xz)) ** 2

    data = dict({'t1': t1,
                 't2': t2,
                 't3': t3,
                 't4': t4,
                 't5': t5,
                 't6': t6,
                 't7': t7,
                 't8': t8,
                 't9': t9,
                 'zeta1': zeta1,
                 'zeta2': zeta2,
                 })

    Var = (4. * (m - 2) / (m * (m - 1))) * zeta1
    Var_z2 = Var + (2. / (m * (m - 1))) * zeta2

    return Var, Var_z2, data

