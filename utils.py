# from ssqueezepy import ssq_cwt, Wavelet
import numpy as np
from scipy.signal import periodogram


def coordinate_trans_rev2(u2, v2, w2, bata, theta):
    """
    Perform coordinate transformation.
    """
    n1 = len(u2)
    us = np.full(n1, np.nan)
    vs = np.full(n1, np.nan)
    ws = np.full(n1, np.nan)
    R = np.array([[np.cos(np.radians(theta)) * np.cos(np.radians(bata)), np.sin(np.radians(bata)), -np.sin(np.radians(theta)) * np.cos(np.radians(bata))],
                  [-np.cos(np.radians(theta)) * np.sin(np.radians(bata)), np.cos(
                      np.radians(bata)), np.sin(np.radians(theta)) * np.sin(np.radians(bata))],
                  [np.sin(np.radians(theta)), 0, np.cos(np.radians(theta))]])
    for j in range(n1):
        U = np.array([u2[j], v2[j], w2[j]])
        Us = np.dot(R, U)
        us[j] = Us[0]
        vs[j] = Us[1]
        ws[j] = Us[2]
    return us, vs, ws


def dis_int(y, x, ll, lu):
    """
    Discrete data integration.
    """
    # Find start and end indices for the integration limits
    start = np.searchsorted(x, ll, side='left')
    end = np.searchsorted(x, lu, side='right') - 1

    # Ensure the indices are within the bounds of the array
    start = max(start, 0)
    end = min(end, len(x) - 1)

    # Calculate the length of each subinterval
    ax = (x[end] - x[start]) / (end - start)

    # Perform the integration using the trapezoidal rule
    if start != end:
        y_int = 0.5 * ax * (y[start] + 2 * np.sum(y[start+1:end]) + y[end])
    else:
        # If start and end are the same, the integral is zero
        y_int = 0

    return y_int


def rotate_coordinate(u2, v2, w2, theta, i, n1, fs):
    """
    Rotate coordinate system and analyze velocity fluctuations.
    """
    bata = np.arange(0, 361)
    n3 = len(bata)
    fw1 = 0.05
    fw2 = 0.5
    psdus_int = np.zeros(n3)

    usgroup = np.zeros((n1, n3))
    vsgroup = np.zeros((n1, n3))
    wsgroup = np.zeros((n1, n3))

    for j in range(n3):
        us, vs, ws = coordinate_trans_rev2(
            u2[:, i], v2[:, i], w2[:, i], bata[j], theta[i])
        usgroup[:, j] = us
        vsgroup[:, j] = vs
        wsgroup[:, j] = ws
        freq, psdus = periodogram(us, fs, nfft=n1)
        psdvs, _ = periodogram(vs, fs, nfft=n1)
        psdws, _ = periodogram(ws, fs, nfft=n1)
        psdus_int[j] = dis_int(psdus, freq, fw1, fw2)

    bata_max1 = np.argmax(psdus_int)

    u = usgroup[:, bata_max1]
    v = vsgroup[:, bata_max1]
    w = wsgroup[:, bata_max1]

    um = np.mean(u)
    vm = np.mean(v)
    wm = np.mean(w)

    uf = u - um
    vf = v - vm
    wf = w - wm

    return bata_max1, uf, vf, wf


def ESA(uf, vf, wf, Fs, Fw1, Fw2):
    n1 = len(uf)
    freq, psdu = periodogram(uf, fs=Fs, nfft=n1)
    _, psdv = periodogram(vf, fs=Fs, nfft=n1)
    _, psdw = periodogram(wf, fs=Fs, nfft=n1)
    psd = np.column_stack((psdu, psdv, psdw))

    # Find indices corresponding to Fw1 and Fw2
    Stop1 = np.floor((Fw1 - freq[0]) / (freq[1] - freq[0])).astype(int)
    Stop2 = np.floor((Fw2 - freq[0]) / (freq[1] - freq[0])).astype(int)

    psd_fit = psd.copy()
    for i in range(3):  # u, v, w
        x = np.log10(freq[1:])
        y = np.log10(psd[1:, i])
        fit_params = np.polyfit(np.concatenate([x[:Stop1], x[Stop2:]]),
                                np.concatenate([y[:Stop1], y[Stop2:]]), 1)
        p1, p2 = fit_params
        yfit = p1 * x + p2
        psd_fit[1:, i] = 10 ** yfit

    # PSD ESA
    psd_ESA = psd.copy()
    psd_ESA[Stop1:Stop2, :] = psd_fit[Stop1:Stop2, :]

    # Accumulated PSD
    df = freq[1] - freq[0]
    n2 = len(freq)
    Ac_psdu_ESA = np.full(n2, np.nan)
    Ac_psdv_ESA = np.full(n2, np.nan)
    Ac_psdw_ESA = np.full(n2, np.nan)
    for i in range(n2):
        Ac_psdu_ESA[i] = 0.5 * (2 * np.sum(psd_ESA[:i + 1, 0]) -
                                psd_ESA[0, 0] - psd_ESA[i, 0]) * df
        Ac_psdv_ESA[i] = 0.5 * (2 * np.sum(psd_ESA[:i + 1, 1]) -
                                psd_ESA[0, 1] - psd_ESA[i, 1]) * df
        Ac_psdw_ESA[i] = 0.5 * (2 * np.sum(psd_ESA[:i + 1, 2]) -
                                psd_ESA[0, 2] - psd_ESA[i, 2]) * df
    Ac_psd_ESA = np.column_stack((Ac_psdu_ESA, Ac_psdv_ESA, Ac_psdw_ESA))

    return psd_ESA, Ac_psd_ESA


# def MA(uf, vf, wf, Fs, Fw2):
#     """
#     Moving Average (MA) method for power spectrum density analysis.

#     Parameters:
#     uf, vf, wf: Arrays of u, v, w fluctuations
#     Fs: Sampling frequency
#     Fw2: Frequency for moving average

#     Returns:
#     psd_MA: Power spectrum density by MA method
#     Ac_psd_MA: Accumulative psd by MA method
#     """
#     window = int(round((1. / Fw2) * Fs))

#     # Applying moving average
#     def smooth(data, window_size):
#         return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

#     uw = smooth(uf, window)
#     vw = smooth(vf, window)
#     ww = smooth(wf, window)

#     # Calculating the fluctuations minus the smoothed data
#     ut = uf[window-1:] - uw
#     vt = vf[window-1:] - vw
#     wt = wf[window-1:] - ww

#     # Power spectrum density calculation
#     # n1 = len(ut)
#     freq, psdu_MA = periodogram(ut, Fs, scaling='density')
#     _, psdv_MA = periodogram(vt, Fs, scaling='density')
#     _, psdw_MA = periodogram(wt, Fs, scaling='density')
#     psd_MA = np.stack((psdu_MA, psdv_MA, psdw_MA), axis=-1)

#     # Accumulated power spectrum density
#     df = freq[1] - freq[0]
#     Ac_psd_MA = np.cumsum(psd_MA * df, axis=0)

#     return psd_MA, Ac_psd_MA


def smooth_2(y, span=5):  # 无法完全复原matlab中的smooth
    # Ensure y is a column vector
    y = np.asarray(y).flatten()

    # Adjust span to be odd and not greater than data length
    span = min(span, len(y))
    if span % 2 == 0:
        span -= 1

    # Apply moving average
    half_span = (span - 1) // 2
    c = np.full(y.shape, np.nan)
    for i in range(len(y)):
        if i < half_span:
            c[i] = np.mean(y[:i + half_span + 1])
        elif i > len(y) - half_span - 1:
            c[i] = np.mean(y[i - half_span:])
        else:
            c[i] = np.mean(y[i - half_span:i + half_span + 1])
    return c


def MA(uf, vf, wf, Fs, Fw2):
    window = round((1. / Fw2) * Fs)

    # Smoothing using the smooth_2 function
    uw = smooth_2(uf, window)
    vw = smooth_2(vf, window)
    ww = smooth_2(wf, window)

    ut = uf - uw
    vt = vf - vw
    wt = wf - ww

    # Power spectrum density
    freq, psdu_MA = periodogram(ut, Fs)
    _, psdv_MA = periodogram(vt, Fs)
    _, psdw_MA = periodogram(wt, Fs)
    psd_MA = np.column_stack((psdu_MA, psdv_MA, psdw_MA))

    # Accumulated PSD
    df = freq[1] - freq[0]
    Ac_psdu_MA = np.cumsum(psdu_MA) * df
    Ac_psdv_MA = np.cumsum(psdv_MA) * df
    Ac_psdw_MA = np.cumsum(psdw_MA) * df
    Ac_psd_MA = np.column_stack((Ac_psdu_MA, Ac_psdv_MA, Ac_psdw_MA))

    return psd_MA, Ac_psd_MA


def Origin(uf, vf, wf, Fs):
    """
    Calculate the original power spectrum density (psd) and the original accumulative power spectrum density (Ac_psd).

    Parameters:
    uf, vf, wf: Arrays of u, v, w fluctuations
    Fs: Sampling frequency

    Returns:
    freq: Frequency array
    psd: Power spectrum density
    Ac_psd: Accumulative power spectrum density
    """
    n1 = len(uf)

    # Power spectrum density calculation
    freq, psdu = periodogram(uf, Fs, scaling='density')
    _, psdv = periodogram(vf, Fs, scaling='density')
    _, psdw = periodogram(wf, Fs, scaling='density')
    psd = np.stack((psdu, psdv, psdw), axis=-1)

    # Accumulated power spectrum density
    df = freq[1] - freq[0]
    n2 = len(freq)
    Ac_psdu = np.nan * np.zeros(n2)
    Ac_psdv = np.nan * np.zeros(n2)
    Ac_psdw = np.nan * np.zeros(n2)

    for i in range(n2):
        Ac_psdu[i] = 0.5 * (2 * np.sum(psdu[:i+1]) - psdu[0] - psdu[i]) * df
        Ac_psdv[i] = 0.5 * (2 * np.sum(psdv[:i+1]) - psdv[0] - psdv[i]) * df
        Ac_psdw[i] = 0.5 * (2 * np.sum(psdw[:i+1]) - psdw[0] - psdw[i]) * df

    Ac_psd = np.stack((Ac_psdu, Ac_psdv, Ac_psdw), axis=-1)

    return freq, psd, Ac_psd


# def RoCoM(uf, vf, wf, Fs, Fw1, Fw2):
#     """
#     Rotating Coordinate Method (RoCoM) for power spectrum density analysis.

#     Parameters:
#     uf, vf, wf: Arrays of u, v, w fluctuations
#     Fs: Sampling frequency
#     Fw1: Lowest wave frequency for the representative range
#     Fw2: Highest wave frequency for the representative range

#     Returns:
#     psd_RoCoM: Power spectrum density by RoCoM method
#     Ac_psd_RoCoM: Accumulative psd by RoCoM method
#     """
#     # n1 = len(uf)

#     # 计算原始功率谱密度 (PSD)
#     freq, psdu = periodogram(uf, fs=Fs, scaling='density')
#     _, psdv = periodogram(vf, fs=Fs, scaling='density')
#     _, psdw = periodogram(wf, fs=Fs, scaling='density')

#     # 平滑 PSD
#     # smooth_logscale 函数需要根据 MATLAB 中的实现适当编写
#     psdu_smooth, _ = smooth_logscale(psdu, freq, 0.05)
#     psdv_smooth, _ = smooth_logscale(psdv, freq, 0.05)
#     psdw_smooth, _ = smooth_logscale(psdw, freq, 0.05)

#     # 计算比率
#     id1 = int(np.floor((Fw1 - freq[0]) / (freq[1] - freq[0])) + 1)
#     id2 = int(np.floor((Fw2 - freq[0]) / (freq[1] - freq[0])) + 1)

#     epsilon = np.finfo(float).eps
#     R_LH_us_vs = np.mean(psdu_smooth[id2:] / (psdv_smooth[id2:] + epsilon))
#     R_LH_ws_vs = np.mean(psdw_smooth[id2:] / (psdv_smooth[id2:] + epsilon))

#     # RoCoM PSD
#     psdu_RoCoM = psdu.copy()
#     psdw_RoCoM = psdw.copy()
#     psdu_RoCoM[id1:id2] = R_LH_us_vs * psdv[id1:id2]
#     psdw_RoCoM[id1:id2] = R_LH_ws_vs * psdv[id1:id2]
#     psd_RoCoM = np.column_stack((psdu_RoCoM, psdv, psdw_RoCoM))

#     # 累积 PSD
#     df = freq[1] - freq[0]
#     n2 = len(freq)
#     Ac_psdu_RoCoM = np.cumsum(psdu_RoCoM) * df - 0.5 * psdu_RoCoM[0] * df
#     Ac_psdv_RoCoM = np.cumsum(psdv) * df - 0.5 * psdv[0] * df
#     Ac_psdw_RoCoM = np.cumsum(psdw_RoCoM) * df - 0.5 * psdw_RoCoM[0] * df
#     Ac_psd_RoCoM = np.column_stack(
#         (Ac_psdu_RoCoM, Ac_psdv_RoCoM, Ac_psdw_RoCoM))

#     return psd_RoCoM, Ac_psd_RoCoM

def smooth_logscale(y, x, dx):
    """
    Smooth the y values based on their corresponding x values in logarithmic scale.

    Parameters:
    y (array_like): The y values.
    x (array_like): The x values.
    dx (float): The smoothing parameter.

    Returns:
    ysmooth (numpy.ndarray): The smoothed y values.
    xsmooth (numpy.ndarray): The x values converted back from logarithmic scale.
    """
    # Check for zero value in x and compute logarithm
    if x[0] == 0:
        xlog = np.log10(x[1:])
        y = y[1:]
    else:
        xlog = np.log10(x)

    # Calculate the average
    n = len(xlog)
    ysmooth = np.empty(n)
    for i in range(n):
        # Range for average
        xlog_lower_limit = xlog[i] - dx
        xlog_upper_limit = xlog[i] + dx

        # Find indices for averaging
        id1 = np.where(xlog > xlog_lower_limit)[0]
        id2 = np.where(xlog < xlog_upper_limit)[0]
        id = np.intersect1d(id1, id2)

        ysmooth[i] = np.mean(y[id])

    xsmooth = 10 ** xlog
    return ysmooth, xsmooth


def RoCoM(uf, vf, wf, Fs, Fw1, Fw2):
    """
    RoCoM (Rotating Coordinate Method) for calculating power spectrum density (PSD).

    Parameters:
    uf, vf, wf (array_like): Fluctuating components (u, v, w fluctuations).
    Fs (float): Sampling frequency.
    Fw1 (float): Lowest wave frequency for the representative range.
    Fw2 (float): Highest wave frequency for the representative range.

    Returns:
    Tuple containing:
    - psdu, psdv, psdw: PSD of u, v, w fluctuations.
    - freq: Frequency array.
    - psdu_smooth, psdv_smooth, psdw_smooth: Smoothed PSDs.
    - psd_RoCoM: PSD by RoCoM method.
    - Ac_psd_RoCoM: Accumulative PSD by RoCoM method.
    """
    # PSD
    n1 = len(uf)
    freq, psdu = periodogram(uf, fs=Fs, nfft=n1)
    _, psdv = periodogram(vf, fs=Fs, nfft=n1)
    _, psdw = periodogram(wf, fs=Fs, nfft=n1)

    # Smooth PSD
    psdu_smooth, _ = smooth_logscale(psdu, freq, 0.05)
    psdv_smooth, _ = smooth_logscale(psdv, freq, 0.05)
    psdw_smooth, _ = smooth_logscale(psdw, freq, 0.05)

    # Post-smooth ratio without wave-frequency range
    id1 = int(np.floor((Fw1 - freq[0]) / (freq[1] - freq[0]) + 1))
    id2 = int(np.floor((Fw2 - freq[0]) / (freq[1] - freq[0]) + 1))

    R_LH_us_vs = np.mean(psdu_smooth[id2:] / psdv_smooth[id2:])
    R_LH_ws_vs = np.mean(psdw_smooth[id2:] / psdv_smooth[id2:])

    # PSD by RoCoM
    psdu_RoCoM = np.copy(psdu)
    psdw_RoCoM = np.copy(psdw)
    psdu_RoCoM[id1:id2] = R_LH_us_vs * psdv[id1:id2]
    psdw_RoCoM[id1:id2] = R_LH_ws_vs * psdv[id1:id2]
    psd_RoCoM = np.array([psdu_RoCoM, psdv, psdw_RoCoM])

    # Accumulated PSD by RoCoM
    df = freq[1] - freq[0]
    n2 = len(freq)
    Ac_psd_RoCoM = np.empty((n2, 3))
    for i in range(n2):
        Ac_psd_RoCoM[i, 0] = 0.5 * \
            (2 * np.sum(psdu_RoCoM[:i+1]) - psdu_RoCoM[0] - psdu_RoCoM[i]) * df
        Ac_psd_RoCoM[i, 1] = 0.5 * \
            (2 * np.sum(psdv[:i+1]) - psdv[0] - psdv[i]) * df
        Ac_psd_RoCoM[i, 2] = 0.5 * \
            (2 * np.sum(psdw_RoCoM[:i+1]) - psdw_RoCoM[0] - psdw_RoCoM[i]) * df

    return psd_RoCoM, Ac_psd_RoCoM

# Note: The smooth_logscale function defined earlier must also be included in the same Python script for this to work.


# def smooth_logscale(y, x, dx):
#     """
#     Smooth a given signal 'y' over 'x' on a logarithmic scale with a smoothing range defined by 'dx'.

#     Parameters:
#     y: Array of signal values
#     x: Array of x-values (must be positive and non-zero)
#     dx: Smoothing range on the log scale

#     Returns:
#     ysmooth: Smoothed signal values
#     xsmooth: Corresponding x-values on the original scale
#     """
#     # Check for zero value in x and prepare x for logarithmic scale
#     if x[0] == 0:
#         xlog = np.log10(x[1:])
#         y = y[1:]
#     else:
#         xlog = np.log10(x)

#     n = len(xlog)
#     ysmooth = np.full(n, np.nan)

#     for i in range(n):
#         # Range for averaging
#         xlog_lower_limit = xlog[i] - dx
#         xlog_upper_limit = xlog[i] + dx

#         # Find values within the range for averaging
#         id1 = np.where(xlog > xlog_lower_limit)[0]
#         id2 = np.where(xlog < xlog_upper_limit)[0]
#         id = np.intersect1d(id1, id2)

#         ysmooth[i] = np.mean(y[id])

#     xsmooth = 10**xlog

#     return ysmooth, xsmooth


# def SWT(uf, vf, wf, Fs, Fw1, Fw2):
#     """
#     SWT (Synchrosqueezed Wavelet Transform)-based method for power spectrum density analysis.

#     Parameters:
#     uf, vf, wf: Arrays of u, v, w fluctuations
#     Fs: Sampling frequency
#     Fw1, Fw2: Frequency range for the representative wave

#     Returns:
#     psd_SWT: Power spectrum density by SWT method
#     Ac_psd_SWT: Accumulative psd by SWT method
#     Uw: Wavelet transformed signals
#     """
#     n1 = len(uf)
#     t = np.linspace(0, n1 / Fs, n1)

#     # CWTopt and other parameters for SWT need to be defined or adapted from a suitable Python library

#     Uf = np.column_stack((uf, vf, wf))
#     Uw = np.zeros_like(Uf)

#     for i in range(3):
#         x = Uf[:, i]
#         # Apply SWT and related operations
#         # Placeholder for synsq_cwt_fw, synsq_filter_pass, synsq_cwt_iw
#         # Actual implementation should perform Synchrosqueezed Wavelet Transform

#     Ut = Uf - Uw
#     ut = Ut[:, 0]
#     vt = Ut[:, 1]
#     wt = Ut[:, 2]

#     # Power spectrum density calculation
#     freq, psdu_SWT = periodogram(ut, Fs, scaling='density')
#     _, psdv_SWT = periodogram(vt, Fs, scaling='density')
#     _, psdw_SWT = periodogram(wt, Fs, scaling='density')
#     psd_SWT = np.column_stack((psdu_SWT, psdv_SWT, psdw_SWT))

#     # Accumulated power spectrum density
#     df = freq[1] - freq[0]
#     n2 = len(freq)
#     Ac_psdu_SWT = np.nan * np.zeros(n2)
#     Ac_psdv_SWT = np.nan * np.zeros(n2)
#     Ac_psdw_SWT = np.nan * np.zeros(n2)

#     for i in range(n2):
#         Ac_psdu_SWT[i] = 0.5 * (2 * np.sum(psdu_SWT[:i+1]) - psdu_SWT[0] - psdu_SWT[i]) * df
#         Ac_psdv_SWT[i] = 0.5 * (2 * np.sum(psdv_SWT[:i+1]) - psdv_SWT[0] - psdv_SWT[i]) * df
#         Ac_psdw_SWT[i] = 0.5 * (2 * np.sum(psdw_SWT[:i+1]) - psdw_SWT[0] - psdw_SWT[i]) * df

#     Ac_psd_SWT = np.column_stack((Ac_psdu_SWT, Ac_psdv_SWT, Ac_psdw_SWT))

#     return psd_SWT, Ac_psd_SWT, Uw


# def SWT(uf, vf, wf, Fs, Fw1, Fw2):
#     """
#     Synchrosqueezed Wavelet Transform (SWT)-based method.

#     Arguments:
#     uf, vf, wf -- u, v, w fluctuation arrays
#     Fs -- Sampling frequency
#     Fw1, Fw2 -- Lowest and highest wave frequencies for the representative range

#     Returns:
#     psd_SWT -- Power spectrum density by SWT method
#     Ac_psd_SWT -- Accumulative PSD by SWT method
#     Uw -- Filtered signals
#     """
#     n1 = len(uf)
#     t = np.linspace(0, n1 / Fs, n1)
#     Uf = np.array([uf, vf, wf]).T
#     Uw = np.zeros_like(Uf)

#     # Synchrosqueezing parameters
#     wavelet = Wavelet('morlet')

#     for i in range(3):
#         x = Uf[:, i]
#         Tx, Wx, fs, _ = ssq_cwt(x, wavelet)
#         # Manual passband filtering
#         fmin, fmax = Fw1, Fw2
#         mask = (fs >= fmin) & (fs <= fmax)
#         Txf = Tx * mask
#         # Inverse transform
#         xf = np.real(np.sum(Txf, axis=1))
#         Uw[:, i] = xf

#     Ut = Uf - Uw
#     psd_SWT = []
#     for i in range(3):
#         f, psd = periodogram(Ut[:, i], Fs)
#         psd_SWT.append(psd)
#     psd_SWT = np.array(psd_SWT).T

#     # Accumulated PSD
#     df = f[1] - f[0]
#     Ac_psd_SWT = np.cumsum(psd_SWT, axis=0) * df

#     return psd_SWT, Ac_psd_SWT, Uw
