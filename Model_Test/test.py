import numpy as np
import matplotlib.pyplot as plt


def tw_auto_corr_factor01(series, time_resolution, n_lags=20, Q=0, n_stds=2, plot=False):
    """
    Computes the autocorrelation function and signal timescale.

    Parameters:
    - series: 1D numpy array, signal time series.
    - time_resolution: time resolution of the time series (in seconds).
    - n_lags: number of lags for the autocorrelation function to compute (default 20).
    - Q: lags beyond which the theoretical ACF is deemed to have died out (default 0).
    - n_stds: number of standard deviations for confidence bounds (default 2).
    - plot: If True, plots the autocorrelation function and confidence bounds.

    Returns:
    - STS: signal timescale.
    - ACF: autocorrelation function.
    - lags: array of lags corresponding to ACF.
    - bounds: confidence bounds.
    """

    series = np.asarray(series)
    if series.ndim != 1:
        raise ValueError("Input 'series' must be a 1D array.")

    n = len(series)
    if n_lags >= n:
        n_lags = min(20, n - 1)

    if Q >= n_lags:
        raise ValueError("'Q' must be less than 'n_lags'.")

    # FFT-based computation of the ACF
    n_fft = 2 ** (np.ceil(np.log2(len(series))) + 1).astype(int)
    F = np.fft.fft(series - np.mean(series), n=n_fft)
    F = F * np.conj(F)
    acf = np.fft.ifft(F).real[:n_lags + 1]  # retain non-negative lags only
    acf /= acf[0]  # normalize
    lags = np.arange(n_lags + 1)

    # Confidence bounds
    sigmaQ = np.sqrt((1 + 2 * np.sum(acf[1:Q + 1] ** 2)) / n)
    bounds = sigmaQ * np.array([n_stds, -n_stds])

    # Calculate signal timescale (STS)
    positive_acf = np.sum(acf[acf > 0]) - 1  # exclude the 0th lag
    sts = positive_acf * time_resolution

    if plot:
        plt.stem(lags, acf, 'r', markerfmt='ro', basefmt=" ", use_line_collection=True)
        plt.axhline(y=bounds[0], color='b', linestyle='--')
        plt.axhline(y=bounds[1], color='b', linestyle='--')
        plt.axhline(0, color='k', lw=1)
        plt.xlabel('Lag')
        plt.ylabel('Sample Autocorrelation')
        plt.title('Sample Autocorrelation Function (ACF)')
        plt.grid(True)
        plt.show()

    return sts, acf, lags, bounds
