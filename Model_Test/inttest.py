import numpy as np
import os
from scipy.interpolate import splrep, sproot

# Parameters
num_nodes = 200  # Number of time series nodes
outdir = '/path/to/out'
os.makedirs(outdir, exist_ok=True)
lags = np.arange(-6, 7)  # Range of TR shifts
tr = 2  # Sampling interval (seconds)
motion_thresh = 0.2  # Motion threshold
min_block_durn = (max(lags) + 1) * tr  # Minimum block duration

# Import subject list
subjects = np.loadtxt('/path/to/subject_list.txt', dtype=str)

# Initialize ACF storage
grp_acfs = np.full((num_nodes, len(lags), len(subjects)), np.nan, dtype=np.float32)


# Functions for block creation, cross-covariance, and HWHM estimation
def create_blocks(format_mask, min_block_duration, tr):
    blocks = []
    current_block = []
    for i, val in enumerate(format_mask):
        if val:
            current_block.append(i)
            if len(current_block) * tr >= min_block_duration:
                blocks.append(current_block)
                current_block = []
        else:
            current_block = []
    return blocks


def lagged_cov(Avg1, Avg2, L):
    """
    Computes unnormalized cross-covariance function out to +/- L lags (TR shifts)
    between each column of Avg1 and Avg2.
    """
    L1, L2 = Avg1.shape[1], Avg2.shape[1]
    r = np.zeros((L1, L2, 2 * L + 1), dtype=np.float32)

    k = 0
    for i in range(-L, L + 1):
        tau = abs(i)
        if i >= 0:
            Avg1_lagged = Avg1[:-tau or None, :]
            Avg2_lagged = Avg2[tau:, :]
        else:
            Avg1_lagged = Avg1[tau:, :]
            Avg2_lagged = Avg2[:-tau or None, :]

        r[:, :, k] = np.dot(Avg1_lagged.T, Avg2_lagged)
        k += 1

    return r


def acf_hwhm(acf, tr):
    """
    Computes the half-width at half-maximum (HWHM) of the ACF.
    """
    num_nodes = acf.shape[1]
    out_hwhm = np.full(num_nodes, np.nan)
    good_mask = ~np.isnan(acf).any(axis=0)
    acf = acf[:, good_mask]

    nlags = (acf.shape[0] - 1) // 2
    lags = tr * np.arange(-nlags, nlags + 1)
    fwhm = np.full(acf.shape[1], np.nan)

    for n in range(acf.shape[1]):
        pp = splrep(lags, acf[:, n] - 0.5)
        try:
            roots = sproot(pp)
            if roots.size > 0:
                fwhm[n] = 2 * abs(roots[0])
        except ValueError:
            fwhm[n] = np.nan

    out_hwhm[good_mask] = fwhm / 2
    return out_hwhm


# Main loop over subjects
for s, subj in enumerate(subjects):
    print(f'Processing {subj}...')

    # Load BOLD data and motion mask
    BOLD = np.loadtxt(f'/path/to/BOLD_{subj}.txt')  # Replace with actual path
    good = np.loadtxt(f'/path/to/good_{subj}.txt', dtype=bool)  # Replace with actual path
    motion = np.loadtxt(f'/path/to/motion_{subj}.txt')
    format_mask = motion <= motion_thresh
    format_mask[:2] = False  # Ignore pre-steady-state frames

    # Create blocks with low motion
    FORMAT = create_blocks(format_mask, min_block_durn, tr)

    # Initialize ACF
    ACFs = np.zeros((num_nodes, len(lags)), dtype=np.float32)
    nframes = 0

    # Zero-mean BOLD data
    run_mean = np.nanmean(BOLD[format_mask], axis=0)
    BOLD -= run_mean

    # Calculate ACF using lagged covariance
    for block in FORMAT:
        nframes += len(block)
        FHCR = np.zeros_like(format_mask, dtype=bool)
        FHCR[block] = True
        acf_block = lagged_cov(BOLD[FHCR], BOLD[FHCR], max(lags))
        ACFs += np.mean(acf_block, axis=1)  # Average across blocks

    # Normalize ACF
    for k, lag in enumerate(lags):
        ACFs[:, k] /= (nframes - abs(lag) * len(FORMAT))
    ACFs /= ACFs[:, lags == 0].reshape(-1, 1)

    # Store ACFs
    grp_acfs[good, :, s] = ACFs

# Calculate mean ACF and HWHM
acf_mean = np.nanmean(grp_acfs, axis=2)
acf_mean /= acf_mean[:, lags == 0].reshape(-1, 1)
hwhm = acf_hwhm(acf_mean, tr)

# Calculate individual HWHM for each subject
hwhms = np.zeros((num_nodes, len(subjects)), dtype=np.float32)
for s in range(len(subjects)):
    hwhms[:, s] = acf_hwhm(grp_acfs[:, :, s], tr)
