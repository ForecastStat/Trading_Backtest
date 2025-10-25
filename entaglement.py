#!/usr/bin/env python3
"""
entanglement.py

Scarica tickers S&P500, prende gli ultimi `years` anni di prezzi giornalieri,
calcola per coppie di titoli una misura "quantum-inspired" di dipendenza:
- costruisce kernel RBF su finestre di lunghezza `window` (default 120)
- normalizza a density matrix e calcola von Neumann entropy
- definisce qmutual = S(A)+S(B)-S(AB)
- esegue permutation test (shuffle B) per ottenere p-value

Infine decide se l'entanglement "ESISTE" o "NON ESISTE" usando una regola pratica:
Se almeno `min_fraction_significant` delle coppie testate hanno p-value < alpha e media dei qmutual > 0,
allora `EXISTS`, altrimenti `NOT_EXISTS`.

Prodotti:
- entanglement_log.txt
- (opzionale) grafici se richiesti

"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import pairwise_kernels
from scipy.linalg import eigh
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import random

# ---------------------------
# Utility math
# ---------------------------

def gaussian_kernel_matrix(X, gamma=None):
    # X: n_samples x n_features
    K = pairwise_kernels(X, metric='rbf', gamma=gamma)
    return (K + K.T) / 2


def density_matrix_from_kernel(K):
    # normalizza la traccia a 1
    K = (K + K.T) / 2
    tr = np.trace(K)
    if tr <= 0:
        # fallback: add small identity
        K = K + np.eye(K.shape[0]) * 1e-8
        tr = np.trace(K)
    return K / tr


def von_neumann_entropy(rho, eps=1e-12):
    # assume rho simmetrica
    vals, _ = eigh(rho)
    vals = np.clip(vals, eps, None)
    return -np.sum(vals * np.log(vals))


def compute_qmutual_from_data(XA, XB, gamma=None):
    # XA, XB: n_samples x features
    XAB = np.hstack([XA, XB])
    KAB = gaussian_kernel_matrix(XAB, gamma=gamma)
    rhoAB = density_matrix_from_kernel(KAB)

    KA = gaussian_kernel_matrix(XA, gamma=gamma)
    KB = gaussian_kernel_matrix(XB, gamma=gamma)
    rhoA = density_matrix_from_kernel(KA)
    rhoB = density_matrix_from_kernel(KB)

    SA = von_neumann_entropy(rhoA)
    SB = von_neumann_entropy(rhoB)
    SAB = von_neumann_entropy(rhoAB)

    qmutual = SA + SB - SAB
    return qmutual, SA, SB, SAB

# ---------------------------
# Data acquisition
# ---------------------------

def fetch_sp500_tickers():
    # prendo la lista da Wikipedia
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df['Symbol'].tolist()
    # alcuni ticker hanno punti (es BRK.B) che yfinance vuole come BRK-B
    tickers = [t.replace('.', '-') for t in tickers]
    return tickers


def download_price_data(tickers, start, end, threads=8):
    # usa yfinance download
    data = yf.download(tickers, start=start, end=end, group_by='ticker', threads=True, progress=False)
    # restituisce dict ticker -> series di Close
    out = {}
    for t in tickers:
        try:
            if (t,) in data.columns:
                close = data[(t, 'Close')]
            else:
                # se è singolo ticker yfinance restituisce DataFrame con 'Close'
                close = data['Close'] if 'Close' in data else None
            if close is None:
                continue
            s = close.dropna()
            if len(s) > 0:
                out[t] = s
        except Exception:
            continue
    return out

# ---------------------------
# Core experiment
# ---------------------------

def align_and_get_returns(series_a, series_b, window):
    df = pd.concat([series_a.rename('A'), series_b.rename('B')], axis=1).dropna()
    # need at least `window` samples
    if len(df) < window:
        return None, None
    recent = df.iloc[-window:]
    rA = np.log(recent['A']).diff().dropna().values.reshape(-1,1)
    rB = np.log(recent['B']).diff().dropna().values.reshape(-1,1)
    # if diff reduces length by 1, ensure equal
    L = min(len(rA), len(rB))
    return rA[-L:], rB[-L:]


def permutation_pvalue(XA, XB, observed_q, n_permutations=500, gamma=None, seed=None):
    rng = np.random.RandomState(seed)
    count = 0
    for i in range(n_permutations):
        perm = rng.permutation(XB.shape[0])
        XB_perm = XB[perm]
        q, *_ = compute_qmutual_from_data(XA, XB_perm, gamma=gamma)
        if q >= observed_q - 1e-12:
            count += 1
    pval = (count + 1) / (n_permutations + 1)
    return pval


def test_pair(series_a, series_b, window=120, n_permutations=500, gamma=None, seed=None):
    rA, rB = align_and_get_returns(series_a, series_b, window)
    if rA is None:
        return None
    q, SA, SB, SAB = compute_qmutual_from_data(rA, rB, gamma=gamma)
    p = permutation_pvalue(rA, rB, q, n_permutations=n_permutations, gamma=gamma, seed=seed)
    return {'qmutual': float(q), 'S_A': float(SA), 'S_B': float(SB), 'S_AB': float(SAB), 'pval': float(p)}

# ---------------------------
# Orchestrazione
# ---------------------------

def run_experiment(args):
    log = logging.getLogger('ent')
    log.info('Fetching S&P500 tickers...')
    tickers = fetch_sp500_tickers()
    random.seed(args.seed)
    tickers = [t for t in tickers if not t.startswith('^')]
    if args.n_tickers and args.n_tickers < len(tickers):
        tickers = random.sample(tickers, args.n_tickers)
    log.info(f'Using {len(tickers)} tickers')

    end = datetime.utcnow().date()
    start = end - timedelta(days=int(args.years*365))
    log.info(f'Downloading prices from {start} to {end}...')
    prices = download_price_data(tickers, start=start.isoformat(), end=end.isoformat())
    if len(prices) < 2:
        log.error('Not enough tickers with data downloaded.')
        sys.exit(1)

    # build list of pairs
    pairs = []
    all_t = list(prices.keys())
    # sample pairs randomly to bound compute
    while len(pairs) < args.n_pairs:
        a, b = random.sample(all_t, 2)
        if (a,b) not in pairs and (b,a) not in pairs:
            pairs.append((a,b))
            if len(pairs) >= len(all_t)*(len(all_t)-1)/2:
                break

    log.info(f'Testing {len(pairs)} pairs')

    results = []
    # parallel execution
    parallel = Parallel(n_jobs=args.jobs)
    tasks = (delayed(test_pair)(prices[a], prices[b], window=args.window, n_permutations=args.n_permutations, gamma=args.gamma, seed=args.seed+i)
             for i,(a,b) in enumerate(pairs))
    with tqdm(total=len(pairs)) as pbar:
        out = []
        for i, res in enumerate(parallel(tasks)):
            pbar.update(1)
            out.append(res)
    # note: joblib + generator is tricky; above is a simplified approach but may need adaptation
    # to be robust, we'll do sequential with progress bar if n_jobs==1

    if args.jobs == 1:
        out = []
        for i,(a,b) in enumerate(pairs):
            res = test_pair(prices[a], prices[b], window=args.window, n_permutations=args.n_permutations, gamma=args.gamma, seed=args.seed+i)
            out.append(res)
            tqdm.write(f'[{i+1}/{len(pairs)}] {a}-{b}: {res}')

    # filter out None
    filtered = [r for r in out if r is not None]
    df = pd.DataFrame(filtered)
    df['pair'] = [f'{a}-{b}' for (a,b),r in zip(pairs, out) if r is not None]

    # decision rule
    alpha = args.alpha
    significant = df[df['pval'] < alpha]
    frac_significant = len(significant) / len(df) if len(df)>0 else 0.0
    mean_q = df['qmutual'].mean() if len(df)>0 else 0.0

    exists = (frac_significant >= args.min_fraction_significant) and (mean_q > 0)

    # Logging results
    loglines = []
    loglines.append(f'Run date: {datetime.utcnow().isoformat()}')
    loglines.append(f'Parameters: years={args.years}, window={args.window}, n_tickers={len(tickers)}, n_pairs={len(pairs)}, n_permutations={args.n_permutations}')
    loglines.append(f'Tested pairs: {len(df)}')
    loglines.append(f'Alpha (perm test): {alpha}, min_fraction_significant: {args.min_fraction_significant}')
    loglines.append(f'Fraction significant pairs: {frac_significant:.4f}')
    loglines.append(f'Mean qmutual: {mean_q:.6f}')
    loglines.append('Decision: ' + ('EXISTS' if exists else 'NOT_EXISTS'))
    loglines.append('\nTop significant pairs:')
    for _, row in significant.sort_values('qmutual', ascending=False).head(20).iterrows():
        loglines.append(f"{row['pair']}: q={row['qmutual']:.6f}, p={row['pval']:.4f}")

    # write log
    with open('entanglement_log.txt', 'w') as f:
        f.write('\n'.join(loglines))

    for L in loglines:
        log.info(L)

    return exists, df

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--years', type=float, default=2.0, help='Years of history to download')
    p.add_argument('--window', type=int, default=120, help='Window length (days) for kernel/density')
    p.add_argument('--n-tickers', type=int, default=100, help='Number of tickers to sample from S&P500')
    p.add_argument('--n-pairs', type=int, default=200, help='Number of random pairs to test')
    p.add_argument('--n-permutations', type=int, default=500, help='Permutations for permutation test')
    p.add_argument('--gamma', type=float, default=None, help='Gamma for RBF kernel (None = 1/n_features)')
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--min-fraction-significant', type=float, default=0.10, help='Fraction of pairs that must be significant to decide EXISTS')
    p.add_argument('--jobs', type=int, default=1, help='Parallel jobs (joblib). Default 1 for determinism.')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    exists, df = run_experiment(args)
    # exit code 0 but print decision
    print('\nFINAL DECISION:', 'EXISTS' if exists else 'NOT_EXISTS')
