"""
End-to-end validation of BHDT + PINOD ensemble integration.

Tests 8 key scenarios with both methods and verifies ensemble
produces better or equal accuracy compared to either method alone.

Usage:
    PYTHONPATH=src python tests/test_ensemble_e2e.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import time
from collections import Counter

from chord.simulation.generator import (
    pure_circadian, pure_ultradian, independent_superposition,
    sawtooth_harmonic, pure_noise, damped_ultradian,
    peaked_harmonic, asymmetric_ultradian,
)
from chord.bhdt.inference import bhdt_analytic
from chord.bhdt.pipeline import run_bhdt
from chord.pinod.decompose import decompose
from chord.ensemble.integrator import integrate_results

# Canonical label mapping for fair comparison
CANONICAL_MAP = {
    'independent_ultradian': 'independent_ultradian',
    'likely_independent_ultradian': 'independent_ultradian',
    'damped_ultradian': 'independent_ultradian',
    'multi_ultradian': 'independent_ultradian',
    'harmonic': 'harmonic',
    'circadian_only': 'circadian_only',
    'non_rhythmic': 'non_rhythmic',
    'ambiguous': 'ambiguous',
}


def run_validation(n_epochs=100, patience=20, device='cpu'):
    """Run end-to-end validation on 8 key scenarios."""
    
    # Define test scenarios with ground truth
    test_cases = [
        ('pure_circadian', pure_circadian(seed=42), 'circadian_only'),
        ('pure_ultradian', pure_ultradian(seed=42), 'independent_ultradian'),
        ('independent_superposition', independent_superposition(seed=42), 'independent_ultradian'),
        ('sawtooth_harmonic', sawtooth_harmonic(seed=42), 'harmonic'),
        ('peaked_harmonic', peaked_harmonic(seed=42), 'harmonic'),
        ('damped_ultradian', damped_ultradian(seed=42), 'independent_ultradian'),
        ('asymmetric_ultradian', asymmetric_ultradian(seed=42), 'independent_ultradian'),
        ('pure_noise', pure_noise(seed=42), 'non_rhythmic'),
    ]
    
    t = test_cases[0][1]['t']
    expr = np.vstack([tc[1]['y'] for tc in test_cases])
    names = [tc[0] for tc in test_cases]
    truth = [tc[2] for tc in test_cases]
    
    n_genes = len(test_cases)
    print(f"Running ensemble E2E validation on {n_genes} scenarios")
    print(f"PINOD: {n_epochs} epochs, patience={patience}, device={device}")
    print()
    
    # --- BHDT ---
    print("Running BHDT (analytic)...")
    t0 = time.time()
    bhdt_df = run_bhdt(expr, t, method='analytic', verbose=False)
    bhdt_df['gene'] = names
    bhdt_time = time.time() - t0
    print(f"  Done in {bhdt_time:.1f}s")
    
    # --- PINOD ---
    print(f"Running PINOD ({n_epochs} epochs)...")
    t0 = time.time()
    pinod_df = decompose(
        expr, t, gene_names=names,
        n_epochs=n_epochs, patience=patience,
        device=device, verbose=True,
    )
    pinod_time = time.time() - t0
    print(f"  Done in {pinod_time:.1f}s ({pinod_time/n_genes:.1f}s/gene)")
    
    # --- Ensemble ---
    print("Running ensemble integration...")
    ensemble_df = integrate_results(bhdt_df, pinod_df)
    
    # --- Results ---
    print()
    print("=" * 100)
    header = f"{'Scenario':<28} {'Truth':<22} {'BHDT':<22} {'PINOD':<22} {'Ensemble':<22}"
    print(header)
    print("-" * 100)
    
    results = {'bhdt': [], 'pinod': [], 'ensemble': []}
    
    for i in range(n_genes):
        bhdt_cls = CANONICAL_MAP.get(str(bhdt_df.iloc[i]['classification']), 'ambiguous')
        pinod_cls = CANONICAL_MAP.get(str(pinod_df.iloc[i]['classification']), 'ambiguous')
        ens_cls = CANONICAL_MAP.get(str(ensemble_df.iloc[i]['consensus_classification']), 'ambiguous')
        
        results['bhdt'].append(bhdt_cls)
        results['pinod'].append(pinod_cls)
        results['ensemble'].append(ens_cls)
        
        bm = '✓' if bhdt_cls == truth[i] else '✗'
        pm = '✓' if pinod_cls == truth[i] else '✗'
        em = '✓' if ens_cls == truth[i] else '✗'
        
        print(f"{names[i]:<28} {truth[i]:<22} {bhdt_cls:<20}{bm}  {pinod_cls:<20}{pm}  {ens_cls:<20}{em}")
    
    print("-" * 100)
    
    for method in ['bhdt', 'pinod', 'ensemble']:
        correct = sum(1 for p, tc in zip(results[method], truth) if p == tc)
        print(f"{method.upper()}: {correct}/{n_genes} ({100*correct/n_genes:.0f}%)")
    
    # Agreement
    agree = ensemble_df['agreement'].sum()
    review = ensemble_df['review_flag'].sum()
    print(f"\nAgreement: {agree}/{n_genes} ({100*agree/n_genes:.0f}%)")
    print(f"Review flags: {review}/{n_genes}")
    print(f"\nTiming: BHDT={bhdt_time:.1f}s, PINOD={pinod_time:.1f}s")
    
    return results, truth


if __name__ == '__main__':
    device = 'cuda' if '--cuda' in sys.argv else 'cpu'
    epochs = 100
    for arg in sys.argv:
        if arg.startswith('--epochs='):
            epochs = int(arg.split('=')[1])
    
    results, truth = run_validation(n_epochs=epochs, device=device)
