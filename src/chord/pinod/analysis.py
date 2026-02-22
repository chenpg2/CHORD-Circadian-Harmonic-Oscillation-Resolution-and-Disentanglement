"""
PINOD analysis: oscillator parameter extraction and interpretation.

After training, extract biologically meaningful parameters from the
learned ODE system and classify genes based on their oscillatory profiles.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


def extract_oscillator_params(
    model,
    expr: np.ndarray,
    timepoints: np.ndarray,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Extract full oscillator parameters from a trained PINOD model.

    Parameters
    ----------
    model : PINODSingleGene (trained)
    expr : (N,) expression values
    timepoints : (N,) time points

    Returns
    -------
    dict with per-oscillator parameters and classification
    """
    from torchdiffeq import odeint
    from chord.pinod.networks import LinearDecoder

    model.eval()
    model = model.to(device)

    if expr.ndim == 1:
        expr = expr.reshape(1, -1)

    expr_t = torch.tensor(expr, dtype=torch.float32, device=device)
    t_eval = torch.tensor(timepoints, dtype=torch.float32, device=device)

    expr_mean = expr_t.mean()
    expr_std = expr_t.std() + 1e-8
    expr_norm = (expr_t - expr_mean) / expr_std

    with torch.no_grad():
        expr_hat, trajectories, amplitudes = model(expr_norm, t_eval)
        expr_hat_denorm = expr_hat * expr_std + expr_mean

        displacements = trajectories[:, :, 0::2]  # (T, 1, K)
        velocities = trajectories[:, :, 1::2]

        K = model.K
        periods = model.ode.periods.cpu().numpy()
        gammas = model.ode.gamma.cpu().numpy()

        # Get decoder mixing weights if available
        has_linear_decoder = isinstance(model.decoder, LinearDecoder)
        if has_linear_decoder:
            mixing_weights = model.decoder.weights.abs().cpu().numpy()
        else:
            mixing_weights = np.ones(K)

        oscillators = []
        for k in range(K):
            disp_k = displacements[:, 0, k].cpu().numpy()
            vel_k = velocities[:, 0, k].cpu().numpy()

            amp_rms = float(np.sqrt(np.mean(disp_k ** 2)))
            amp_peak = float(np.max(np.abs(disp_k)))

            # Effective contribution = displacement amplitude * mixing weight
            effective_amp = amp_rms * float(mixing_weights[k])

            if amp_peak > 1e-8:
                peak_idx = np.argmax(disp_k)
                phase_hours = float(timepoints[peak_idx] % periods[k])
                phase_rad = float(2 * np.pi * phase_hours / periods[k])
            else:
                phase_hours = 0.0
                phase_rad = 0.0

            omega_k = 2 * np.pi / periods[k]
            energy = 0.5 * vel_k ** 2 + 0.5 * omega_k ** 2 * disp_k ** 2
            mean_energy = float(np.mean(energy))

            corr_norm = float(sum(
                p.pow(2).sum().item()
                for p in model.ode.corrections[k].parameters()
            ))

            oscillators.append({
                "index": k,
                "period": float(periods[k]),
                "gamma": float(gammas[k]),
                "amplitude_rms": amp_rms,
                "amplitude_peak": amp_peak,
                "effective_amplitude": effective_amp,
                "mixing_weight": float(mixing_weights[k]),
                "phase_rad": phase_rad,
                "phase_hours": phase_hours,
                "mean_energy": mean_energy,
                "correction_norm": corr_norm,
                "displacement": disp_k,
                "velocity": vel_k,
            })

        mse = float(((expr_hat_denorm.cpu().numpy() - expr) ** 2).mean())
        r2 = float(1.0 - mse / max(np.var(expr), 1e-10))

    # Classify oscillators as active/inactive using EFFECTIVE amplitude
    # Use a relative threshold: active if effective_amp > 15% of max
    eff_amps = [o["effective_amplitude"] for o in oscillators]
    max_eff = max(eff_amps) if eff_amps else 1.0
    for osc in oscillators:
        osc["active"] = osc["effective_amplitude"] > 0.15 * max_eff

    # Also compute variance explained by each oscillator
    total_var = float(np.var(expr))
    for osc in oscillators:
        osc_signal = osc["displacement"] * osc["mixing_weight"]
        osc["var_explained"] = float(np.var(osc_signal)) / max(total_var, 1e-10)

    active_periods = [o["period"] for o in oscillators if o["active"]]

    return {
        "oscillators": oscillators,
        "active_periods": active_periods,
        "n_active": sum(1 for o in oscillators if o["active"]),
        "reconstruction_mse": mse,
        "reconstruction_r2": r2,
        "expr_hat": expr_hat_denorm.cpu().numpy().flatten(),
    }


def classify_gene_pinod(analysis_result: Dict) -> Dict[str, Any]:
    """Classify a gene based on PINOD oscillator analysis.

    Uses effective amplitude (displacement * mixing weight) and variance
    explained to determine which oscillators are truly contributing.

    Categories:
      - 'independent_ultradian': active 12h oscillator with significant contribution
      - 'harmonic': 12h signal explained by 24h nonlinear correction
      - 'circadian_only': only 24h oscillator contributes significantly
      - 'multi_ultradian': multiple ultradian oscillators active
      - 'damped_ultradian': 12h active but high gamma (decaying)
      - 'non_rhythmic': no active oscillators or poor reconstruction
    """
    oscillators = analysis_result["oscillators"]
    n_active = analysis_result["n_active"]
    r2 = analysis_result["reconstruction_r2"]

    # Poor reconstruction = non-rhythmic
    # R2 < 0.55 indicates the model can't explain the signal well,
    # suggesting no strong rhythmic component
    if r2 < 0.55:
        return {
            "classification": "non_rhythmic",
            "confidence": 0.8,
            "evidence": f"Poor reconstruction (R2={r2:.3f})",
        }

    if n_active == 0:
        return {
            "classification": "non_rhythmic",
            "confidence": 0.9,
            "evidence": "No active oscillators",
        }

    # Categorise oscillators by period range
    circ = [o for o in oscillators if 20 < o["period"] < 28 and o["active"]]
    ultra_12 = [o for o in oscillators if 10 < o["period"] < 14 and o["active"]]
    ultra_8 = [o for o in oscillators if 6 < o["period"] < 10 and o["active"]]

    has_circ = len(circ) > 0
    has_12h = len(ultra_12) > 0
    has_8h = len(ultra_8) > 0

    # Use variance explained as the primary discriminator
    var_circ = max((o["var_explained"] for o in circ), default=0.0)
    var_12h = max((o["var_explained"] for o in ultra_12), default=0.0)
    var_8h = max((o["var_explained"] for o in ultra_8), default=0.0)
    total_var_explained = var_circ + var_12h + var_8h

    # Relative contributions
    if total_var_explained > 0:
        frac_circ = var_circ / total_var_explained
        frac_12h = var_12h / total_var_explained
        frac_8h = var_8h / total_var_explained
    else:
        frac_circ = frac_12h = frac_8h = 0.0

    # Significance threshold: oscillator must explain >10% of total variance
    sig_threshold = 0.10

    sig_circ = frac_circ > sig_threshold
    sig_12h = frac_12h > sig_threshold
    sig_8h = frac_8h > sig_threshold

    # Check damping for 12h
    if has_12h:
        gamma_12 = ultra_12[0]["gamma"]
        is_damped = gamma_12 > 0.05
    else:
        gamma_12 = 0.0
        is_damped = False

    # Check harmonic ratio: if periods are in exact integer ratios AND
    # the circadian component strongly dominates, the sub-harmonics are
    # likely mathematical artifacts of a non-sinusoidal circadian waveform.
    #
    # Key principle: for INDEPENDENT 12h oscillators, the 12h variance
    # should be comparable to 24h (ratio > 0.5). For harmonics, 12h
    # variance is much smaller than 24h (ratio < 0.35).
    is_harmonic = False
    harmonic_evidence = ""
    if has_circ and has_12h:
        T_circ = circ[0]["period"]
        ratio_12 = T_circ / ultra_12[0]["period"]
        var_ratio = var_12h / max(var_circ, 1e-10)

        # Harmonic if: period ratio ~2.0 AND 12h variance is small relative to 24h
        if abs(ratio_12 - 2.0) < 0.10 and var_ratio < 0.35:
            is_harmonic = True
            harmonic_evidence = f"T_24/T_12={ratio_12:.2f}~2.0, 12h/24h var ratio={var_ratio:.2f} (harmonic-like)"

    # Classification decision tree
    if not sig_circ and not sig_12h and not sig_8h:
        return {
            "classification": "non_rhythmic",
            "confidence": 0.75,
            "evidence": f"No oscillator explains >{sig_threshold*100:.0f}% variance",
        }

    if sig_circ and not sig_12h and not sig_8h:
        return {
            "classification": "circadian_only",
            "confidence": 0.8,
            "evidence": f"24h explains {frac_circ*100:.0f}% of variance",
        }

    # Harmonic detection: circadian dominant + integer period ratios
    if is_harmonic:
        return {
            "classification": "harmonic",
            "confidence": 0.75,
            "evidence": harmonic_evidence,
        }

    if sig_12h and is_damped:
        return {
            "classification": "damped_ultradian",
            "confidence": 0.75,
            "evidence": f"12h active but damped (gamma={gamma_12:.4f})",
            "gamma_12": gamma_12,
        }

    # Multiple ultradian: only if circadian is NOT dominant AND
    # the secondary ultradian is genuinely significant (not just noise)
    n_sig_ultra = sum([sig_12h, sig_8h])
    if n_sig_ultra >= 2 and frac_circ < 0.35 and min(frac_12h, frac_8h) > 0.20:
        return {
            "classification": "multi_ultradian",
            "confidence": 0.7,
            "evidence": f"Multiple ultradian oscillators significant",
        }

    if sig_12h:
        conf = min(0.9, 0.5 + frac_12h * 0.5)
        return {
            "classification": "independent_ultradian",
            "confidence": conf,
            "evidence": f"Independent 12h ({frac_12h*100:.0f}% variance, gamma={gamma_12:.4f})",
        }

    if sig_8h:
        return {
            "classification": "independent_ultradian",
            "confidence": 0.6,
            "evidence": f"Independent 8h ({frac_8h*100:.0f}% variance)",
        }

    return {
        "classification": "ambiguous",
        "confidence": 0.3,
        "evidence": "Could not determine classification",
    }
