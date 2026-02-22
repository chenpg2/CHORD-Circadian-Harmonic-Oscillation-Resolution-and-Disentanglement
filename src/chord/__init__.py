"""
CHORD — Circadian Harmonic Oscillation Resolver & Disentangler.

A statistical framework for detecting, disentangling, and decomposing
circadian and ultradian rhythms in transcriptomic data.

Main public API
---------------
detect        : Rhythm detection via cosinor / harmonic regression.
disentangle   : Bayesian Harmonic Disentanglement (BHDT).
decompose     : Physics-Informed Neural ODE Decomposition (PINOD).
"""

__version__ = "1.0.0"


def __getattr__(name: str):
    """Lazy-import public API so optional deps don't fail on package import."""
    if name == "detect":
        from chord.bhdt.detect import detect
        return detect
    if name == "disentangle":
        from chord.bhdt.disentangle import disentangle
        return disentangle
    if name == "decompose":
        from chord.pinod.decompose import decompose
        return decompose
    if name == "run":
        from chord.pipeline import run
        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["detect", "disentangle", "decompose", "run", "__version__"]
