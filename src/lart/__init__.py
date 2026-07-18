"""Latency-Response Theory model for joint accuracy/CoT-length evaluation."""

from .api import IRTResult, LaRTResult, fit_irt, fit_lart
from .estimation import (
    fisher_info_theta_lart,
    fisher_info_theta_irt,
    irt_saem_full,
    lart_saem_full,
    update_indi_fixed_all,
)
from .synthetic import SyntheticLaRTData, generate_lart_data

__all__ = [
    "IRTResult",
    "LaRTResult",
    "SyntheticLaRTData",
    "fisher_info_theta_lart",
    "fisher_info_theta_irt",
    "fit_irt",
    "fit_lart",
    "generate_lart_data",
    "irt_saem_full",
    "lart_saem_full",
    "update_indi_fixed_all",
]

__version__ = "0.1.0"
