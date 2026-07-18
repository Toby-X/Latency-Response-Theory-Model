"""Latency-Response Theory model for joint accuracy/CoT-length evaluation."""

from .api import IRTResult, LaRTResult, fit_irt, fit_lart
from .estimation import (
    MIRT_SAEM_full,
    cMIRT_SAEM_full,
    fisher_info_theta_c,
    fisher_info_theta_irt,
    update_indi_fixed_all,
)
from .synthetic import SyntheticLaRTData, generate_lart_data

__all__ = [
    "IRTResult",
    "LaRTResult",
    "MIRT_SAEM_full",
    "SyntheticLaRTData",
    "cMIRT_SAEM_full",
    "fisher_info_theta_c",
    "fisher_info_theta_irt",
    "fit_irt",
    "fit_lart",
    "generate_lart_data",
    "update_indi_fixed_all",
]

__version__ = "0.1.0"
