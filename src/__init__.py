"""Discrete-time survival: evaluation-interpolation audit.

Self-contained library implementing the components of the paper
"Naive Survival Interpolation Fakes Boundary-Geometry Effects in
Discrete-Time Survival".

Modules
-------
survival        : survival head, discrete NLL, risk, and the fixed interpolation
boundaries     : interval-boundary strategies (quantile/uniform/alltime/log/custom/AAIC)
interpolation   : the four survival-interpolation schemes (naive/step/linear/pc)
metrics         : Harrell C-index, KM, IPCW AUC/Brier, Uno C-index
stats           : paired tests for non-independent folds + BH-FDR
"""

__version__ = "0.1.0"