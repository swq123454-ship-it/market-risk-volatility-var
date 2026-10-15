"""Market-risk analytics: volatility forecasting, VaR/ES, and their backtests.

The package is a study rather than a library, so the modules are ordered the way
the analysis runs:

``config``     every tunable, in one place, echoed into the results bundle
``generator``  the simulated market and its data-generating process
``data``       panel assembly for synthetic and real data modes
``features``   realised measures and the strictly-lagged design matrix
``kernels``    the recursive variance loops, with an optional Numba fast path
``nn``         a from-scratch GRU with an analytic backward pass
``volmodels``  the forecast family and the walk-forward engine
``var``        five VaR routes plus Expected Shortfall
``backtest``   Kupiec, Christoffersen, Engle-Manganelli, Basel traffic light
``losses``     QLIKE, MSE, tick loss, Diebold-Mariano
``evaluate``   model comparison and calibration tests
``plots``      figures
``report``     the end-to-end pipeline and the results bundle
``api``        a small risk-engine service over the fitted filter
"""

__all__ = [
    "api",
    "backtest",
    "config",
    "data",
    "evaluate",
    "features",
    "generator",
    "kernels",
    "losses",
    "nn",
    "plots",
    "report",
    "var",
    "volmodels",
]

__version__ = "1.0.0"
