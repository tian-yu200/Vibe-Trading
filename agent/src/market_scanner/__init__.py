"""A-share market scanner core (CONTRACT.md sections 4-7).

Package layout:

* ``models``     — identifier helpers, funnel stage constants, envelope builders.
* ``config``     — :class:`~src.market_scanner.config.ScanConfig` with validation.
* ``exclusions`` — ordered exclusion funnel (CONTRACT section 5).
* ``scoring``    — cross-sectional percentile sub-scores and composite (section 6).
* ``pipeline``   — ``run_scan`` orchestrating injectable sources into a ScanRun (section 7).
* ``review``     — AI review step (landed in a later task).

Data source adapters live in ``market_scanner.sources`` (owned elsewhere).
"""
