"""Market data source adapters for the A-share scanner.

Owned per CONTRACT.md section 1 (hermes). Every public fetch function follows
the section 3 rules:

* Never raises on network or parse errors — returns empty/partial data plus a
  :class:`~sources.status.SourceStatus` of ``failed`` or ``degraded``.
* Fetchers are injectable (``fetch``/``session`` params) for offline tests.
* Frames are indexed by ``code`` (6-digit string).
* Money is CNY floats; percent fields are percent numbers (12.5 == 12.5%).
"""

from __future__ import annotations
