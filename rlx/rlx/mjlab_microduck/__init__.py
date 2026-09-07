"""MicroDuck task suite for mjlab, distributed as an optional RLX integration."""

from __future__ import annotations

import sys

# The upstream task suite historically imports ``mjlab_microduck`` as a
# top-level package. Preserve that internal contract while packaging it under
# ``rlx`` so existing task modules and external mjlab metadata keep working.
sys.modules.setdefault("mjlab_microduck", sys.modules[__name__])
