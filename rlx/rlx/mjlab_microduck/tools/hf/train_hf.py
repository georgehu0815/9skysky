"""Back-compat shim: the submission logic moved to mjlab_microduck.hf_jobs.

Prefer the integrated flag:
    uv run rlx-mjlab-train <task> <train args...> --hf-jobs [--namespace <ns>] [...]

This script keeps the old invocation working:
    uv run python -m rlx.mjlab_microduck.tools.hf.train_hf <task> [submission flags] <train args...>
"""

import sys

from rlx.mjlab_microduck.hf_jobs import submit

if __name__ == "__main__":
    sys.exit(submit(sys.argv[1:]))
