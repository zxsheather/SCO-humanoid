from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Callable


def close_summary_writer(runner: object) -> None:
    writer = getattr(runner, "writer", None)
    if writer is None:
        return
    writer.flush()
    writer.close()


def run_with_isaac_exit_guard(main: Callable[[], int]) -> None:
    """Exit without running Isaac Gym's fragile interpreter teardown path."""
    try:
        code = int(main())
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
