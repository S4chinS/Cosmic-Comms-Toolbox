"""Helpers for starting the Orekit JVM with optional extra classpath entries.

This repo uses python-orekit (JCC). The JVM classpath must be finalized before
the first `orekit.initVM()` call; after the JVM starts, adding jars is not
reliable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import orekit  # type: ignore

_JVM_STARTED = False


def ensure_orekit_vm_started(*, extra_jars: Iterable[str | Path] = ()) -> None:
    """Start the JVM once, appending any extra jars to the classpath beforehand.

    - **extra_jars**: jar paths to add (if they exist)
    """
    global _JVM_STARTED

    jars: list[str] = []
    for j in extra_jars:
        p = Path(j).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Required jar not found: {p}")
        jars.append(str(p))

    if not _JVM_STARTED:
        if jars:
            # Append to the JCC classpath before initVM starts the JVM.
            cp = getattr(orekit, "_orekit").CLASSPATH
            for j in jars:
                if j not in cp:
                    cp = cp + os.pathsep + j
            getattr(orekit, "_orekit").CLASSPATH = cp

        orekit.initVM()
        _JVM_STARTED = True
    elif jars:
        # JVM is already running; classpath changes won't reliably apply.
        cp = getattr(getattr(orekit, "_orekit"), "CLASSPATH", "")
        missing = [j for j in jars if j not in cp]
        if missing:
            raise RuntimeError(
                "JVM already started without required jar(s). Restart the app so they are "
                f"picked up on the initial orekit.initVM(): {missing}"
            )

    # Ensure the current thread is attached (safe to call repeatedly).
    orekit.getVMEnv().attachCurrentThread()

