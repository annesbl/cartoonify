from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn


def main() -> None:
    # Projekt-Root = Ordner dieser Datei
    root = Path(__file__).resolve().parent

    # Sicherstellen, dass "backend" als Package importierbar ist
    os.chdir(str(root))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # String-Import ist für uvicorn am robustesten
    uvicorn.run(
        "backend.app:create_app",
        host="127.0.0.1",
        port=8000,
        factory=True,
        reload=False,  # auf CPU lieber erstmal aus
        log_level="info",
    )


if __name__ == "__main__":
    main()
