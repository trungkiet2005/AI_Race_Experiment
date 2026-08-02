#!/usr/bin/env python3
"""Apply the Windows scrollbar-width fix to a canonical portable report build."""

from __future__ import annotations

import argparse
from pathlib import Path


STYLE = "<style data-ai-race-windows-overflow-fix>html,body{overflow-x:hidden!important;max-width:100%!important}</style>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("html", type=Path)
    args = parser.parse_args()
    content = args.html.read_text(encoding="utf-8")
    if STYLE not in content:
        if "</head>" not in content:
            raise ValueError("Portable report has no </head> marker")
        content = content.replace("</head>", STYLE + "\n</head>", 1)
        args.html.write_text(content, encoding="utf-8")
    print(args.html)


if __name__ == "__main__":
    main()
