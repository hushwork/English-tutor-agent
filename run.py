#!/usr/bin/env python3
"""Run the English Tutor — CLI or Web mode."""

import argparse
import sys
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description="English Tutor — AI-powered English learning")
    parser.add_argument(
        "--web", action="store_true",
        help="Start the web server instead of the CLI",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Web server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Web server port (default: 8080)",
    )
    args = parser.parse_args()

    if args.web:
        from english_tutor.web_server import run_web
        print(f"🌊 English Tutor Web Server")
        print(f"   Open http://{args.host}:{args.port} on your phone or computer")
        print(f"   Press Ctrl+C to stop")
        run_web(host=args.host, port=args.port)
    else:
        from english_tutor.cli import run as cli_run
        cli_run()


if __name__ == "__main__":
    main()
