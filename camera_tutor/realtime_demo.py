#!/usr/bin/env python3
"""Camera Tutor — 实时语音对话 (WebSocket).

Thin entry point for CameraTutorAgent. The agent class handles
all lifecycle, connection management, and sub-manager orchestration.

运行:
  python3 camera_tutor/realtime_demo.py

依赖:
  pip install websocket-client sounddevice numpy
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from camera_tutor.agent import CameraTutorAgent, AgentConfig


def main():
    """Entry point: configure, setup, and run the agent."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy library logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)

    config = AgentConfig()
    agent = CameraTutorAgent(config=config)

    print("=" * 55)
    print("  Camera Tutor — 实时语音对话")
    print("  (WebSocket · 服务端 VAD · 实时打断)")
    print("=" * 55)
    print()

    try:
        agent.setup()
        agent.start()
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()
        print("\n\n👋 再见！")


if __name__ == "__main__":
    main()
