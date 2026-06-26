#!/usr/bin/env python3
"""Restore a saved workspace layout — launch programs and position them.
Usage: restore-workspace-layout.py [layout-file]

Reads the layout JSON, launches each program with its stored command,
then applies the saved position/size via swaymsg.
"""

import json
import os
import subprocess
import sys
import time

LAYOUT_FILE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.environ.get("ENGLISH_TUTOR_DATA_DIR", os.path.dirname(os.path.abspath(__file__))),
    "workspace-layout.json",
)


def load_layout(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def window_exists(app_id: str, title: str) -> bool:
    """Check if a window matching app_id (or title) already exists."""
    result = subprocess.run(
        ["swaymsg", "-t", "get_tree"], capture_output=True, text=True
    )
    tree = json.loads(result.stdout)

    def search(node):
        if node.get("app_id") == app_id or node.get("name") == title:
            return True
        for child in node.get("nodes", []) + node.get("floating_nodes", []):
            if search(child):
                return True
        return False

    return search(tree)


def wait_for_window(app_id: str, title: str, timeout: float = 5.0) -> bool:
    """Poll until a window with the given app_id (or title) appears."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if window_exists(app_id, title):
            return True
        time.sleep(0.2)
    return False


def position_window(app_id: str, title: str, x: int, y: int, w: int, h: int):
    """Use swaymsg to move and resize a matching window."""
    # Try matching by app_id first, then by title
    criteria = f'app_id="{app_id}"' if app_id else f'title="{title}"'

    # Move to position
    subprocess.run(
        ["swaymsg", criteria, "move", "position", str(x), str(y)],
        capture_output=True,
    )
    # Resize
    subprocess.run(
        ["swaymsg", criteria, "resize", "set", str(w), str(h)],
        capture_output=True,
    )

    print(f"   ✓ positioned: {app_id or title}")


def main():
    if not os.path.exists(LAYOUT_FILE):
        print(f"❌ Layout file not found: {LAYOUT_FILE}")
        print("   First, arrange your windows, then run: save-layout.py")
        sys.exit(1)

    layout = load_layout(LAYOUT_FILE)
    ws = layout.get("workspace", "1")
    windows = layout.get("windows", [])

    if not windows:
        print("⚠️  No windows in layout file")
        return

    print(f"📐 Restoring workspace '{ws}' layout ({len(windows)} windows)...")

    # 1. Switch to the target workspace
    subprocess.run(["swaymsg", "workspace", ws])

    # 2. Launch each program (if command is known and window doesn't exist)
    for win in windows:
        cmd = win.get("command", "").strip()
        app_id = win.get("app_id", "")
        title = win.get("title", "")

        if not cmd:
            print(f"   ⚠️  Skipping {app_id or title}: no command configured")
            print(f"      Edit {LAYOUT_FILE} and add a 'command' field")
            continue

        if window_exists(app_id, title):
            print(f"   - {app_id or title}: already running")
            continue

        print(f"   🔄 Launching: {cmd}")
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 3. Wait for all windows to appear, then position them
    print("   ⏳ Waiting for windows to appear...")
    for win in windows:
        app_id = win.get("app_id", "")
        title = win.get("title", "")
        cmd = win.get("command", "").strip()

        if not cmd:
            continue

        if wait_for_window(app_id, title, timeout=8.0):
            position_window(
                app_id, title,
                win["x"], win["y"],
                win["width"], win["height"],
            )
        else:
            print(f"   ⚠️  Timeout waiting for: {app_id or title}")

    print("✅ Layout restored!")


if __name__ == "__main__":
    main()
