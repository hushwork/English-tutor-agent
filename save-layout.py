#!/usr/bin/env python3
"""Save current workspace layout to a JSON file.
Usage: save-workspace-layout.py [workspace] [output-file]

Captures all windows in the given workspace (default "1") along with
their app_id, title, position, size — so the layout can be restored later.
"""

import json
import os
import subprocess
import sys

WORKSPACE = sys.argv[1] if len(sys.argv) > 1 else "1"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser(
    os.environ.get("ENGLISH_TUTOR_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
    + "/workspace-layout.json"
)

def get_tree():
    result = subprocess.run(["swaymsg", "-t", "get_tree"], capture_output=True, text=True)
    return json.loads(result.stdout)

def find_workspace(node, name):
    if node.get("type") == "workspace" and node.get("name") == name:
        return node
    for child in node.get("nodes", []) + node.get("floating_nodes", []):
        found = find_workspace(child, name)
        if found:
            return found
    return None

def extract_windows(workspace_node, windows):
    """Extract top-level layout slots from a workspace.
    Each direct child of the workspace represents one window slot."""
    for child in workspace_node.get("nodes", []) + workspace_node.get("floating_nodes", []):
        rect = child.get("rect", {})
        if rect.get("width", 0) <= 50 and rect.get("height", 0) <= 50:
            continue

        pid = child.get("pid")
        app_id = child.get("app_id") or ""
        name = child.get("name") or ""

        # If this child has a PID it's a real window
        if pid is not None:
            windows.append({
                "app_id": app_id,
                "title": name,
                "pid": pid,
                "x": rect.get("x", 0),
                "y": rect.get("y", 0),
                "width": rect.get("width", 0),
                "height": rect.get("height", 0),
            })
        else:
            # Layout container (split/stack/tabbed) — use its bounding box
            # and try to find app_id from its first real child
            first_child = _find_first_window(child)
            windows.append({
                "app_id": first_child.get("app_id", "") if first_child else app_id,
                "title": first_child.get("name", name) if first_child else name,
                "pid": None,
                "x": rect.get("x", 0),
                "y": rect.get("y", 0),
                "width": rect.get("width", 0),
                "height": rect.get("height", 0),
            })


def _find_first_window(node):
    """Find the first real window (has PID) in a node tree."""
    pid = node.get("pid")
    if pid is not None:
        return node
    for child in node.get("nodes", []) + node.get("floating_nodes", []):
        result = _find_first_window(child)
        if result:
            return result
    return None


def get_command_for_window(win) -> str:
    """Try to guess the launch command from app_id/title.
    Returns empty string if unknown — user should fill it in manually."""
    app = win["app_id"]
    title = win["title"]

    known = {
        "foot": "foot",
        "firefox": "firefox",
        "firefox_firefox": "firefox",
        "chromium": "chromium",
        "chromium-browser": "chromium-browser",
        "code": "code",
        "code-oss": "code",
        "Alacritty": "alacritty",
        "kitty": "kitty",
        "thunar": "thunar",
        "nautilus": "nautilus",
        "pcmanfm": "pcmanfm",
    }
    if app in known:
        return known[app]
    return ""


def main():
    tree = get_tree()
    ws = find_workspace(tree, WORKSPACE)
    if not ws:
        print(f"❌ Workspace '{WORKSPACE}' not found")
        sys.exit(1)

    windows = []
    extract_windows(ws, windows)

    layout = {
        "workspace": WORKSPACE,
        "layout": ws.get("layout", "splith"),
        "windows": [],
    }

    for w in windows:
        entry = {
            "app_id": w["app_id"],
            "title": w["title"],
            "x": w["x"],
            "y": w["y"],
            "width": w["width"],
            "height": w["height"],
            "command": get_command_for_window(w),
        }
        layout["windows"].append(entry)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(layout, f, indent=2, ensure_ascii=False)

    print(f"✅ Workspace '{WORKSPACE}' layout saved to {OUTPUT}")
    print(f"   {len(layout['windows'])} window(s) captured:")
    for w in layout["windows"]:
        cmd = w["command"] or "⚠️  need to fill in command"
        print(f"   · {w['app_id'] or w['title']:25s}  {w['width']}x{w['height']} @ ({w['x']},{w['y']})  → {cmd}")
    print()
    print("📝 提示: 编辑该 JSON 文件, 给每个窗口填上正确的 command, 恢复时用.")


if __name__ == "__main__":
    main()
