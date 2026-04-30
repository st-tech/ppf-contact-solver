# File: console.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Console message queue singleton.
# Extracted from ui/console.py to break core/ → ui/ dependency.

from datetime import datetime
from threading import Lock

import bpy  # pyright: ignore

from .groups import get_addon_data


class Console:
    """Class to manage the console text editor."""

    def __init__(self):
        from ..core.utils import get_category_name

        self.console_name = get_category_name()
        self.messages = []
        self.lock = Lock()

    def get_or_create(self):
        """Get or create the console text object."""
        console_text = bpy.data.texts.get(self.console_name)
        if not console_text:
            console_text = bpy.data.texts.new(self.console_name)
        return console_text

    def show(self, last_lines=10):
        """Show the console in a new or existing window and scroll to the last 10 lines."""
        console_text = self.get_or_create()

        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if (
                    area.type == "TEXT_EDITOR"
                    and area.spaces.active.text == console_text
                ):
                    area.spaces.active.top = max(
                        0, len(console_text.lines) - last_lines
                    )
                    return

        bpy.ops.wm.window_new()
        new_window = bpy.context.window_manager.windows[-1]
        for area in new_window.screen.areas:
            if area.type != "TEXT_EDITOR":
                area.type = "TEXT_EDITOR"
            area.spaces.active.show_line_numbers = True
            area.spaces.active.show_syntax_highlight = False
            area.spaces.active.text = console_text
            area.spaces.active.top = max(0, len(console_text.lines) - last_lines)

    def write(self, message, timestamp=True):
        """Add a message to the queue."""
        with self.lock:
            if timestamp:
                timestamp_str = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                for line in message.splitlines():
                    if line:
                        self.messages.append(f"{timestamp_str} {line}")
            else:
                self.messages.append(message)

    def process_messages(self):
        """Write messages to the console text."""
        with self.lock:
            if self.messages:
                console_text = self.get_or_create()
                while self.messages:
                    message = self.messages.pop(0)
                    console_text.write(f"{message}\n")
                    scene = bpy.context.scene
                    addon_data = get_addon_data(scene)
                    state = getattr(addon_data, "state", None) if addon_data else None
                    log_path = getattr(state, "log_file_path", "") if state else ""
                    if log_path:
                        try:
                            import os
                            log_dir = os.path.dirname(log_path)
                            if log_dir and not os.path.exists(log_dir):
                                os.makedirs(log_dir, exist_ok=True)

                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(f"{message}\n")
                        except Exception as e:
                            print(f"[Console] Failed to write log: {e}")

                max_lines = get_addon_data(bpy.context.scene).state.max_console_lines + 1
                if len(console_text.lines) > max_lines:
                    new_content = "\n".join(
                        line.body for line in console_text.lines[-max_lines:]
                    )
                    console_text.clear()
                    console_text.from_string(new_content)


console = Console()
