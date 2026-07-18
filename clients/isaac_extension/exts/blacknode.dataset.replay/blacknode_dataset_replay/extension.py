"""Isaac Sim menu integration for the Blacknode dataset replay client."""
from pathlib import Path

import omni.ext
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items


class Extension(omni.ext.IExt):
    def on_startup(self, _ext_id):
        self._client = {}
        self._menu = [MenuItemDescription(
            name="Blacknode Dataset Replay",
            onclick_fn=self._show_window,
        )]
        add_menu_items(self._menu, "Window")

    def _client_script(self):
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "isaac_sim_stream.py"
            if candidate.is_file():
                return candidate
        raise FileNotFoundError("isaac_sim_stream.py was not found beside the Isaac extension")

    def _show_window(self):
        if "show_blacknode_isaac_window" not in self._client:
            script = self._client_script()
            self._client["__file__"] = str(script)
            self._client["__name__"] = "blacknode_isaac_stream"
            exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), self._client)
        self._client["show_blacknode_isaac_window"]()

    def on_shutdown(self):
        remove_menu_items(self._menu, "Window")
        stop = self._client.get("stop_blacknode_isaac_stream")
        if stop:
            stop()
        window = self._client.get("_window")
        if window is not None:
            window.visible = False
        self._client.clear()
