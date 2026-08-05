"""ComfyUI-H3-SeedScout — MiniMax H3 seed scouting sampler (v2, interactive)."""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Served as /extensions/ComfyUI-H3-SeedScout/*.js (nodes.py:2279, EXTENSION_WEB_DIRS).
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
