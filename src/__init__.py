"""A read-only renderer for Wii NW4R layouts (.arc archives with BRLYT, BRLAN, TPL and BRFNT files).

    from wii_layout_renderer import LayoutPreviewWidget

    view = LayoutPreviewWidget()
    view.load_archive("dialog.arc", font_dirs=["Font"])
    view.preview.set_animation("dialog_in.brlan", loop=False)     # or frame=12, playing=False
    view.preview.add_state_hook(lambda state, context: ...)       # change the state before it is drawn
"""
from .animation import AnimationTrack, TrackInfo
from .formats import FormatError
from .hooks import FrameContext, HookHandle, StateHook
from .preview import LayoutPreview
from .render import LayoutRenderer
from .render.geometry import drawn_bounds
from .resources import FontLibrary, LayoutArchive, Resources, Texture
from .state import (BoundingState, Color, LayoutState, MaterialState, PaneState, PictureState, TexCoords,
                    TextBoxState, TextureMap, TextureSrt, Vec2, Vec3, WindowFrameState, WindowInflation,
                    WindowState)
from .widget import LayoutPreviewWidget

__all__ = [
    "AnimationTrack", "BoundingState", "Color", "FontLibrary", "FormatError", "FrameContext", "HookHandle",
    "LayoutArchive", "LayoutPreview", "LayoutPreviewWidget", "LayoutRenderer", "LayoutState", "MaterialState",
    "PaneState", "PictureState", "Resources", "StateHook", "TexCoords", "TextBoxState", "Texture", "TextureMap",
    "TextureSrt", "TrackInfo", "Vec2", "Vec3", "WindowFrameState", "WindowInflation", "WindowState",
    "drawn_bounds",
]
