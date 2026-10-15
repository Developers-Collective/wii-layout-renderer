"""Where a layout's files come from: its archive, and fonts that live outside it."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, TypeVar, Union

import numpy as np

from .formats import Brfnt, FormatError, Layout, LayoutAnimation, U8, read_brlan, read_brlyt, read_tpl

log = logging.getLogger(__name__)

FileSource = Union[str, PathLike, bytes]
T = TypeVar("T")


@dataclass(frozen=True)
class Texture:
    """``pixels`` is a read-only (height, width, 4) RGBA uint8 array."""

    pixels: np.ndarray
    has_alpha: bool

    @property
    def width(self) -> int:
        return self.pixels.shape[1]

    @property
    def height(self) -> int:
        return self.pixels.shape[0]


def _read_texture(data: bytes) -> Texture:
    pixels = read_tpl(data)
    return Texture(pixels, bool((pixels[..., 3] != 255).any()))


def _base_name(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1].lower()


class LayoutArchive:
    """A layout archive (.arc). Files are looked up by file name, in any folder and any case."""

    def __init__(self, archive: U8, source: str = "<memory>"):
        self.source = source
        self._files = archive.files
        self._by_name: dict[str, str] = {}
        for path in self._files:  # the first file with a name wins
            self._by_name.setdefault(_base_name(path), path)
        self._cache: dict[tuple[str, str], object] = {}

    @classmethod
    def open(cls, source: FileSource) -> LayoutArchive:
        """Read an archive from a path or from its bytes."""
        if isinstance(source, (bytes, bytearray, memoryview)):
            return cls(U8.from_bytes(bytes(source)))
        return cls(U8.load_from(source), os.fspath(source))

    def __repr__(self) -> str:
        return f"<LayoutArchive {self.source!r}>"

    def find(self, name: str) -> Optional[bytes]:
        path = self._by_name.get(_base_name(name))
        return self._files[path] if path is not None else None

    def layout_names(self) -> list[str]:
        return self._names_with_suffix(".brlyt")

    def animation_names(self) -> list[str]:
        return self._names_with_suffix(".brlan")

    def layout(self, name: str) -> Layout:
        """Raises KeyError when the layout is missing, FormatError when it is damaged."""
        return self._required("layout", name, read_brlyt)

    def animation(self, name: str) -> LayoutAnimation:
        """Raises KeyError when the animation is missing, FormatError when it is damaged."""
        return self._required("animation", name, read_brlan)

    def texture(self, name: str) -> Optional[Texture]:
        """None when the texture is missing or damaged."""
        return self._optional("texture", name, _read_texture)

    def font(self, name: str) -> Optional[Brfnt]:
        """None when the archive has no such font or it is damaged."""
        return self._optional("font", name, lambda data: Brfnt(data, name))

    def _names_with_suffix(self, suffix: str) -> list[str]:
        return [path.rsplit("/", 1)[-1] for path in self._files if path.lower().endswith(suffix)]

    def _required(self, kind: str, name: str, read: Callable[[bytes], T]) -> T:
        key = (kind, _base_name(name))
        if key not in self._cache:
            data = self.find(name)
            if data is None:
                raise KeyError(f"{self.source} has no file named {name!r}.")
            self._cache[key] = read(data)
        return self._cache[key]

    def _optional(self, kind: str, name: str, read: Callable[[bytes], T]) -> Optional[T]:
        key = (kind, _base_name(name))
        if key not in self._cache:
            data = self.find(name)
            try:
                self._cache[key] = read(data) if data is not None else None
            except FormatError as error:
                log.warning("Cannot read %s from %s: %s", name, self.source, error)
                self._cache[key] = None
        return self._cache[key]


class FontLibrary:
    """Fonts outside layout archives, like the game's Font folder.

    ``fonts`` maps font file names to a path or the font's bytes; ``font_dirs`` are searched for a file with
    the font's name in any case. Given fonts come before the folders.
    """

    def __init__(self, fonts: Optional[Mapping[str, FileSource]] = None, font_dirs: Iterable[FileSource] = ()):
        self._sources = {_base_name(name): source for name, source in (fonts or {}).items()}
        self._dirs = [Path(directory) for directory in font_dirs]
        self._dir_files: Optional[dict[str, Path]] = None
        self._fonts: dict[str, Optional[Brfnt]] = {}

    def font(self, name: str) -> Optional[Brfnt]:
        key = _base_name(name)
        if key not in self._fonts:
            self._fonts[key] = self._load(key)
        return self._fonts[key]

    def _load(self, key: str) -> Optional[Brfnt]:
        source = self._sources.get(key) or self._files_in_dirs().get(key)
        if source is None:
            return None
        try:
            if isinstance(source, (bytes, bytearray, memoryview)):
                return Brfnt(source, key)
            return Brfnt.from_file(source)
        except (OSError, FormatError) as error:
            log.warning("Cannot read font %s: %s", key, error)
            return None

    def _files_in_dirs(self) -> dict[str, Path]:
        if self._dir_files is None:
            self._dir_files = {}
            for directory in self._dirs:
                try:
                    entries = sorted(directory.iterdir())
                except OSError as error:
                    log.warning("Cannot list font folder %s: %s", directory, error)
                    continue
                for entry in entries:
                    self._dir_files.setdefault(entry.name.lower(), entry)
        return self._dir_files


class Resources:
    """The textures and fonts a layout draws with.

    Fonts come from the font library before the archive, the order an NW4R text box looks them up in: the
    fonts the game registers globally, then a font file inside the layout's archive.
    """

    def __init__(self, archive: Optional[LayoutArchive] = None, fonts: Optional[FontLibrary] = None):
        self.archive = archive
        self.fonts = fonts or FontLibrary()

    def texture(self, name: str) -> Optional[Texture]:
        return self.archive.texture(name) if self.archive is not None else None

    def font(self, name: str) -> Optional[Brfnt]:
        font = self.fonts.font(name)
        if font is None and self.archive is not None:
            font = self.archive.font(name)
        return font
