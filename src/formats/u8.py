"""Wii U8 archives (.arc), read-only."""
from __future__ import annotations

import struct
from collections import deque
from os import PathLike
from typing import NamedTuple, Union

from .binary_reader import FormatError

U8_MAGIC = b"U\xaa8-"


class U8:
    class Header(NamedTuple):
        magic: bytes
        file_system_table_offset: int
        file_system_table_length: int
        file_data_offset: int

        @classmethod
        def unpack_from(cls, data: bytes) -> U8.Header:
            return cls(*struct.unpack_from(">4sIII", data, 0))

    class FileEntry(NamedTuple):
        # is_directory = bool8 always 0
        name_offset: int
        data_offset: int
        data_length: int

    class DirectoryEntry(NamedTuple):
        # is_directory = bool8 always 1
        name_offset: int
        parent_entry_index: int
        skip_index: int

    ENTRY_SIZE = 12

    header: Header
    files: dict[str, bytes]

    @staticmethod
    def load_from(file_path: Union[str, PathLike]) -> U8:
        with open(file_path, "rb") as file:
            return U8.from_bytes(file.read())

    @staticmethod
    def from_bytes(data: bytes) -> U8:
        u8 = U8()
        u8.unpack(bytes(data))
        return u8

    @classmethod
    def _unpack_entry(cls, data: bytes, offset: int) -> tuple[bool, U8.FileEntry | U8.DirectoryEntry]:
        is_directory, name_high, name_low, field1, field2 = struct.unpack_from(">BBHII", data, offset)
        name_offset = (name_high << 16) | name_low
        entry_type = cls.DirectoryEntry if is_directory else cls.FileEntry
        return bool(is_directory), entry_type(name_offset, field1, field2)

    def unpack(self, data: bytes) -> None:
        try:
            self._unpack(data)
        except (struct.error, IndexError, UnicodeDecodeError) as error:
            raise FormatError(f"The U8 archive is damaged: {error}") from None

    def _unpack(self, data: bytes) -> None:
        if len(data) < 0x20 or data[:4] != U8_MAGIC:
            raise FormatError("Not a Wii U8 archive. A .szs, .LZ or .LH file has to be decompressed first.")
        self.header = self.Header.unpack_from(data)
        table_offset = self.header.file_system_table_offset

        _, root = self._unpack_entry(data, table_offset)
        if not isinstance(root, self.DirectoryEntry) or \
                not 1 <= root.skip_index <= (len(data) - table_offset) // self.ENTRY_SIZE:
            raise FormatError("Invalid U8 file table")
        file_system_table: list[U8.FileEntry | U8.DirectoryEntry] = [root]
        for i in range(1, root.skip_index):
            _, entry = self._unpack_entry(data, table_offset + i * self.ENTRY_SIZE)
            file_system_table.append(entry)

        names_offset = table_offset + root.skip_index * self.ENTRY_SIZE

        def read_name(name_offset: int) -> str:
            start = names_offset + name_offset
            return data[start:data.index(b"\0", start)].decode("ascii")

        class DirectoryStackEntry(NamedTuple):
            end_index: int
            name: str

        directory_stack: deque[DirectoryStackEntry] = deque([
            DirectoryStackEntry(root.skip_index - 1,
                                read_name(root.name_offset))
        ])

        self.files = {}

        for i, entry in enumerate(file_system_table):
            if i == 0:
                continue  # we already handled the root
            if not directory_stack:
                raise FormatError("Invalid U8 directory nesting")

            current_dir_name = directory_stack[-1].name

            if isinstance(entry, self.DirectoryEntry):
                directory_stack.append(DirectoryStackEntry(
                    entry.skip_index - 1,
                    f"{current_dir_name}/{read_name(entry.name_offset)}"
                ))

            else:
                name = f"{current_dir_name}/{read_name(entry.name_offset)}"
                end = entry.data_offset + entry.data_length
                if end > len(data):
                    raise FormatError(f"{name} ends past the end of the archive.")
                self.files[name] = data[entry.data_offset:end]

            # The innermost open directory may be the one just opened: an empty directory ends at its own entry.
            while directory_stack and i == directory_stack[-1].end_index:
                directory_stack.pop()

        if directory_stack:
            raise FormatError("Invalid U8 directory nesting")
