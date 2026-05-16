"""
LSF binary file reader.

Reads DUNE LSF (LSTS Serialized Format) log files — raw concatenations
of IMC packets.  Supports both plain .lsf and gzip-compressed .lsf.gz.

Each packet:
    [20-byte header][payload (hdr.size bytes)][2-byte CRC-16-IBM footer]

Header layout (little-endian when sync == 0xFE54):
    Offset  Size  Field
    0       2     sync        (0xFE54 LE / 0x54FE BE)
    2       2     mgid        message id
    4       2     size        payload length
    6       8     timestamp   fp64, Unix epoch seconds
    14      2     src         source IMC address
    16      1     src_ent     source entity
    17      2     dst         destination IMC address
    19      1     dst_ent     destination entity
"""

import gzip
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

from imc_parser import FIXED_TYPE_SIZES, FieldDef, MessageDef

SYNC_LE = 0xFE54
SYNC_BE = 0x54FE
HEADER_SIZE = 20
FOOTER_SIZE = 2

STRUCT_FMT_LE = "<HHHdHBHB"
STRUCT_FMT_BE = ">HHHdHBHB"

_CHUNK_SIZE = 1 * 1024 * 1024  # 1 MiB read chunks for gzip streaming


@dataclass
class PacketHeader:
    sync: int
    mgid: int
    size: int
    timestamp: float
    src: int
    src_ent: int
    dst: int
    dst_ent: int


@dataclass
class IMCMessage:
    header: PacketHeader
    fields: dict          # abbrev -> decoded value
    abbrev: str           # message type abbreviation
    name: str             # human-readable message name
    raw_payload: bytes    # raw payload bytes (kept for debugging)


_UNPACK_LE = {
    "int8_t":   lambda buf, off: (struct.unpack_from("<b", buf, off)[0], off + 1),
    "uint8_t":  lambda buf, off: (struct.unpack_from("<B", buf, off)[0], off + 1),
    "int16_t":  lambda buf, off: (struct.unpack_from("<h", buf, off)[0], off + 2),
    "uint16_t": lambda buf, off: (struct.unpack_from("<H", buf, off)[0], off + 2),
    "int32_t":  lambda buf, off: (struct.unpack_from("<i", buf, off)[0], off + 4),
    "uint32_t": lambda buf, off: (struct.unpack_from("<I", buf, off)[0], off + 4),
    "int64_t":  lambda buf, off: (struct.unpack_from("<q", buf, off)[0], off + 8),
    "fp32_t":   lambda buf, off: (struct.unpack_from("<f", buf, off)[0], off + 4),
    "fp64_t":   lambda buf, off: (struct.unpack_from("<d", buf, off)[0], off + 8),
}

_UNPACK_BE = {
    "int8_t":   lambda buf, off: (struct.unpack_from(">b", buf, off)[0], off + 1),
    "uint8_t":  lambda buf, off: (struct.unpack_from(">B", buf, off)[0], off + 1),
    "int16_t":  lambda buf, off: (struct.unpack_from(">h", buf, off)[0], off + 2),
    "uint16_t": lambda buf, off: (struct.unpack_from(">H", buf, off)[0], off + 2),
    "int32_t":  lambda buf, off: (struct.unpack_from(">i", buf, off)[0], off + 4),
    "uint32_t": lambda buf, off: (struct.unpack_from(">I", buf, off)[0], off + 4),
    "int64_t":  lambda buf, off: (struct.unpack_from(">q", buf, off)[0], off + 8),
    "fp32_t":   lambda buf, off: (struct.unpack_from(">f", buf, off)[0], off + 4),
    "fp64_t":   lambda buf, off: (struct.unpack_from(">d", buf, off)[0], off + 8),
}


def _read_plaintext(buf: bytes, offset: int, unpack: dict) -> tuple:
    length, offset = unpack["uint16_t"](buf, offset)
    text = buf[offset:offset + length].decode("utf-8", errors="replace")
    return text, offset + length


def _read_rawdata(buf: bytes, offset: int, unpack: dict) -> tuple:
    length, offset = unpack["uint16_t"](buf, offset)
    data = buf[offset:offset + length]
    return data.hex(), offset + length


def _read_inline_message(buf: bytes, offset: int, unpack: dict,
                         msg_defs: dict) -> tuple:
    """Read an inline IMC message field (uint16 id prefix + serialized payload)."""
    msg_id, offset = unpack["uint16_t"](buf, offset)
    if msg_id == 0xFFFF:
        return None, offset
    if msg_id not in msg_defs:
        return f"<unknown_msg_id={msg_id}>", offset
    inner_def = msg_defs[msg_id]
    inner_fields, offset = _deserialize_fields(buf, offset, inner_def.fields,
                                                unpack, msg_defs)
    return {inner_def.abbrev: inner_fields}, offset


def _read_message_list(buf: bytes, offset: int, unpack: dict,
                       msg_defs: dict) -> tuple:
    """Read a message-list field (uint16 count prefix + N inline messages)."""
    count, offset = unpack["uint16_t"](buf, offset)
    messages = []
    for _ in range(count):
        msg, offset = _read_inline_message(buf, offset, unpack, msg_defs)
        messages.append(msg)
    return messages, offset


def _deserialize_fields(buf: bytes, offset: int, field_defs: list[FieldDef],
                        unpack: dict, msg_defs: dict) -> tuple[dict, int]:
    """Deserialize payload fields according to their definitions."""
    result = {}
    for fdef in field_defs:
        if offset > len(buf):
            break

        ftype = fdef.type
        try:
            if ftype in unpack:
                value, offset = unpack[ftype](buf, offset)
            elif ftype == "plaintext":
                value, offset = _read_plaintext(buf, offset, unpack)
            elif ftype == "rawdata":
                value, offset = _read_rawdata(buf, offset, unpack)
            elif ftype == "message":
                value, offset = _read_inline_message(buf, offset, unpack, msg_defs)
            elif ftype == "message-list":
                value, offset = _read_message_list(buf, offset, unpack, msg_defs)
            else:
                value = f"<unsupported_type:{ftype}>"
                break
        except (struct.error, IndexError):
            value = "<decode_error>"
            break

        result[fdef.abbrev] = value

    return result, offset


def _consume_packets(buf: bytes, filter_ids, msg_defs, keep_raw):
    """Extract all complete packets from a buffer, return (packets, remainder)."""
    packets = []
    offset = 0
    length = len(buf)

    while offset + HEADER_SIZE + FOOTER_SIZE <= length:
        raw_sync = struct.unpack_from("<H", buf, offset)[0]
        if raw_sync == SYNC_LE:
            hdr_fmt = STRUCT_FMT_LE
            unpack = _UNPACK_LE
        elif raw_sync == SYNC_BE:
            hdr_fmt = STRUCT_FMT_BE
            unpack = _UNPACK_BE
        else:
            offset += 1
            continue

        vals = struct.unpack_from(hdr_fmt, buf, offset)
        hdr = PacketHeader(*vals)

        packet_end = offset + HEADER_SIZE + hdr.size + FOOTER_SIZE
        if packet_end > length:
            break

        payload_bytes = buf[offset + HEADER_SIZE:offset + HEADER_SIZE + hdr.size]
        offset = packet_end

        if filter_ids is not None and hdr.mgid not in filter_ids:
            continue

        mdef = msg_defs.get(hdr.mgid)
        if mdef is None:
            continue

        fields, _ = _deserialize_fields(payload_bytes, 0, mdef.fields, unpack, msg_defs)
        packets.append(IMCMessage(
            header=hdr, fields=fields, abbrev=mdef.abbrev,
            name=mdef.name, raw_payload=payload_bytes if keep_raw else b"",
        ))

    return packets, buf[offset:]


def _open_lsf(path: Path) -> BinaryIO:
    """Open an LSF file, auto-detecting gzip compression."""
    if path.suffix == ".gz" or path.suffixes[-2:] == [".lsf", ".gz"]:
        return gzip.open(path, "rb")
    return open(path, "rb")


def iter_packets(path: Path, msg_defs: dict[int, MessageDef],
                 filter_abbrevs: Optional[set[str]] = None,
                 keep_raw: bool = False,
                 ) -> Iterator[IMCMessage]:
    """
    Iterate over all IMC packets in an LSF file.

    Parameters
    ----------
    path : Path to .lsf or .lsf.gz file
    msg_defs : message definitions from parse_imc_xml()
    filter_abbrevs : if set, only yield messages whose abbrev is in this set
    keep_raw : if True, store raw payload bytes in IMCMessage (default: False)
    """
    filter_ids = None
    if filter_abbrevs:
        filter_ids = {mid for mid, mdef in msg_defs.items()
                      if mdef.abbrev in filter_abbrevs}

    def _process_buf(buf: bytes):
        offset = 0
        length = len(buf)
        while offset + HEADER_SIZE + FOOTER_SIZE <= length:
            if offset + 2 > length:
                break

            raw_sync = struct.unpack_from("<H", buf, offset)[0]
            if raw_sync == SYNC_LE:
                hdr_fmt = STRUCT_FMT_LE
                unpack = _UNPACK_LE
            elif raw_sync == SYNC_BE:
                hdr_fmt = STRUCT_FMT_BE
                unpack = _UNPACK_BE
            else:
                offset += 1
                continue

            if offset + HEADER_SIZE > length:
                break

            vals = struct.unpack_from(hdr_fmt, buf, offset)
            hdr = PacketHeader(*vals)

            packet_end = offset + HEADER_SIZE + hdr.size + FOOTER_SIZE
            if packet_end > length:
                break

            payload_bytes = buf[offset + HEADER_SIZE:offset + HEADER_SIZE + hdr.size]
            offset = packet_end

            if filter_ids is not None and hdr.mgid not in filter_ids:
                continue

            mdef = msg_defs.get(hdr.mgid)
            if mdef is None:
                continue

            fields, _ = _deserialize_fields(payload_bytes, 0, mdef.fields,
                                             unpack, msg_defs)

            yield IMCMessage(
                header=hdr,
                fields=fields,
                abbrev=mdef.abbrev,
                name=mdef.name,
                raw_payload=payload_bytes if keep_raw else b"",
            )

    def _stream_from_gzip(f: BinaryIO):
        """Stream-process a gzipped LSF file in chunks, yielding complete packets."""
        buf = b""
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            buf += chunk
            # Extract all complete packets from the buffer
            consumed = _consume_packets(buf, filter_ids, msg_defs, keep_raw)
            for pkt in consumed[0]:
                yield pkt
            buf = consumed[1]

        # Final pass on remaining data
        for pkt in _consume_packets(buf, filter_ids, msg_defs, keep_raw)[0]:
            yield pkt

    with _open_lsf(path) as f:
        # Plain files: mmap for zero-copy access
        if path.suffix != ".gz":
            try:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
                    yield from _process_buf(buf)
                return
            except (OSError, ValueError):
                pass
            f.seek(0)
            yield from _process_buf(f.read())
            return

        # Gzip files: streaming to keep memory bounded
        yield from _stream_from_gzip(f)
