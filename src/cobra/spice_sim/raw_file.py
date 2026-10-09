"""Reader for SPICE raw files, the result format VACASK writes (``<analysis>.raw``).

A raw file holds one or more plots, each a text header followed by its values,
either as text (``Values:``) or as little-endian doubles (``Binary:``), point
by point.  Complex plots store a real and an imaginary double per value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RawPlot:
    """One plot of a raw file: its name and one array per variable."""

    name: str
    variables: dict[str, np.ndarray] = field(default_factory=dict)


def read_raw(path: str | Path) -> list[RawPlot]:
    """Read every plot in the raw file at *path*.

    Raises :class:`ValueError` when the file is not a raw file it can read.
    """
    data = Path(path).read_bytes()
    plots: list[RawPlot] = []
    position = 0
    while position < len(data) and data[position:].strip():
        plot, position = _read_plot(data, position)
        plots.append(plot)
    if not plots:
        raise ValueError(f"No plot in raw file {path}")
    return plots


def _read_plot(data: bytes, position: int) -> tuple[RawPlot, int]:
    header: dict[str, str] = {}
    names: list[str] = []
    while True:
        end = data.find(b"\n", position)
        if end < 0:
            raise ValueError("Raw file header ends without a Values: or Binary: line")
        line = data[position:end].decode("utf-8", errors="replace").rstrip("\r")
        position = end + 1
        if line.startswith("\t") and "Variables" in header:
            names.append(line.split("\t")[2])
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if key in ("Values", "Binary"):
            break
        header[key] = value.strip()

    try:
        count = int(header["No. Variables"])
        points = int(header["No. Points"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Raw file header lacks the variable or point count: {exc}") from exc
    if len(names) != count:
        raise ValueError(f"Raw file lists {len(names)} variables but declares {count}")
    is_complex = "complex" in header.get("Flags", "").lower()
    width = 2 if is_complex else 1

    if key == "Binary":
        size = points * count * width * 8
        table = np.frombuffer(data, dtype="<f8", count=points * count * width, offset=position)
        position += size
    else:
        table, position = _read_values(data, position, points, count, width)
    table = table.reshape(points, count, width)
    values = table[..., 0] + 1j * table[..., 1] if is_complex else table[..., 0]
    variables = {name: np.array(values[:, index]) for index, name in enumerate(names)}
    return RawPlot(header.get("Plotname", ""), variables), position


def _read_values(
    data: bytes, position: int, points: int, count: int, width: int
) -> tuple[np.ndarray, int]:
    """Parse the text block of a plot: per point an index, then one line per variable."""
    numbers: list[float] = []
    wanted = points * count * width
    while len(numbers) < wanted:
        end = data.find(b"\n", position)
        end = len(data) if end < 0 else end
        fields = data[position:end].decode("ascii").split()
        if not fields and end >= len(data):
            raise ValueError("Raw file ends before all values were read")
        position = end + 1
        if len(numbers) % (count * width) == 0 and fields:
            fields = fields[1:]  # the point index
        for item in fields:
            numbers.extend(float(part) for part in item.split(","))
    return np.array(numbers[:wanted]), position
