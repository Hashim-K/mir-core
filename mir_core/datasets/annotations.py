"""
Read and write .beats annotation files, and convert from legacy formats.

.beats format (tab-separated, right-optional columns):
    time                    — beats only
    time\\tbeat_position     — beats with metrical position (1 = downbeat)
    time\\tbeat_position\\tbar — full annotation with bar number

All times are in seconds.

Converters:
    from_candombe_csv     — Candombe CSV with bar.beat encoding
    from_beats_tsv        — .beats tab-separated (BRID, Candombe w/o bar)
    from_salsa_dataset    — Salsa Dataset millisecond timestamps
    from_salsaset_csv     — SalsaSet comma-separated CSV
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np


@dataclass
class BeatAnnotation:
    """Parsed .beat annotation.

    Attributes:
        times: Beat onset times in seconds, shape (N,).
        positions: Beat position within bar (1-indexed), shape (N,) or None.
        bars: Bar number (1-indexed), shape (N,) or None.
    """

    times: np.ndarray
    positions: Optional[np.ndarray] = None
    bars: Optional[np.ndarray] = None

    @property
    def beat_times(self) -> np.ndarray:
        return self.times

    @property
    def downbeat_times(self) -> Optional[np.ndarray]:
        if self.positions is None:
            return None
        return self.times[self.positions == 1]


# ------------------------------------------------------------------
# Reader / Writer
# ------------------------------------------------------------------

def read_beat(path: Union[str, Path]) -> BeatAnnotation:
    """Read a .beats annotation file."""
    path = Path(path)
    times, positions, bars = [], [], []
    has_positions = None
    has_bars = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            times.append(float(parts[0]))

            if len(parts) >= 2:
                if has_positions is None:
                    has_positions = True
                positions.append(int(parts[1]))
            else:
                if has_positions is None:
                    has_positions = False

            if len(parts) >= 3:
                if has_bars is None:
                    has_bars = True
                bars.append(int(parts[2]))
            else:
                if has_bars is None:
                    has_bars = False

    return BeatAnnotation(
        times=np.array(times, dtype=np.float64),
        positions=np.array(positions, dtype=np.int32) if has_positions else None,
        bars=np.array(bars, dtype=np.int32) if has_bars else None,
    )


def write_beat(path: Union[str, Path], ann: BeatAnnotation) -> None:
    """Write a BeatAnnotation to a .beats file."""
    path = Path(path)
    with open(path, "w") as f:
        for i, t in enumerate(ann.times):
            parts = [f"{t:.9f}".rstrip("0").rstrip(".")]
            if ann.positions is not None:
                parts.append(str(ann.positions[i]))
            if ann.bars is not None:
                parts.append(str(ann.bars[i]))
            f.write("\t".join(parts) + "\n")


# ------------------------------------------------------------------
# Converters
# ------------------------------------------------------------------

def from_candombe_csv(path: Union[str, Path]) -> BeatAnnotation:
    """Convert Candombe CSV (time,bar.beat) to BeatAnnotation."""
    times, positions, bars = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            times.append(float(parts[0]))
            bar_beat = parts[1]  # e.g. "1.1" -> bar=1, beat=1
            bar, beat = bar_beat.split(".")
            bars.append(int(bar))
            positions.append(int(beat))

    return BeatAnnotation(
        times=np.array(times, dtype=np.float64),
        positions=np.array(positions, dtype=np.int32),
        bars=np.array(bars, dtype=np.int32),
    )


def from_beats_tsv(path: Union[str, Path]) -> BeatAnnotation:
    """Convert .beats TSV (time\\tposition) to BeatAnnotation."""
    return read_beat(path)  # same format


def from_salsa_dataset(path: Union[str, Path]) -> BeatAnnotation:
    """Convert Salsa Dataset TXT (millisecond timestamps) to BeatAnnotation."""
    times = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            times.append(int(line) / 1000.0)

    return BeatAnnotation(times=np.array(times, dtype=np.float64))


def from_salsaset_csv(path: Union[str, Path]) -> BeatAnnotation:
    """Convert SalsaSet CSV (time,position) to BeatAnnotation."""
    times, positions = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            times.append(float(parts[0]))
            positions.append(int(parts[1]))

    return BeatAnnotation(
        times=np.array(times, dtype=np.float64),
        positions=np.array(positions, dtype=np.int32),
    )
