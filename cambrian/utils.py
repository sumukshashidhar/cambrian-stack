from __future__ import annotations

# -----------------------------------------------------------------------------
# Utility: Parse human-readable numbers (e.g., "10B", "100M")
# -----------------------------------------------------------------------------
def parse_number(value: str | int | None) -> int | None:
    """Parse human-readable number strings like '10B', '100M', '5K' to integers."""
    if value is None or isinstance(value, int): return value
    value = str(value).strip().upper()
    if not value: return None
    
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
    for suffix, mult in multipliers.items():
        if value.endswith(suffix): return int(float(value[:-1]) * mult)
    return int(float(value))

