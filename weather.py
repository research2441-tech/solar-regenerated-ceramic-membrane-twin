"""Real typical-meteorological-year (TMY3) weather: hourly global horizontal
irradiance and dry-bulb temperature for the sites used in the study.

Sources (public, unmodified):
  723170TYA.CSV  Greensboro Piedmont Triad Intl., NC (pvlib-python test data)
  USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3.epw (EnergyPlus weather set)
  USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw     (EnergyPlus weather set)
  703165TY.csv   Sand Point, AK (pvlib-python test data, low-irradiance stress)
"""
import os
import numpy as np

DATA = os.path.dirname(os.path.abspath(__file__))

SITES = {
    "PHX": ("USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMY3.epw", "epw"),
    "GSO": ("723170TYA.CSV", "tmy3"),
    "CHI": ("USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw", "epw"),
    "SDP": ("703165TY.csv", "tmy3"),
}


def _load_tmy3(path):
    ghi, tdb = [], []
    with open(path) as f:
        f.readline(); hdr = f.readline().strip().split(",")
        ig = hdr.index("GHI (W/m^2)"); it = hdr.index("Dry-bulb (C)")
        for line in f:
            p = line.strip().split(",")
            ghi.append(float(p[ig])); tdb.append(float(p[it]))
    return np.array(ghi), np.array(tdb)


def _load_epw(path):
    ghi, tdb = [], []
    with open(path) as f:
        for _ in range(8):
            f.readline()
        for line in f:
            p = line.strip().split(",")
            tdb.append(float(p[6])); ghi.append(float(p[13]))
    return np.array(ghi), np.array(tdb)


_cache = {}


def load_site(code):
    """Return dict with hourly GHI [W m-2] and ambient temperature [C], 8760 h."""
    if code in _cache:
        return _cache[code]
    fn, kind = SITES[code]
    path = os.path.join(DATA, fn)
    ghi, tdb = (_load_tmy3 if kind == "tmy3" else _load_epw)(path)
    ghi = np.clip(ghi[:8760], 0, None); tdb = tdb[:8760]
    _cache[code] = {"ghi": ghi, "tamb": tdb}
    return _cache[code]


def series(code, start_day, n_hours, scale=1.0):
    """Hourly weather slice with wrap-around over the TMY year."""
    w = load_site(code)
    idx = (start_day * 24 + np.arange(n_hours)) % 8760
    return w["ghi"][idx] * scale, w["tamb"][idx]


def day_ahead_forecast(ghi_hist, horizon, hour_now, rng, n_scen=1):
    """Irradiance forecast available to controllers: blend of the mean of the
    previous three days (same clock hour) and yesterday, with multiplicative
    scenario perturbations. Uses only past observations."""
    H = len(ghi_hist)
    out = np.zeros((n_scen, horizon))
    for k in range(horizon):
        hrs = [hour_now + k - 24 * d for d in (1, 2, 3) if hour_now + k - 24 * d >= 0 and hour_now + k - 24 * d < H]
        if hrs:
            m3 = np.mean([ghi_hist[h] for h in hrs])
            y1 = ghi_hist[hrs[0]]
            base = 0.5 * m3 + 0.5 * y1
        else:
            base = 0.0
        out[:, k] = base
    if n_scen > 1:
        mult = np.clip(rng.lognormal(0.0, 0.35, size=(n_scen, 1)), 0.2, 1.6)
        out = out * mult
    return out
