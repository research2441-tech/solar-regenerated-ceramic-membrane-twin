"""Rule-based operating strategies used as baselines."""
import numpy as np
from plant import DEFAULT

CAP = DEFAULT["perm_tank"]


class LevelControl:
    """Permeate-tank hysteresis: start filtering below lo*CAP, stop above hi*CAP."""

    def __init__(self, lo=0.55, hi=0.95):
        self.lo, self.hi = lo, hi; self.on = True

    def want(self, Vp, Vf):
        if Vp < self.lo * CAP:
            self.on = True
        elif Vp > self.hi * CAP:
            self.on = False
        return self.on and Vf > 0.05


class Base:
    name = "base"
    coated = True

    def reset(self, ctx, plant):
        self.ctx = ctx; self.plant = plant; self.lc = LevelControl()
        self.last_tmp = 20.0; self.o = None; self.reg_today = -1; self.reg_left = 0

    def observe(self, h, o):
        self.o = o
        if o["J"] > 0 and np.isfinite(o["tmp"]):
            self.last_tmp = o["tmp"]

    def _vols(self):
        o = self.o
        if o is None:
            return self.plant.Vp, self.plant.Vf
        return o["Vp"], o["Vf"]


class Conventional(Base):
    """Uncoated RHA ceramic MF: level control, backwash, TMP-triggered CIP."""
    name = "CONV-MF"; coated = False

    def __init__(self, tmp_cip=120.0):
        self.tmp_cip = tmp_cip

    def act(self, h):
        if self.o is not None and self.o["cip_left"] > 0:
            return "C", 2
        if self.last_tmp > self.tmp_cip:
            self.last_tmp = 20.0
            return "C", 2
        Vp, Vf = self._vols()
        return ("F", 2) if self.lc.want(Vp, Vf) else ("I", 2)


class FixedSolar(Conventional):
    """Photocatalytic membrane with a fixed daily solar regeneration window."""
    name = "PC-FIX"; coated = True

    def __init__(self, tmp_cip=120.0, hours=(11, 12)):
        super().__init__(tmp_cip); self.hours = hours

    def act(self, h):
        m = super().act(h)
        if m[0] == "C":
            return m
        if (h % 24) in self.hours:
            return "R", 2
        return m


class ThresholdSolar(Conventional):
    """Event-triggered regeneration: when TMP or permeate UV254 ratio exceeds a
    threshold, regenerate for up to 3 h during the next sunlit hours."""
    name = "PC-TRIG"; coated = True

    def __init__(self, tmp_cip=120.0, tmp_reg=55.0, uv_ratio=0.40, ghi_min=250.0):
        super().__init__(tmp_cip); self.tmp_reg = tmp_reg; self.uvr = uv_ratio; self.gmin = ghi_min
        self.pending = False; self.left = 0

    def act(self, h):
        m = super().act(h)
        if m[0] == "C":
            return m
        o = self.o
        if o is not None:
            uvr = o["uv_p"] / max(o["uv_f"], 1e-6) if np.isfinite(o.get("uv_p", np.nan)) else 0.0
            if self.last_tmp > self.tmp_reg or uvr > self.uvr:
                self.pending = True
        if self.left > 0:
            self.left -= 1
            return "R", 2
        if self.pending and o is not None and o["I"] > self.gmin:
            self.pending = False; self.left = 2; self.last_tmp = 20.0
            return "R", 2
        return m
