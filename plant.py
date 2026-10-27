"""Ground-truth plant: solar-regenerated photocatalytic RHA ceramic membrane
module treating screened greywater (constant-flux dead-end filtration with
periodic backwash). Deliberately richer than the digital twin: compressible
cake, cake hardening, temperature-dependent organic fouling, scaling that
shades the photocatalyst, coating attrition under chemical cleaning,
photocorrosion, feed-turbidity light attenuation, sensor noise and drift.

Resistances are expressed in units of 1e12 m-1; TMP in kPa; flux in L m-2 h-1.
"""
import numpy as np
from membrane_design import viscosity, select_design, intrinsic_resistance, sieving_lrv

DESIGN = select_design()
FLUX_LEVELS = {1: 35.0, 2: 50.0, 3: 65.0}

DEFAULT = dict(
    area=2.5, feed_tank=1.2, perm_tank=0.8,
    alpha_c=0.28, comp_s=0.30, k_hard=0.020, eta_bw=0.82, bw_every=0.5, bw_vol=1.2,
    alpha_o=0.035, J_crit=45.0, Ro_max=12.0, gamma_T=0.045,
    alpha_s=6.0e-5, cond0=400.0,
    k_ph=0.6, K_R=2.0, beta=0.7, I_ref=400.0, k_ph_hard=0.35,
    tau_h=0.85, eps_t=0.06, d_ch=0.6, kappa_c=1.5, kappa_s=0.4, eps_p=0.15,
    q_cap=110.0, k_theta=0.25,
    delta_a=0.030, k_corr=2.0e-5, delta_sf=0.0015,
    tmp_max=200.0,
    ancillary_W=12.0, eis_W=2.0, recirc_W=25.0, pump_eff=0.5,
    cip_hours=2, cip_naocl_g=12.5, cip_citric_g=250.0,
)

PPCP = {  # base rejection, adsorptive removal (coated, theta=0), exponent, photo rate
    "tcs": (0.15, 0.85, 0.7, 0.9),
    "mep": (0.05, 0.70, 1.2, 0.6),
    "bp3": (0.10, 0.75, 1.0, 0.7),
}
ECOLI_D = 0.65   # effective rigid diameter [um]
MP_D = 2.0


class Plant:
    def __init__(self, inf, ghi, tw, demand, rng, coated=True, params=None, sensor=None):
        self.p = dict(DEFAULT); self.p.update(params or {})
        self.inf, self.ghi, self.tw, self.demand = inf, ghi, tw, demand
        self.rng = rng; self.coated = coated
        self.Rm = intrinsic_resistance(DESIGN["w_b"], DESIGN["Ts"], coated=coated) / 1e12
        self.Rc = 0.0; self.Rh = 0.0; self.Ro = 0.0; self.Rs = 0.0; self.theta = 0.0
        self.a = 1.0 if coated else 0.0
        self.sf = DESIGN["sf"]
        self.Vf = 0.5 * self.p["feed_tank"]; self.Vp = 0.5 * self.p["perm_tank"]
        self.t_since_bw = 0.0; self.cip_left = 0
        self.lrv_ec0 = sieving_lrv(ECOLI_D); self.lrv_mp0 = sieving_lrv(MP_D)
        self.sensor = dict(tmp_sd=0.01, tmp_off=0.3, uv_sd=0.02, eis_sd=0.03, drift=0.0, miss=0.0)
        self.sensor.update(sensor or {})
        self.h = 0
        self.dose_kwh = 0.0
        self.n_cip = 0

    # ------------------------------------------------------------------
    def _light(self, I, mode, turb):
        p = self.p
        I_s = I * p["tau_h"]
        if mode == "R":
            att = np.exp(-p["eps_p"] * p["d_ch"])
        elif mode == "F":
            att = np.exp(-p["eps_t"] * turb * p["d_ch"])
        else:
            att = np.exp(-p["eps_t"] * 0.5 * turb * p["d_ch"])
        return I_s * att * np.exp(-p["kappa_c"] * self.Rc)

    def _photo_rate(self, Ieff):
        p = self.p
        a_eff = self.a * np.exp(-p["kappa_s"] * self.Rs)
        return a_eff * (max(Ieff, 0.0) / p["I_ref"]) ** p["beta"]

    def tmp(self, J, T):
        Rt = self.Rm + self.Rc * (1.0 + 0.0) + self.Rh + self.Ro + self.Rs
        mu = viscosity(T)
        tmp = mu * J / 3.6e6 * Rt * 1e12 / 1e3
        # cake compressibility (plant only)
        if tmp > 1e-6:
            fc = (max(tmp, 5.0) / 50.0) ** self.p["comp_s"]
            tmp = mu * J / 3.6e6 * (self.Rm + self.Rc * fc + self.Rh + self.Ro + self.Rs) * 1e12 / 1e3
        return tmp

    # ------------------------------------------------------------------
    def step(self, mode, level=2):
        """Advance one hour. mode in F (filter), R (solar regeneration),
        C (chemical clean, 2 h), I (idle). Returns (obs, rec)."""
        p = self.p; h = self.h; inf = self.inf
        if self.cip_left > 0:
            mode = "C"
        I = self.ghi[h]; T = self.tw[h]
        tss, doc, turb, cond = inf["tss"][h], inf["doc"][h], inf["turb"][h], inf["cond"][h]
        Qin = inf["flow"][h]
        # feed tank receives greywater; overflow to sewer
        self.Vf += Qin
        overflow = max(0.0, self.Vf - p["feed_tank"]); self.Vf -= overflow
        J_set = FLUX_LEVELS.get(level, 50.0) if mode == "F" else 0.0
        dt = 1.0 / 6.0
        Vprod = 0.0; Ehyd = 0.0; tmp_acc = 0.0; n_f = 0; Jsum = 0.0
        naocl = 0.0; citric = 0.0
        if mode == "C" and self.cip_left == 0:
            self.cip_left = p["cip_hours"]; self.n_cip += 1
            naocl, citric = p["cip_naocl_g"], p["cip_citric_g"]
        if mode == "R":
            # regeneration starts with a backwash, then the channel is refilled with permeate
            self._backwash(); Vprod -= p["bw_vol"] * p["area"] / 1000.0
        for _ in range(6):
            if mode == "F" and self.Vf > 0.02:
                J = J_set
                tmpv = self.tmp(J, T)
                if tmpv > p["tmp_max"]:
                    J = J * p["tmp_max"] / tmpv; tmpv = p["tmp_max"]
                q = J * p["area"] / 1000.0 * dt
                q = min(q, self.Vf - 0.02); J = q / (p["area"] / 1000.0 * dt)
                self.Vf -= q; Vprod += q
                Ehyd += tmpv * 1e3 * q / p["pump_eff"] / 3.6e6  # kWh
                tmp_acc += tmpv; n_f += 1; Jsum += J
                mflux = tss * J / 1000.0 * dt
                fc = (max(tmpv, 5.0) / 50.0) ** p["comp_s"]
                self.Rc += p["alpha_c"] * mflux * fc
                dRo = p["alpha_o"] * doc * J / 1000.0 * (J / p["J_crit"]) ** 0.6 * dt * np.exp(p["gamma_T"] * (T - 20.0)) * (1 - self.Ro / p["Ro_max"])
                self.Ro += max(dRo, 0.0) * 1.0
                self.Rs += p["alpha_s"] * J * max(0.0, cond - p["cond0"]) / 1000.0 * dt * np.exp(0.03 * (T - 20.0))
                self.theta += (J * doc / 1000.0 * dt / p["q_cap"]) * (1 - self.theta) * (1.0 if self.coated else 4.0)
                self.t_since_bw += dt
                if self.t_since_bw >= p["bw_every"] - 1e-9:
                    self._backwash(); Vprod -= p["bw_vol"] * p["area"] / 1000.0
            # cake hardening (always)
            hard = p["k_hard"] * self.Rc * dt
            self.Rc -= hard; self.Rh += hard
            if mode == "C":
                pass
            elif self.coated:
                Ieff = self._light(I, mode if mode in ("F", "R") else "I", turb)
                g = self._photo_rate(Ieff)
                self.Ro -= p["k_ph"] * g * self.Ro / (p["K_R"] + self.Ro) * dt
                self.Rh -= p["k_ph_hard"] * g * self.Rh / (p["K_R"] + self.Rh) * dt
                self.theta -= p["k_theta"] * g * self.theta * dt
                self.dose_kwh += Ieff / 1000.0 * dt
                self.a -= p["k_corr"] * Ieff / 1000.0 * dt * self.a
            self.Ro = max(self.Ro, 0.0); self.Rh = max(self.Rh, 0.0); self.theta = min(max(self.theta, 0.0), 1.0)
        if mode == "C":
            self.cip_left -= 1
            if self.cip_left == 0:
                self.Rc *= 0.05; self.Rh *= 0.15; self.Ro *= 0.10; self.Rs *= 0.35; self.theta *= 0.30
                self.a *= (1 - p["delta_a"]); self.sf *= (1 - p["delta_sf"])
        Vprod = max(Vprod, 0.0)
        # permeate tank and reuse demand
        quality = self._quality(mode, I, turb, h)
        divert = 0.0
        if Vprod > 0 and not quality["basic_ok"]:
            divert = Vprod; Vprod = 0.0
        self.Vp += Vprod
        spill = max(0.0, self.Vp - p["perm_tank"]); self.Vp -= spill
        d = self.demand[h]; served = min(d, self.Vp); self.Vp -= served
        Eanc = (p["ancillary_W"] + (p["recirc_W"] if mode == "R" else 0.0)) / 1000.0
        tmp_mean = tmp_acc / n_f if n_f else self.tmp(50.0, T) * 0.0
        J_mean = Jsum / n_f if n_f else 0.0
        obs = self._observe(mode, J_mean, tmp_mean, T, I, quality, h)
        rec = dict(mode=mode, level=level, J=J_mean, tmp=tmp_mean, Vprod=Vprod, divert=divert, overflow=overflow,
                   spill=spill, demand=d, served=served, Ehyd=Ehyd, Eanc=Eanc, naocl=naocl, citric=citric,
                   Rc=self.Rc, Rh=self.Rh, Ro=self.Ro, Rs=self.Rs, theta=self.theta, a=self.a, sf=self.sf,
                   Vf=self.Vf, Vp=self.Vp, I=I, T=T, **quality)
        self.h += 1
        return obs, rec

    def _backwash(self):
        p = self.p
        eff = p["eta_bw"] * np.exp(-0.05 * self.Rh)
        self.Rc *= (1 - eff); self.Ro *= 0.97; self.t_since_bw = 0.0

    def _quality(self, mode, I, turb, h):
        p = self.p; inf = self.inf
        cake = self.Rc / (1.0 + self.Rc) + 0.5 * self.Rh / (1 + self.Rh)
        lrv_ec = self.lrv_ec0 + 1.2 * cake
        ec_p = inf["ecoli_log"][h] - lrv_ec
        turb_p = 0.08 + turb * 10 ** (-(2.4 + 0.6 * cake))
        lrv_mp = min(self.lrv_mp0 + cake, 6.5)
        r0, ra = (0.35, 0.40) if self.coated else (0.35, 0.15)
        cod_p = inf["cod"][h] * (1 - inf["cod_dis"][h] / inf["cod"][h]) * 0.03 + \
            inf["cod_dis"][h] * (1 - r0 - ra * (1 - self.theta))
        Ieff_f = self._light(I, "F", turb) if (self.coated and mode == "F") else 0.0
        g = self._photo_rate(Ieff_f) if self.coated else 0.0
        rem = {}; num = 0.0; den = 0.0
        for k, (b0, ba, n, kp) in PPCP.items():
            ba_eff = ba if self.coated else 0.35 * ba
            phot = 1 - np.exp(-kp * g * 0.15)
            r = 1 - (1 - b0) * (1 - ba_eff * (1 - self.theta) ** n) * (1 - phot)
            rem[k] = r; num += r * inf[k][h]; den += inf[k][h]
        ppcp = num / den
        las_p = inf["las"][h] * (1 - 0.25 - (0.55 if self.coated else 0.2) * (1 - self.theta))
        basic_ok = (turb_p <= 2.0) and (ec_p <= np.log10(2.2))
        org_ok = (cod_p <= 70.0) and (ppcp >= 0.60)
        return dict(turb_p=turb_p, ec_p=ec_p, lrv_ec=lrv_ec, lrv_mp=lrv_mp, cod_p=cod_p, ppcp=ppcp,
                    las_p=las_p, rem_tcs=rem["tcs"], rem_mep=rem["mep"], rem_bp3=rem["bp3"],
                    basic_ok=basic_ok, org_ok=org_ok)

    def eis_features(self):
        r_mem = 1.0 + 0.30 * self.Ro + 0.50 * self.Rs + 0.10 * self.Rh
        r_cake = 0.80 * self.Rc ** 0.9 + 0.60 * self.Rh
        y1 = (r_mem - 1.0) + 0.15 * r_cake
        y2 = r_cake + 0.10 * (r_mem - 1.0)
        return y1, y2

    def _observe(self, mode, J, tmp, T, I, q, h):
        s = self.sensor; rng = self.rng; inf = self.inf
        miss = rng.random() < s["miss"]
        drift = 1.0 + s["drift"] * h / 1000.0
        y1, y2 = self.eis_features()
        o = dict(mode=mode, J=J, T=T + rng.normal(0, 0.2), I=I * (1 + rng.normal(0, 0.03)),
                 tmp=(tmp * drift * (1 + rng.normal(0, s["tmp_sd"])) + rng.normal(0, s["tmp_off"])) if J > 0 else np.nan,
                 turb_f=inf["turb"][h] * (1 + rng.normal(0, 0.05)), turb_p=q["turb_p"] * (1 + rng.normal(0, 0.05)),
                 uv_f=inf["doc"][h] * 0.020 * (1 + rng.normal(0, s["uv_sd"])),
                 uv_p=(q["cod_p"] / 2.9) * 0.012 * (1 + rng.normal(0, s["uv_sd"])) if J > 0 else np.nan,
                 cond=inf["cond"][h] * (1 + rng.normal(0, 0.02)), tss_f=inf["tss"][h] * (1 + rng.normal(0, 0.08)),
                 doc_f=inf["doc"][h] * (1 + rng.normal(0, 0.05)),
                 eis1=y1 * (1 + rng.normal(0, s["eis_sd"])) + rng.normal(0, 0.02),
                 eis2=y2 * (1 + rng.normal(0, s["eis_sd"])) + rng.normal(0, 0.02),
                 Vf=self.Vf, Vp=self.Vp, cip_left=self.cip_left, Qin=inf["flow"][h])
        if miss:
            for k in ("tmp", "uv_p", "eis1", "eis2"):
                o[k] = np.nan
        return o
