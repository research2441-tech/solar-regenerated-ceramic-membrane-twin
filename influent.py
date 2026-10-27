"""Stochastic greywater influent for a cluster of households.

Composition statistics follow published greywater surveys (bathroom-only
"light" greywater and mixed greywater with laundry). Concentrations are
log-normal with AR(1) correlation, a diurnal flow pattern with morning and
evening peaks, and discrete laundry events that raise surfactant,
microplastic and particle loads. Values are AFTER a 100 um screen and a
short settling stage.
"""
import numpy as np

# median and log-sd of post-screening concentrations
TYPES = {
    "light": {  # shower + hand basin
        "tss": (55.0, 0.45), "cod": (230.0, 0.40), "fdis": 0.55, "las": (7.0, 0.5),
        "ecoli": (4.3, 0.6),  # log10 CFU/100 mL (normal, mean, sd)
        "cond": (520.0, 0.20), "mp": (180.0, 0.6),  # particles L-1
        "tcs": (1.2, 0.6), "mep": (6.0, 0.6), "bp3": (2.5, 0.7),  # ug L-1
        "laundry_per_day": 0.0, "turb_per_tss": 0.95,
    },
    "mixed": {  # bathroom + laundry
        "tss": (85.0, 0.50), "cod": (330.0, 0.45), "fdis": 0.50, "las": (15.0, 0.55),
        "ecoli": (4.8, 0.7),
        "cond": (720.0, 0.25), "mp": (900.0, 0.8),
        "tcs": (1.8, 0.6), "mep": (4.5, 0.6), "bp3": (1.8, 0.7),
        "laundry_per_day": 1.6, "turb_per_tss": 1.0,
    },
}

DIURNAL = np.array([0.15, 0.10, 0.08, 0.08, 0.15, 0.55, 1.60, 2.30, 2.00, 1.40, 1.00, 0.90,
                    0.95, 0.85, 0.75, 0.80, 0.95, 1.20, 1.55, 1.85, 1.75, 1.30, 0.75, 0.35])
DIURNAL = DIURNAL / DIURNAL.mean()


def generate(n_hours, gtype, rng, households=10, lpcd=62.0, persons=2.6, shock=False, start_hour=0):
    p = TYPES[gtype]
    daily = households * persons * lpcd / 1000.0  # m3 d-1
    hod = (start_hour + np.arange(n_hours)) % 24
    flow = daily / 24.0 * DIURNAL[hod] * np.exp(rng.normal(0, 0.25, n_hours))  # m3 h-1
    # weekly modulation (weekend +12 %)
    dow = ((start_hour + np.arange(n_hours)) // 24) % 7
    flow *= np.where(dow >= 5, 1.12, 1.0)

    def ar_lognormal(med, sd, phi=0.85):
        e = np.zeros(n_hours); z = rng.normal(0, 1, n_hours)
        for t in range(1, n_hours):
            e[t] = phi * e[t - 1] + np.sqrt(1 - phi ** 2) * z[t]
        return med * np.exp(sd * e)

    out = {"flow": flow}
    for k in ("tss", "cod", "las", "cond", "mp", "tcs", "mep", "bp3"):
        out[k] = ar_lognormal(*p[k])
    mu, sd = p["ecoli"]
    out["ecoli_log"] = mu + sd * rng.normal(0, 1, n_hours) * 0.5 + 0.0
    # laundry events: elevated surfactant, microplastics, particles over 2 h
    lam = p["laundry_per_day"] * (1.8 if shock else 1.0)
    if lam > 0:
        n_ev = rng.poisson(lam * n_hours / 24.0)
        for _ in range(n_ev):
            t0 = rng.integers(0, n_hours)
            if not (8 <= (start_hour + t0) % 24 <= 21):
                continue
            sl = slice(t0, min(n_hours, t0 + 2))
            f = rng.uniform(1.6, 3.2)
            out["las"][sl] *= f; out["mp"][sl] *= f * 1.5; out["tss"][sl] *= 1.3
            out["flow"][sl] += rng.uniform(0.06, 0.10)
    if shock:
        # rare organic shock episodes (product overdosing, hair dye, cleaning)
        n_sh = rng.poisson(n_hours / (24 * 9.0))
        for _ in range(n_sh):
            t0 = rng.integers(0, n_hours); sl = slice(t0, min(n_hours, t0 + rng.integers(3, 9)))
            out["cod"][sl] *= rng.uniform(1.8, 3.0); out["las"][sl] *= rng.uniform(1.5, 2.5)
            out["tss"][sl] *= rng.uniform(1.5, 2.5)
    out["doc"] = out["cod"] * p["fdis"] / 2.9   # mg C L-1 (COD/DOC ~ 2.9)
    out["cod_dis"] = out["cod"] * p["fdis"]
    out["turb"] = out["tss"] * p["turb_per_tss"]
    return out


def water_temperature(tamb, rng):
    """Greywater temperature from the 24-h running ambient mean plus warm
    shower water; bounded to the 14-34 C range reported for greywater."""
    k = np.ones(24) / 24.0
    tm = np.convolve(np.r_[np.full(23, tamb[0]), tamb], k, mode="valid")
    tw = 24.0 + 0.30 * (tm - 18.0) + rng.normal(0, 0.6, len(tamb))
    return np.clip(tw, 14.0, 34.0)


def reuse_demand(n_hours, rng, households=10, persons=2.6, start_hour=0):
    """Toilet-flushing plus garden demand for the treated water [m3 h-1]."""
    hod = (start_hour + np.arange(n_hours)) % 24
    flush = households * persons * 0.030 / 24.0 * DIURNAL[hod]  # ~30 L p-1 d-1
    garden = np.where((hod >= 6) & (hod <= 8) | (hod >= 18) & (hod <= 19), 0.045, 0.0)
    return np.clip(flush * np.exp(rng.normal(0, 0.2, n_hours)) + garden, 0, None)
