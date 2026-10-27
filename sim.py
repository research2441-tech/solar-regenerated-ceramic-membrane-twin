"""Closed-loop simulation harness and run-level metrics."""
import numpy as np
import influent, weather
from plant import Plant

SCEN = {"normal": dict(gtype="mixed", shock=False), "shock": dict(gtype="mixed", shock=True),
        "light": dict(gtype="light", shock=False)}

# screening-level economic and climate factors (manuscript Table, sensitivity in Sec. V)
FACT = dict(elec_cost=0.12, elec_gwp=0.70, naocl_cost=2.0e-3, naocl_gwp=2.6e-3,   # per g active chlorine
            citric_cost=1.5e-3, citric_gwp=5.5e-3, potable_cost=1.2, potable_gwp=0.40,   # per m3 makeup
            coat_cost=60.0, coat_gwp=25.0, memb_cost=180.0, memb_gwp=95.0)  # per module (2.5 m2)


def make_env(site, start_day, n_days, scen, seed, irr_scale=1.0, sensor=None, params=None, coated=True):
    rng = np.random.default_rng(seed)
    H = n_days * 24
    sc = SCEN[scen]
    inf = influent.generate(H, sc["gtype"], rng, shock=sc["shock"], start_hour=start_day * 24)
    ghi, ta = weather.series(site, start_day, H, irr_scale)
    tw = influent.water_temperature(ta, rng)
    dem = influent.reuse_demand(H, rng, start_hour=start_day * 24)
    plant = Plant(inf, ghi, tw, dem, np.random.default_rng(seed + 7919), coated=coated, params=params, sensor=sensor)
    return plant, dict(site=site, start_day=start_day, H=H, scen=scen, seed=seed, ghi=ghi, tw=tw, gtype=sc["gtype"])


def run(controller, site="GSO", start_day=0, n_days=60, scen="normal", seed=1, irr_scale=1.0,
        sensor=None, params=None, coated=None, log=True):
    if coated is None:
        coated = getattr(controller, "coated", True)
    plant, ctx = make_env(site, start_day, n_days, scen, seed, irr_scale, sensor, params, coated)
    controller.reset(ctx, plant)
    recs, obs_l = [], []
    for h in range(ctx["H"]):
        mode, level = controller.act(h)
        o, r = plant.step(mode, level)
        controller.observe(h, o)
        recs.append(r); obs_l.append(o)
    return recs, obs_l, plant, ctx


def summarize(recs, plant, ctx, extra_W=0.0):
    import pandas as pd
    df = pd.DataFrame(recs)
    days = ctx["H"] / 24.0
    prod = df.Vprod.sum(); dem = df.demand.sum(); served = df.served.sum()
    f = df[df.J > 0]
    w = f.Vprod + f.divert
    vol_ok_basic = (w * f.basic_ok).sum() / max(w.sum(), 1e-9)
    vol_ok_org = (f.Vprod * f.org_ok).sum() / max(f.Vprod.sum(), 1e-9)
    E = df.Ehyd.sum() + df.Eanc.sum() + extra_W / 1000.0 * ctx["H"]
    naocl = df.naocl.sum(); citric = df.citric.sum()
    makeup = dem - served
    # durability: coating renewal when activity < 0.6, module replacement when strength < 70 % of design
    da = max(1.0 - plant.a, 1e-6) if plant.coated else 0.0
    life_coat = (0.4 / da * days / 365.0) if plant.coated else np.inf
    dsf = max(1.0 - plant.sf / df.sf.iloc[0] * (1.0) if len(df) else 0.0, 1e-6)
    from plant import DESIGN
    dsf = max(1 - plant.sf / DESIGN["sf"], 1e-6)
    life_memb = min(0.30 / dsf * days / 365.0, 10.0)
    yr = 365.0 / days
    vol_yr = prod * yr
    coat_per_yr = (1.0 / min(life_coat, 10.0)) if plant.coated else 0.0
    cost = (E * yr * FACT["elec_cost"] + naocl * yr * FACT["naocl_cost"] + citric * yr * FACT["citric_cost"] +
            makeup * yr * FACT["potable_cost"] + coat_per_yr * FACT["coat_cost"] + FACT["memb_cost"] / life_memb) / max(vol_yr, 1e-9)
    gwp = (E * yr * FACT["elec_gwp"] + naocl * yr * FACT["naocl_gwp"] + citric * yr * FACT["citric_gwp"] +
           makeup * yr * FACT["potable_gwp"] + coat_per_yr * FACT["coat_gwp"] + FACT["memb_gwp"] / life_memb) / max(vol_yr, 1e-9)
    # permeability normalised to 20 C and to the clean membrane
    from membrane_design import viscosity
    Rm = plant.Rm
    if len(f):
        Rtot = f.tmp * 1e3 / (viscosity(f["T"]) * f.J / 3.6e6) / 1e12
        norm_perm = float(np.mean(Rm / Rtot.clip(lower=Rm)))
        p10 = float(np.percentile(Rm / Rtot.clip(lower=Rm), 10))
    else:
        norm_perm = p10 = np.nan
    # regeneration efficiency: resistance removed per regeneration hour relative to start
    reg = df[df["mode"] == "R"]
    n_reg = len(reg)
    return dict(demand_met=served / dem, makeup_m3=makeup, prod_m3_d=prod / days, recovery=prod / max(df.overflow.sum() + prod + df.divert.sum(), 1e-9),
                basic_ok=vol_ok_basic, org_ok=vol_ok_org, ppcp=float((f.ppcp * f.Vprod).sum() / max(f.Vprod.sum(), 1e-9)),
                cod_p=float((f.cod_p * f.Vprod).sum() / max(f.Vprod.sum(), 1e-9)),
                las_p=float((f.las_p * f.Vprod).sum() / max(f.Vprod.sum(), 1e-9)),
                lrv_ec=float(f.lrv_ec.mean()) if len(f) else np.nan, lrv_mp=float(f.lrv_mp.mean()) if len(f) else np.nan,
                tmp_mean=float(f.tmp.mean()) if len(f) else np.nan, tmp_p95=float(np.percentile(f.tmp, 95)) if len(f) else np.nan,
                norm_perm=norm_perm, norm_perm_p10=p10,
                n_cip=int(plant.n_cip), naocl_g=float(naocl), reg_h_d=n_reg / days,
                sec_kwh_m3=E / max(prod, 1e-9), a_end=float(plant.a), sf_end=float(plant.sf),
                life_coat_y=float(min(life_coat, 10.0)), life_memb_y=float(life_memb),
                cost_m3=float(cost), gwp_m3=float(gwp), Rs_end=float(plant.Rs), Rh_end=float(plant.Rh))
