"""Structure-property model of the asymmetric rice-husk-ash (RHA)/kaolin
ceramic membrane: biochar-templated macroporous support plus a thin
RHA-silica/photocatalyst separation layer.

Relations (calibrated to ranges reported for low-cost RHA, kaolin and
biochar-templated ceramics; see manuscript Section III-B):
  porosity      eps = eps0 + a_b*w_b - a_T*(Ts-1000)/100
  pore diameter d   = d0*(1+c_b*w_b)*exp(-c_T*(Ts-1000)/100)
  permeability  k   = eps*d^2/(32*tau^2), tau = eps^-1/2 (Bruggeman)
  strength      sf  = s0(Ts)*exp(-b*eps)  (Ryshkewitch)
  sieving       flow-weighted log-normal pore model with steric hindrance
"""
import numpy as np

P = dict(eps0=0.30, a_b=0.012, a_T=0.045, d0=0.35, c_b=0.05, c_T=0.12,
         s0max=120.0, s0T=150.0, b=3.5, c_f=0.025, Ls=3.0e-3,
         top_eps=0.40, top_d=0.20, top_sg=1.45, top_L=15e-6)


def viscosity(T):
    """Water viscosity [Pa s] (Vogel-type correlation)."""
    return 2.414e-5 * 10 ** (247.8 / (T + 133.15))


def support(w_b, Ts, p=P):
    eps = p["eps0"] + p["a_b"] * w_b - p["a_T"] * (Ts - 1000.0) / 100.0
    eps = np.clip(eps, 0.15, 0.65)
    d = p["d0"] * (1 + p["c_b"] * w_b) * np.exp(-p["c_T"] * (Ts - 1000.0) / 100.0) * 1e-6
    tau = eps ** -0.5
    k = eps * d ** 2 / (32 * tau ** 2)
    s0 = p["s0max"] * (1 - np.exp(-(Ts - 900.0) / p["s0T"]))
    sf = s0 * np.exp(-p["b"] * eps) * np.exp(-p["c_f"] * w_b)  # macropore flaws
    return eps, d, k, sf


def top_layer_k(p=P):
    eps, d = p["top_eps"], p["top_d"] * 1e-6
    return eps * d ** 2 / (32 * eps ** -1.0)


def intrinsic_resistance(w_b, Ts, coated=True, p=P):
    eps, d, k, sf = support(w_b, Ts, p)
    R = p["Ls"] / k
    if coated:
        R = R + p["top_L"] / top_layer_k(p)
    else:
        # uncoated reference: RHA-silica layer only, same pore size, 70 % thickness
        R = R + 0.7 * p["top_L"] / top_layer_k(p)
    return R  # m-1


def permeability_LMHbar(R, T=20.0):
    return 1.0 / (viscosity(T) * R) * 3.6e6 * 1e5


def sieving_lrv(dpart_um, p=P, n=4000):
    """Log removal value of rigid particles of diameter dpart for the
    log-normal top-layer pore population (flow ~ r^4, hindrance (1-l)^2)."""
    rng = np.random.default_rng(0)
    d = p["top_d"] * np.exp(np.log(p["top_sg"]) * rng.standard_normal(n))
    w = d ** 4
    lam = np.clip(dpart_um / d, 0, 1)
    T = np.sum(w * (1 - lam) ** 2 * (1 - 0.5 * lam ** 2) * (lam < 1)) / np.sum(w)
    return min(-np.log10(max(T, 1e-9)), 6.5)


def design_space():
    wb = np.linspace(0, 25, 26); Ts = np.linspace(950, 1250, 31)
    W, T = np.meshgrid(wb, Ts)
    eps, d, k, sf = support(W, T)
    R = intrinsic_resistance(W, T)
    Lp = permeability_LMHbar(R)
    return W, T, eps, d * 1e6, Lp, sf


def select_design(sf_min=12.0):
    W, T, eps, d, Lp, sf = design_space()
    ok = sf >= sf_min
    i = np.argmax(np.where(ok, Lp, -1))
    j = np.unravel_index(i, W.shape)
    return dict(w_b=float(W[j]), Ts=float(T[j]), eps=float(eps[j]), d_um=float(d[j]),
                Lp=float(Lp[j]), sf=float(sf[j]), R_m=float(intrinsic_resistance(W[j], T[j])))


if __name__ == "__main__":
    d = select_design()
    print(d)
    print("uncoated Rm", intrinsic_resistance(d["w_b"], d["Ts"], coated=False))
    for dp in (0.5, 0.6, 1.0, 2.0, 5.0):
        print(dp, round(sieving_lrv(dp), 2))
