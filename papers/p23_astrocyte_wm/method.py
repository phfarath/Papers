"""Mecanismo central do paper p23 — memória de trabalho em SNN + astrócitos.

Core mechanism of p23 — working memory in a spiking/rate network with
astrocytes (Gordleeva et al. 2021).

Ideia: um padrão ativado brevemente (o "item" de WM) é sustentado por
conectividade recorrente. Astrócitos integram a atividade local numa
variável lenta tipo Ca²⁺; quando o Ca²⁺ cruza um limiar, as sinapses entre
neurônios co-ativos são potencializadas (min(g_i, g_j) — a sinapse
tripartida precisa dos dois lados ativos). Um item sustentado tempo
suficiente ganha persistência; um distrator breve não cruza o limiar e é
suprimido pela inibição global — robustez a distratores.

Idea: a briefly activated pattern (the WM "item") is sustained by
recurrent connectivity. Astrocytes integrate local activity in a slow
Ca²⁺-like variable; when Ca²⁺ crosses a threshold, synapses between
co-active neurons are potentiated (min(g_i,g_j) — the tripartite synapse
needs both sides active). A sustained item gains persistence; a brief
distractor never crosses the threshold and is suppressed by global
inhibition — distractor robustness.
"""

import numpy as np


class WMNetwork:
    """Rede recorrente rate-based (SNN-lite) com modulação astroglial.

    Recurrent rate network (SNN-lite) with astrocytic modulation.

    Dynamics per step:
        u += dt (−u + W_eff·r + I − inhib·mean(r) + noise)/τ_n
        r  = max(0, tanh(u))
        C += dt (−C + r)/τ_C                      (astrocyte Ca²⁺ trace)
        W_eff = W · (1 + α·min(g_i, g_j)),  g = max(0, C − C_thr)
    """

    def __init__(
        self,
        n=60,
        n_asm=10,
        astro=True,
        alpha=3.5,
        tau_c=1.0,
        c_thr=0.35,
        w_in=0.12,
        w_out=0.02,
        inhib=1.5,
        tau_n=0.02,
        noise=0.4,
        seed=0,
    ):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.n, self.n_asm = n, n_asm
        self.asm = np.arange(n_asm)
        self.distr = np.arange(n_asm, 2 * n_asm)
        W = rng.uniform(0, w_out, (n, n))
        W[np.ix_(self.asm, self.asm)] += w_in
        W[np.ix_(self.distr, self.distr)] += w_in
        np.fill_diagonal(W, 0.0)
        self.W = W
        self.astro = astro
        self.alpha, self.tau_c, self.c_thr = alpha, tau_c, c_thr
        self.inhib, self.tau_n, self.noise = inhib, tau_n, noise
        self.u = np.zeros(n)
        self.r = np.zeros(n)
        self.C = np.zeros(n)

    def step(self, drive, dt):
        g = np.maximum(0.0, self.C - self.c_thr)
        We = self.W * (1.0 + self.alpha * np.minimum.outer(g, g)) if self.astro else self.W
        self.u += (
            dt
            * (
                -self.u
                + We @ self.r
                + drive
                - self.inhib * self.r.mean()
                + self.rng.normal(0, self.noise, self.n)
            )
            / self.tau_n
        )
        self.r = np.maximum(0.0, np.tanh(self.u))
        self.C += dt * (-self.C + self.r) / self.tau_c
        return self.r

    def simulate(
        self,
        t_total=4.0,
        dt=0.005,
        cue_dur=1.0,
        dist_at=2.0,
        dist_dur=0.3,
        cue_amp=3.0,
        dist_amp=2.2,
    ):
        """Protocolo WM: cue no assembly → delay → distrator → leitura.

        WM protocol: cue the item assembly → delay → distractor → readout.
        Returns dict with final rates and the item/distractor rate traces.
        """
        item_tr, dist_tr, c_tr = [], [], []
        for tt in np.arange(0, t_total, dt):
            drive = np.zeros(self.n)
            if tt < cue_dur:
                drive[self.asm] = cue_amp
            if dist_at <= tt < dist_at + dist_dur:
                drive[self.distr] = dist_amp
            self.step(drive, dt)
            item_tr.append(self.r[self.asm].mean())
            dist_tr.append(self.r[self.distr].mean())
            c_tr.append(self.C[self.asm].mean())
        return {
            "item": float(self.r[self.asm].mean()),
            "distractor": float(self.r[self.distr].mean()),
            "item_tr": np.array(item_tr),
            "dist_tr": np.array(dist_tr),
            "ca_tr": np.array(c_tr),
            "held": float(self.r[self.asm].mean()) > 2 * float(self.r[self.distr].mean()),
        }


def run_batch(astro, seeds, **kw):
    """Agrega `simulate` sobre seeds. / Aggregate `simulate` over seeds."""
    items, dists, held = [], [], []
    for s in seeds:
        res = WMNetwork(astro=astro, seed=s, **kw).simulate()
        items.append(res["item"])
        dists.append(res["distractor"])
        held.append(res["held"])
    return {
        "item": float(np.mean(items)),
        "item_std": float(np.std(items)),
        "distractor": float(np.mean(dists)),
        "held": float(np.mean(held)),
    }
