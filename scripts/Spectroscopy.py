import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from ase.io import read

# =========================
# Constants and conversions
# =========================
kB = 8.617333262145e-5          # eV K^-1
hartree2eV = 27.211396641308
bohr2A = 0.529177249
eps0const = 5.5263499562e-3     # [e * Volt^{-1} * Å^{-1}]
THz2cminv = 0.03335640951981521 * 1e3
sig_g = 20.0                    # Gaussian broadening (cm^-1)
cm_inv2Ry = 0.000124 * 13.605698320654
c_mlmd = "#0055d4"              # legacy color (unused now, we style in-class)

# =========================
# Utilities (broadening etc)
# =========================
def gaussian(x, A, mu, sig):
    """A·N(μ,σ)"""
    return A / np.sqrt(2.0 * np.pi * sig**2) * np.exp(-0.5 * ((x - mu) / sig) ** 2)

def gaussian_kernel(x, sig_g):
    """Normalized Gaussian kernel centered at 0."""
    k = np.exp(-0.5 * (x / sig_g) ** 2)
    return k / np.sum(k)

def gaussian_broaden(ω, data, sig_g, mode=0):
    """
    Apply Gaussian broadening to a spectrum sampled on ω (cm^-1).
      mode 0: symmetric via convolution (safe near 0)
      mode 1: place a Gaussian at each point (DFPT-like; asymmetric near 0)
    """
    if sig_g is None or sig_g <= 0:
        return data
    ω = np.asarray(ω)
    data = np.asarray(data)

    if mode == 0:
        ω_2 = np.concatenate((-ω[::-1][:-1], ω))
        data_2 = np.concatenate((data[::-1][:-1], data))
        ker = gaussian_kernel(ω_2, sig_g)
        out = np.convolve(data_2, ker, mode="same")[len(data)-1:len(ω_2)]
        return out
    else:
        out = np.zeros_like(ω, dtype=float)
        for i, amp in enumerate(data):
            g = gaussian(ω, 1.0, ω[i], sig_g)
            out += amp * g / np.sum(g)
        return out

# =========================
# Spectroscopy class
# =========================
class Spectroscopy:
    """Analyse vibrational and dielectric properties from MLMD data."""

    # ----- styling & small helpers -----
    @staticmethod
    def _set_pub_style():
        import matplotlib as mpl
        mpl.rcParams.update({
            "figure.dpi": 150, "savefig.dpi": 300,
            "font.size": 11, "axes.labelsize": 12.5, "axes.titlesize": 12.5,
            "axes.linewidth": 1.0, "xtick.direction": "out", "ytick.direction": "out",
            "xtick.major.size": 4, "ytick.major.size": 4, "xtick.minor.size": 2, "ytick.minor.size": 2,
            "legend.frameon": False,
        })

    @staticmethod
    def _pretty_axes(ax):
        from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="minor", length=2)
        sf = ScalarFormatter(useMathText=True)
        sf.set_powerlimits((-2, 3))
        ax.yaxis.set_major_formatter(sf)

    @staticmethod
    def _robust_ylim(y, lo=0, hi=100, pad=0.07):
        y = np.asarray(y)
        y = y[np.isfinite(y)]
        if y.size == 0:
            return (-1, 1)
        ql, qh = np.percentile(y, [lo, hi])
        if ql == qh:
            ql -= 1.0; qh += 1.0
        rng = qh - ql
        return (ql - pad*rng, qh + pad*rng)

    @staticmethod
    def _linefill(
        ax, x, y, *,
        color="C0",
        label=None,
        lw=1.8,
        alpha=0.95,
        fill_alpha=0.25,
        do_fill=True,
        **plot_kwargs,
    ):
        """
        Helper to plot a line, optionally with filled area to zero.

        do_fill=True  -> line + filled area
        do_fill=False -> line only (for comparison plots)
        """
        line, = ax.plot(x, y, color=color, lw=lw, alpha=alpha,
                        label=label, **plot_kwargs)
        if do_fill:
            ax.fill_between(x, 0*y, y, color=color,
                            alpha=fill_alpha, linewidth=0)
        return line
    
    @staticmethod
    def _maybe_smear(x, y, sigma_cm1, do_smear=True):
        if (not do_smear) or (sigma_cm1 is None) or (sigma_cm1 <= 0):
            return y
        return gaussian_broaden(x, y, sigma_cm1, mode=0)

    @staticmethod
    def _normalize(y, mode="none", x=None):
        """'none' | 'var' (no-op) | 'area' (unit positive area)."""
        if mode == "none":
            return y
        if mode == "var":
            return y
        if mode == "area":
            if x is None:
                return y
            area = np.trapz(np.clip(y, a_min=0.0, a_max=None), x)
            return y / area if area > 0 else y
        raise ValueError("normalize must be 'none', 'var', or 'area'.")

    # ----- constructor & parsing -----
    def __init__(self, file_mlmd, dt, temp, ωi, ωf, do_IR=True, do_Raman=True, freq_damp=True):
        """
        file_mlmd: ASE-readable trajectory with arrays/info for polarization etc.
        dt (fs)   : timestep in femtoseconds
        temp (K)  : temperature (for ε_ion)
        ωi/ωf     : plotting window (cm^-1)
        """
        self.dt = dt
        self.temp = temp
        self.ωi = ωi
        self.ωf = ωf
        self.do_IR = do_IR
        self.do_Raman = do_Raman

        self.parse_mlmd(file_mlmd)

        if self.do_IR:
            self.get_autocorr_P(freq_damp=freq_damp)
        if self.do_Raman:
            self.get_autocorr_α()

        if self.do_IR:
            self.get_IR(normalize="area")         # choose 'area' for shape comparability
        if self.do_Raman:
            self.get_Raman(normalize="area")

        self.get_εinf_ε0()
        if self.do_IR:
            self.get_εω_mlmd()

    def unit_pol(self, P):
        """
        Wrap polarization into principal branch of the polarization lattice.
        P: (n_t,3) in e/Å^2.
        """
        lattice = self.p          # 3x3 Å
        V = self.V                # Å^3
        P_q = lattice / V         # e·R/V; P already e/Å^2 so e-factor implicit
        # Convert to fractional polarization (units of P_q)
        P_frac = np.linalg.solve(P_q.T, P.T).T
        # Wrap into [-0.5, 0.5)
        P_frac = (P_frac + 0.5) % 1.0 - 0.5
        return P_frac @ P_q.T

    def parse_mlmd(self, file_mlmd):
        """Parse required arrays from trajectory. (Assumes orthorhombic cell.)"""
        # NOTE: drop the first chunk if you want – keep as in your original
        data = read(file_mlmd, ":")[10000:]

        self.V = data[0].get_volume()
        self.n_t = len(data)
        self.time = np.arange(self.n_t) * self.dt / 1000.0  # fs -> ps

        self.E = np.asarray([a.get_total_energy() for a in data])
        self.F = np.asarray([a.get_forces() for a in data]).reshape(self.n_t, -1, 3)
        self.P = np.asarray([a.info["MACE_polarisation"] for a in data]).reshape(self.n_t, 3)
        self.Z = np.asarray([a.arrays["MACE_becs"] for a in data]).reshape(self.n_t, -1, 3, 3)
        self.α = np.asarray([a.info["MACE_polarisability"] for a in data]).reshape(self.n_t, 3, 3)

        # Apply modulo polarization
        self.p = data[0].get_cell()
        self.P = self.unit_pol(self.P)

        # Useful quantities
        self.n_ω = int(self.n_t/2) + 1                   # rFFT length
        self.var_P = np.var(self.P, axis=0, dtype=np.float64)
        self.var_α = np.array([np.var(self.α[:, i, i], dtype=np.float64) for i in range(3)])
        self.P_avg = np.mean(self.P, axis=0)
        self.α_avg = np.array([np.mean(self.α[:, i, i]) for i in range(3)])

    # ----- autocorrelations -----
    def get_autocorr_mlmd(self, time, signal):
        """
        Autocorrelation of a real signal and its spectrum.
        Returns (freq_cm^-1, C(t), S(ω) complex).
        """
        signal = np.asarray(signal)
        N = len(signal)
        dt_ps = time[1] - time[0]
        # Frequency axis (cm^-1) for rFFT
        freq_cminv = np.fft.rfftfreq(N, dt_ps) * THz2cminv

        signal_centered = signal - signal.mean()
        C_t = np.correlate(signal_centered, signal_centered, mode="full")[N-1:] / N
        S_w = np.fft.rfft(C_t) * (dt_ps / THz2cminv) * np.pi  # proportional spectrum

        return freq_cminv, C_t, S_w

    def get_autocorr_P(self, freq_damp=True):
        """Polarization autocorrelation (normalized over components)."""
        num = self.P.shape[1]
        self.P_AF_t = np.zeros((self.n_t, num))
        self.P_AF_ω = np.zeros((self.n_ω, num), dtype=complex)

        for i in range(num):
            self.ω, self.P_AF_t[:, i], self.P_AF_ω[:, i] = self.get_autocorr_mlmd(self.time, self.P[:, i])
            if freq_damp:
                real = self.P_AF_ω[:, i].real.copy()
                imag = self.P_AF_ω[:, i].imag
                thr = 0.01 * np.max(np.abs(real))
                real[np.abs(real) < thr] = 0.0
                self.P_AF_ω[:, i] = real + 1j * imag

        norm = np.sum(self.var_P)
        if norm == 0:
            raise ValueError("Sum of var_P is zero, cannot normalise.")
        self.P_AF_t_av = np.sum(self.P_AF_t, axis=1) / norm
        self.P_AF_ω_av = np.sum(self.P_AF_ω, axis=1) / norm   # <— FIX: no stray minus

    def get_autocorr_α(self):
        """Full 3×3 α(t) autocorrelation (normalized)."""
        ncomp = 3
        tot = ncomp * ncomp
        self.α_AF_t = np.zeros((self.n_t, tot))
        self.α_AF_ω = np.zeros((self.n_ω, tot), dtype=complex)

        idx = 0
        for i in range(ncomp):
            for j in range(ncomp):
                sig = self.α[:, i, j]
                self.ω, self.α_AF_t[:, idx], self.α_AF_ω[:, idx] = self.get_autocorr_mlmd(self.time, sig)
                idx += 1

        norm = np.sum(self.var_α)
        if norm == 0:
            raise ValueError("Sum of var_α is zero, cannot normalize.")
        self.α_AF_t_av = np.sum(self.α_AF_t, axis=1) / norm
        self.α_AF_ω_av = np.sum(self.α_AF_ω, axis=1) / norm

    # ----- spectra from autocorrelations (fixed normalization/sign) -----
    def get_IR(self, normalize="none"):
        """IR(ω) ∝ ω² · Re S_P(ω)."""
        spec = np.clip(self.P_AF_ω_av.real, a_min=0.0, a_max=None) * (self.ω ** 2)
        self.IR = self._normalize(spec, normalize, x=self.ω)

    def get_Raman(self, normalize="none"):
        """Raman(ω) ∝ ω² · Re S_α(ω)."""
        spec = np.clip(self.α_AF_ω_av.real, a_min=0.0, a_max=None) * (self.ω ** 2)
        self.Raman = self._normalize(spec, normalize, x=self.ω)

    # ----- dielectric constants -----
    def get_εinf_ε0(self):
        """Compute ε∞, εion, ε0 from α and P fluctuations (xyz-averaged stored later)."""
        self.εinf = np.zeros(3)
        self.εion = np.zeros(3)
        self.ε0   = np.zeros(3)
        for i in range(3):
            self.εinf[i] = 1 + np.mean(self.α[:, i, i]) / eps0const
            self.εion[i] = self.V * self.var_P[i] / (kB * self.temp) / eps0const
            self.ε0[i]   = self.εinf[i] + self.εion[i]
        print(self.εinf, self.εion, self.ε0)

    def get_εω_mlmd(self):
        """Frequency-dependent dielectric function (xyz-averaged later)."""
        ε_ω = np.zeros((self.n_ω, 3), dtype=complex)
        for i in range(3):
            ε_ω[:, i] = 1 + (self.ε0[i] - 1) * (1 - 1j * self.ω[:] * self.P_AF_ω[:, i] / self.var_P[i])
        self.ε_ω_av_re = np.sum(ε_ω.real, axis=1) / 3.0
        self.ε_ω_av_im = np.sum(ε_ω.imag, axis=1) / 3.0

    # ======================
    # Publication-quality plots
    # ======================
    @staticmethod
    def _square(axs):
        """Force square axes box aspect on one or many axes."""
        try:
            for a in np.ravel(axs):
                a.set_box_aspect(1)
        except Exception:
            axs.set_box_aspect(1)

    # ---------- autocorrelation (square) ----------
    def plot_autocorr_P(self, title="Polarization autocorrelation"):
        self._set_pub_style()
        fig, ax = plt.subplots(figsize=(4.2, 4.2), constrained_layout=False)
        plt.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.88)
        self._linefill(ax, self.time, self.P_AF_t_av, color="C0",
                    label=r"$\langle P(t)\!\cdot\!P(0)\rangle/\mathrm{var}(P)$")
        ax.set_xlabel("time (ps)"); ax.set_ylabel("normalized autocorr.")
        ax.set_title(title)
        ax.set_ylim(*self._robust_ylim(self.P_AF_t_av))
        self._pretty_axes(ax); ax.legend(loc="best")
        self._square(ax)
        return fig, ax

    def plot_autocorr_α(self, title="Polarizability autocorrelation"):
        self._set_pub_style()
        fig, ax = plt.subplots(figsize=(4.2, 4.2), constrained_layout=False)
        plt.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.88)
        self._linefill(ax, self.time, self.α_AF_t_av, color="C1",
                    label=r"$\langle \alpha(t)\!\cdot\!\alpha(0)\rangle/\mathrm{var}(\alpha)$")
        ax.set_xlabel("time (ps)"); ax.set_ylabel("normalized autocorr.")
        ax.set_title(title)
        ax.set_ylim(*self._robust_ylim(self.α_AF_t_av))
        self._pretty_axes(ax); ax.legend(loc="best")
        self._square(ax)
        return fig, ax

    # ---------- IR / Raman (square) ----------
    def plot_IR(
        self,
        smear=True,
        sigma_cm1=20.0,
        title="IR spectrum",
        label_main="MLMD",
        others=None,
        labels_others=None,
        colors_others=None,
    ):
        """
        IR spectrum. Optionally overlay spectra from other Spectroscopy objects.

        others: list or single Spectroscopy instance
        labels_others: list of legend labels for others
        colors_others: list of colors for others (defaults to C1, C2, ...)
        """
        self._set_pub_style()
        y_main = self._maybe_smear(self.ω, self.IR, sigma_cm1, smear)

        fig, ax = plt.subplots(figsize=(4.4, 4.4), constrained_layout=False)
        plt.subplots_adjust(left=0.18, right=0.98, bottom=0.20, top=0.88)
        ax.set_xlim(self.ωi, self.ωf)

        # main curve (filled)
        self._linefill(ax, self.ω, y_main, color="#4C78A8",
                       label=label_main, do_fill=True)
        y_all = [y_main]

        # handle overlays
        if others is not None:
            if not isinstance(others, (list, tuple)):
                others = [others]

            n_others = len(others)
            if labels_others is None:
                labels_others = [f"comp {i+1}" for i in range(n_others)]
            if colors_others is None:
                colors_others = [f"C{i+1}" for i in range(n_others)]

            for other, lab, col in zip(others, labels_others, colors_others):
                y_o = self._maybe_smear(other.ω, other.IR, sigma_cm1, smear)
                self._linefill(ax, other.ω, y_o, color=col,
                               label=lab, do_fill=False, lw=1.6)
                y_all.append(y_o)

        ax.set_ylim(*self._robust_ylim(np.concatenate([np.asarray(v) for v in y_all])))
        ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
        ax.set_ylabel("IR intensity (a.u.)")
        ax.set_title(title)
        self._pretty_axes(ax)
        ax.legend(loc="upper right")
        self._square(ax)
        return fig, ax


    def plot_raman(
        self,
        smear=True,
        sigma_cm1=20.0,
        title="Raman spectrum",
        label_main="MLMD",
        others=None,
        labels_others=None,
        colors_others=None,
    ):
        """
        Raman spectrum with optional overlay of other Spectroscopy objects.
        """
        self._set_pub_style()
        y_main = self._maybe_smear(self.ω, self.Raman, sigma_cm1, smear)

        fig, ax = plt.subplots(figsize=(4.4, 4.4), constrained_layout=False)
        plt.subplots_adjust(left=0.18, right=0.98, bottom=0.20, top=0.88)
        ax.set_xlim(self.ωi, self.ωf)

        # main curve (filled)
        self._linefill(ax, self.ω, y_main, color="#F58518",
                       label=label_main, do_fill=True)
        y_all = [y_main]

        # overlays
        if others is not None:
            if not isinstance(others, (list, tuple)):
                others = [others]

            n_others = len(others)
            if labels_others is None:
                labels_others = [f"comp {i+1}" for i in range(n_others)]
            if colors_others is None:
                colors_others = [f"C{i+1}" for i in range(n_others)]

            for other, lab, col in zip(others, labels_others, colors_others):
                y_o = self._maybe_smear(other.ω, other.Raman, sigma_cm1, smear)
                self._linefill(ax, other.ω, y_o, color=col,
                               label=lab, do_fill=False, lw=1.6)
                y_all.append(y_o)

        ax.set_ylim(*self._robust_ylim(np.concatenate([np.asarray(v) for v in y_all])))
        ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
        ax.set_ylabel("Raman intensity (a.u.)")
        ax.set_title(title)
        self._pretty_axes(ax)
        ax.legend(loc="upper right")
        self._square(ax)
        return fig, ax


    # ---------- ε(ω): two square axes stacked ----------
    def plot_ε(
        self,
        smear=True,
        sigma_cm1=20.0,
        title="Dielectric function",
        label_main="MLMD",
        others=None,
        labels_others=None,
        colors_re_others=None,
        colors_im_others=None,
    ):
        """
        ε(ω) plots (Re and -Im) with optional overlays from other Spectroscopy objects.

        others: list/single Spectroscopy
        labels_others: legend labels
        colors_re_others: colors for Re[ε] overlays
        colors_im_others: colors for -Im[ε] overlays
        """
        self._set_pub_style()

        y_re = self.ε_ω_av_re.copy()
        y_im = -self.ε_ω_av_im.copy()
        if smear:
            y_re = self._maybe_smear(self.ω, y_re, sigma_cm1, True)
            y_im = self._maybe_smear(self.ω, y_im, sigma_cm1, True)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(4.6, 9.4),
            sharex=True, constrained_layout=False
        )
        plt.subplots_adjust(left=0.18, right=0.98, bottom=0.12, top=0.92, hspace=0.12)

        # ---------- Re[ε] ----------
        ax1.set_title(title)
        ax1.set_xlim(self.ωi, self.ωf)

        self._linefill(ax1, self.ω, y_re, color="#4C78A8",
                       label=label_main, do_fill=True)
        eps_inf_bar = float(np.mean(self.εinf))
        eps_0_bar   = float(np.mean(self.ε0))
        # ax1.axhline(eps_inf_bar - 1, color="0.4", ls="--", lw=1.0,
        #             label=r"$\bar{\epsilon}_\infty$")
        # ax1.axhline(eps_0_bar, color="0.6", ls="--", lw=1.0,
        #             label=r"$\bar{\epsilon}_0$")

        yre_all = [y_re]

        # overlays
        if others is not None:
            if not isinstance(others, (list, tuple)):
                others = [others]
            n_others = len(others)
            if labels_others is None:
                labels_others = [f"comp {i+1}" for i in range(n_others)]
            if colors_re_others is None:
                colors_re_others = [f"C{i+1}" for i in range(n_others)]

            for other, lab, col in zip(others, labels_others, colors_re_others):
                y_re_o = other.ε_ω_av_re.copy()
                if smear:
                    y_re_o = self._maybe_smear(other.ω, y_re_o, sigma_cm1, True)
                self._linefill(ax1, other.ω, y_re_o, color=col,
                               label=lab, do_fill=True, lw=1.6)
                yre_all.append(y_re_o)

        yre_for_ylim = np.concatenate([np.asarray(v) for v in yre_all] +
                                      [np.atleast_1d(eps_inf_bar), np.atleast_1d(eps_0_bar)])
        ax1.set_ylim(*self._robust_ylim(yre_for_ylim))
        ax1.set_ylabel(r"Re[$\epsilon(\omega)$]")
        self._pretty_axes(ax1)
        ax1.legend(ncol=3, loc="upper right")

        # ---------- -Im[ε] ----------
        self._linefill(ax2, self.ω, y_im, color="#E45756",
                       label=label_main, do_fill=True)
        yim_all = [y_im]

        if others is not None:
            if colors_im_others is None:
                colors_im_others = [f"C{i+1}" for i in range(len(others))]
            for other, lab, col in zip(others, labels_others, colors_im_others):
                y_im_o = -other.ε_ω_av_im.copy()
                if smear:
                    y_im_o = self._maybe_smear(other.ω, y_im_o, sigma_cm1, True)
                self._linefill(ax2, other.ω, y_im_o, color=col,
                               label=lab, do_fill=True, lw=1.6)
                yim_all.append(y_im_o)

        ax2.set_ylim(*self._robust_ylim(np.concatenate([np.asarray(v) for v in yim_all])))
        ax2.set_xlabel(r"$\omega$ (cm$^{-1}$)")
        ax2.set_ylabel(r"$-\mathrm{Im}\,\epsilon(\omega)$")
        self._pretty_axes(ax2)
        ax2.legend(loc="upper right")

        self._square([ax1, ax2])
        return fig, (ax1, ax2)


    # ---------- 2×2 grid with square panels ----------
    def plot_spectra_grid(
        self,
        smear=True,
        sigma_cm1=20.0,
        label_main="Direct",
        others=None,
        labels_others=None,
        colors_others=None,
    ):
        """
        2×2 grid: IR, Raman, Re[ε], and -Im[ε] with a shared x-axis.

        Allows overlay of spectra from other Spectroscopy instances.

        others: list/single Spectroscopy
        labels_others: legend labels (used in all four panels)
        colors_others: list of colors for others (same color used for all panels)
        """
        self._set_pub_style()
        fig, axs = plt.subplots(
            2, 2,
            figsize=(9.0, 9.0),
            sharex=True,          # shared x only
            constrained_layout=False
        )
        plt.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.95,
                            wspace=0.26, hspace=0.22)

        # normalise "others" handling
        if others is not None and not isinstance(others, (list, tuple)):
            others = [others]
        if others is not None:
            n_others = len(others)
            if labels_others is None:
                labels_others = [f"comp {i+1}" for i in range(n_others)]
            if colors_others is None:
                colors_others = [f"C{i+1}" for i in range(n_others)]

        # -------- IR (top-left) --------
        ax = axs[0, 0]
        y_main = self._maybe_smear(self.ω, self.IR, sigma_cm1, smear)
        self._linefill(ax, self.ω, y_main, color="#4C78A8",
                       label=label_main, do_fill=True)
        y_all = [y_main]

        if others is not None:
            for other, lab, col in zip(others, labels_others, colors_others):
                y_o = self._maybe_smear(other.ω, other.IR, sigma_cm1, smear)
                self._linefill(ax, other.ω, y_o, color="#F58518",
                               label=lab, do_fill=True, lw=1.6)
                y_all.append(y_o)

        ax.set_ylim(*self._robust_ylim(np.concatenate([np.asarray(v) for v in y_all])))
        self._pretty_axes(ax)
        ax.set_title("IR")
        ax.set_ylabel("intensity (a.u.)")
        ax.legend()

        # -------- Raman (top-right) --------
        ax = axs[0, 1]
        y_main = self._maybe_smear(self.ω, self.Raman, sigma_cm1, smear)
        self._linefill(ax, self.ω, y_main, color="#4C78A8",
                       label=label_main, do_fill=True)
        y_all = [y_main]

        if others is not None:
            for other, lab, col in zip(others, labels_others, colors_others):
                y_o = self._maybe_smear(other.ω, other.Raman, sigma_cm1, smear)
                self._linefill(ax, other.ω, y_o, color="#F58518",
                               label=lab, do_fill=True, lw=1.6)
                y_all.append(y_o)

        ax.set_ylim(*self._robust_ylim(np.concatenate([np.asarray(v) for v in y_all])))
        self._pretty_axes(ax)
        ax.set_title("Raman")
        ax.legend()

        # -------- Re[ε] (bottom-left) --------
        ax = axs[1, 0]
        y_main = self.ε_ω_av_re.copy()
        if smear:
            y_main = self._maybe_smear(self.ω, y_main, sigma_cm1, True)
        self._linefill(ax, self.ω, y_main, color="#4C78A8",
                       label=label_main, do_fill=True)
        eps_inf_bar = float(np.mean(self.εinf))
        eps_0_bar   = float(np.mean(self.ε0))
        # ax.axhline(eps_inf_bar - 1, color="0.4", ls="--", lw=1.0,
        #            label=r"$\bar{\epsilon}_\infty$")
        # ax.axhline(eps_0_bar, color="0.6", ls="--", lw=1.0,
        #            label=r"$\bar{\epsilon}_0$")

        y_all = [y_main]

        if others is not None:
            for other, lab, col in zip(others, labels_others, colors_others):
                y_o = other.ε_ω_av_re.copy()
                if smear:
                    y_o = self._maybe_smear(other.ω, y_o, sigma_cm1, True)
                self._linefill(ax, other.ω, y_o, color="#F58518",
                               label=lab, do_fill=True, lw=1.6)
                y_all.append(y_o)

        y_for_ylim = np.concatenate(
            [np.asarray(v) for v in y_all] +
            [np.atleast_1d(eps_inf_bar), np.atleast_1d(eps_0_bar)]
        )
        ax.set_ylim(*self._robust_ylim(y_for_ylim))
        self._pretty_axes(ax)
        ax.set_title(r"Re $\epsilon$")
        ax.set_ylabel(r"$\epsilon$ $(\varepsilon_0)$")
        ax.legend()

        # -------- -Im[ε] (bottom-right) --------
        ax = axs[1, 1]
        y_main = -self.ε_ω_av_im.copy()
        if smear:
            y_main = self._maybe_smear(self.ω, y_main, sigma_cm1, True)
        self._linefill(ax, self.ω, y_main, color="#4C78A8",
                       label=label_main, do_fill=True)
        y_all = [y_main]

        if others is not None:
            for other, lab, col in zip(others, labels_others, colors_others):
                y_o = -other.ε_ω_av_im.copy()
                if smear:
                    y_o = self._maybe_smear(other.ω, y_o, sigma_cm1, True)
                self._linefill(ax, other.ω, y_o, color="#F58518",
                               label=lab, do_fill=True, lw=1.6)
                y_all.append(y_o)

        ax.set_ylim(*self._robust_ylim(np.concatenate([np.asarray(v) for v in y_all])))
        self._pretty_axes(ax)
        ax.set_title(r"Imag $\epsilon$")
        ax.set_ylabel(r"$-\mathrm{Im}\,\epsilon$")
        ax.legend()

        # shared x settings
        axs[1, 0].set_xlim(self.ωi, self.ωf)

        for ax in axs[0, :]:
            ax.tick_params(labelbottom=False)

        fig.supxlabel(r"$\omega$ (cm$^{-1}$)")
        self._square(axs)
        return fig, axs


