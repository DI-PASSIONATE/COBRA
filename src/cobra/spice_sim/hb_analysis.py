import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Column name and axis label for each plot_quantity mode
_QUANTITY_META = {
    "power":   {"col": "P_out_dBm",   "ylabel": "Power (dBm)",    "unit": "dBm"},
    "voltage": {"col": "V_out_dBV",   "ylabel": "Voltage (dBV)",  "unit": "dBV"},
    "current": {"col": "I_out_dBmA",  "ylabel": "Current (dBmA)", "unit": "dBmA"},
}


class HBAnalysis:
    def __init__(
        self,
        filename: str,
        sim_type: str = "hb",
        freq_fundamental: float | None = None,
        v_out: str | None = None,
        i_out: str | None = None,
        plot_quantity: str = "power",
        settle_time: float = 0.0,
        settle_fraction: float = 0.0,
    ):
        """
        Parses a Xyce .prn simulation dataset and plots the output spectrum.

        Parameters
        ----------
        filename         : path to the Xyce .prn or .FD.prn result file
        sim_type         : "hb"        — Harmonic Balance result (frequency-domain phasors)
                           "transient" — Transient result (time-domain); FFT is applied
                                         internally so the result matches HB convention.
        freq_fundamental : fundamental RF frequency in Hz
                           - LNA / PA (single tone): e.g. 130e9
                           - Mixer / two-tone HB:    None  (all mixing products shown)
                           - Transient:              set to filter FFT output to harmonics;
                                                     None shows all FFT bins
        v_out            : voltage signal name, e.g. "V(OUT)"
                           Required for plot_quantity="power" or "voltage"; otherwise ignored.
        i_out            : current signal name of the 0V ammeter/vsource in series with the load,
                           e.g. "I(VOUT)"
                           Required for plot_quantity="power" or "current"; otherwise ignored.
        plot_quantity    : what to plot on the y-axis:
                           "power"   — output power P [dBm] = 10*log10(|V_pk*I_pk|/1mW)
                                       needs both v_out and i_out
                           "voltage" — voltage magnitude [dBV] = 20*log10(|V_pk|)
                                       needs only v_out
                           "current" — current magnitude [dBmA] = 20*log10(|I_pk|/1mA)
                                       needs only i_out
        settle_time      : (transient only) absolute time in SECONDS to discard at the
                           start of the signal, to remove the start-up transient.
                           e.g. 200e-6 drops the first 200 us. Takes precedence over
                           settle_fraction. Default 0.0 = use the full signal.
        settle_fraction  : (transient only) alternative to settle_time, given as a fraction
                           of the signal length instead of absolute time. e.g. 0.2 drops the
                           first 20%. Only used if settle_time is 0.
                           In both cases the remaining tail is trimmed to an integer number of
                           fundamental periods (needs freq_fundamental) to avoid leakage.
                           Ignored for sim_type="hb".
        """
        if sim_type not in ("hb", "transient"):
            raise ValueError(f"sim_type must be 'hb' or 'transient', got '{sim_type}'")
        if plot_quantity not in _QUANTITY_META:
            raise ValueError(
                f"plot_quantity must be 'power', 'voltage', or 'current', got '{plot_quantity}'"
            )
        if not 0.0 <= settle_fraction < 1.0:
            raise ValueError(f"settle_fraction must be in [0, 1), got {settle_fraction}")
        if settle_time < 0.0:
            raise ValueError(f"settle_time must be >= 0, got {settle_time}")
        self.filename = filename
        self.sim_type = sim_type
        self.freq_fundamental = freq_fundamental
        self.v_out = v_out
        self.i_out = i_out
        self.plot_quantity = plot_quantity
        self.settle_time = settle_time
        self.settle_fraction = settle_fraction
        self.df = self._parse()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def spectrum_plot(self, output_path: str = "spectrum.png") -> None:
        """
        Plot spectrum as vertical lines (Cadence / ADS style).
        Left-click on any line to place a numbered marker.
        """
        meta = _QUANTITY_META[self.plot_quantity]
        col = meta["col"]
        unit = meta["unit"]

        if col not in self.df.columns:
            raise ValueError(
                f"Column '{col}' not found — check v_out/i_out names and plot_quantity."
            )

        freqs_ghz = self.df["freq_Hz"].values / 1e9
        values = self.df[col].values

        fig, ax = plt.subplots(figsize=(12, 6))

        y_min = min(values) - abs(min(values)) * 0.15
        y_max = max(values) + abs(max(values) - y_min) * 0.15
        ax.vlines(freqs_ghz, y_min, values, colors="red", linewidth=1.5)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-max(freqs_ghz) * 0.02, max(freqs_ghz) * 1.05)

        ax.set_xlabel("freq (GHz)")
        ax.set_ylabel(meta["ylabel"])

        if self.freq_fundamental:
            mode = f"f0={self.freq_fundamental/1e9:.0f} GHz"
        else:
            mode = "multi-tone" if self.sim_type == "hb" else "all FFT bins"

        signal_label = " / ".join(filter(None, [self.v_out, self.i_out]))
        ax.set_title(
            f"{meta['ylabel']} Spectrum [{self.sim_type.upper()}] — "
            f"{signal_label}  [{mode}]"
        )
        ax.grid(True, alpha=0.4, linestyle="--")

        # Interactive markers: left-click snaps to nearest frequency line
        marker_count = [0]

        def on_click(event):
            if event.inaxes != ax or event.button != 1:
                return
            idx = int(np.argmin(np.abs(freqs_ghz - event.xdata)))
            freq, val = freqs_ghz[idx], values[idx]
            marker_count[0] += 1
            ax.plot(freq, val, "o", color="red", markersize=7,
                    markerfacecolor="none", markeredgewidth=1.5)
            ax.annotate(
                f"M{marker_count[0]}: {freq:.3f} GHz  {val:.2f} {unit}",
                xy=(freq, val),
                xytext=(8, 8), textcoords="offset points",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow",
                      "edgecolor": "gray", "alpha": 0.9},
                arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8},
            )
            fig.canvas.draw()

        fig.canvas.mpl_connect("button_press_event", on_click)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.show()
        print(f"Plot saved to {output_path}")

    # ------------------------------------------------------------------
    # Private parsing infrastructure
    # ------------------------------------------------------------------

    def _parse(self) -> pd.DataFrame:
        with open(self.filename, "r") as f:
            content = f.read()

        # Clean out potential line-wrap log blocks and simulation end message
        content = re.sub(r"\\", "", content)  # Remove line-wrap continuation characters
        content = re.sub(r"End of Xyce.*", "", content, flags=re.IGNORECASE)

        tokens = content.strip().split()
        if not tokens:
            raise ValueError(f"File '{self.filename}' is empty or invalid.")

        # Reconstruct tabular layout by identifying headers vs rows
        headers = []
        data_start_idx = 0
        for i, t in enumerate(tokens):
            if t == "0" and i > 0 and tokens[i-1].lower() not in ("index", "freq", "time"):
                data_start_idx = i
                break
            headers.append(t)
        else:
            raise ValueError("Could not locate the start of numerical data rows (Index 0).")

        row_tokens = tokens[data_start_idx:]
        num_cols = len(headers)
        
        # Safe check for unexpected partial or trailing text fragments
        excess = len(row_tokens) % num_cols
        if excess:
            row_tokens = row_tokens[:-excess]

        rows_list = [row_tokens[i : i + num_cols] for i in range(0, len(row_tokens), num_cols)]
        df_raw = pd.DataFrame(rows_list, columns=headers)

        # Force numerical downcasting
        for col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

        if self.sim_type == "hb":
            return self._process_hb_prn(df_raw)
        return self._process_transient_prn(df_raw)

    def _process_hb_prn(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        if "FREQ" not in df_raw.columns:
            raise ValueError("Xyce frequency-domain .prn file must contain a 'FREQ' column.")

        freqs = df_raw["FREQ"].values

        # Find Re() and Im() split components and combine them into complex numbers
        signals = {}
        for col in df_raw.columns:
            m = re.match(r"Re\((.*)\)", col)
            if m:
                name = m.group(1)
                im_col = f"Im({name})"
                if im_col in df_raw.columns:
                    signals[name] = df_raw[col].values + 1j * df_raw[im_col].values
                else:
                    signals[name] = df_raw[col].values + 0j

        self._check_signals(signals)

        # Drop redundant negative frequencies
        freqs_pos = freqs[freqs >= 0]
        rows = []
        for freq in freqs_pos:
            idx = int(np.where(freqs == freq)[0][0])
            row = {"freq_Hz": freq, "harmonic": self._harmonic(freq)}

            for name, vals in signals.items():
                v = vals[idx]
                row[f"{name}_mag"] = abs(v)
                row[f"{name}_phase_deg"] = np.degrees(np.angle(v))

            row.update(self._quantity_row(signals, idx))
            rows.append(row)

        return pd.DataFrame(rows)

    def _process_transient_prn(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        time_col = next((c for c in df_raw.columns if c.upper() == "TIME"), None)
        if not time_col:
            raise ValueError("Xyce transient .prn file must contain a 'TIME' column.")

        times = df_raw[time_col].values
        N = len(times)
        dt = (times[-1] - times[0]) / (N - 1)

        signals = {c: df_raw[c].values for c in df_raw.columns if c.upper() not in ("INDEX", "TIME")}
        self._check_signals(signals)

        start, n_keep = self._settle_window(N, dt)
        if (start, n_keep) != (0, N):
            signals = {name: vals[start : start + n_keep] for name, vals in signals.items()}
            N = n_keep

        freqs_fft = np.fft.rfftfreq(N, d=dt)
        fft_signals = {name: np.fft.rfft(vals) / N for name, vals in signals.items()}

        if self.freq_fundamental:
            max_harmonic = int(freqs_fft[-1] / self.freq_fundamental)
            harmonic_freqs = [k * self.freq_fundamental for k in range(max_harmonic + 1)]
            keep_idx = [int(np.argmin(np.abs(freqs_fft - f))) for f in harmonic_freqs]
            keep_idx = sorted(set(keep_idx))
        else:
            keep_idx = list(range(len(freqs_fft)))

        rows = []
        for k in keep_idx:
            freq = freqs_fft[k]
            row = {"freq_Hz": freq, "harmonic": self._harmonic(freq)}

            for name, fft in fft_signals.items():
                phasor = fft[k]
                row[f"{name}_mag"] = abs(phasor)
                row[f"{name}_phase_deg"] = np.degrees(np.angle(phasor))

            row.update(self._quantity_row(fft_signals, k))
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _quantity_row(self, signals: dict, idx: int) -> dict:
        q = self.plot_quantity
        if q == "power":
            vpk = signals[self.v_out][idx]
            ipk = signals[self.i_out][idx]
            # Xyce records full amplitude profiles across positive/negative frequency lines
            p_w = 2 * abs(vpk * ipk)
            return {
                "P_out_W": p_w,
                "P_out_dBm": 10 * np.log10(max(p_w / 1e-3, 1e-30)),
            }
        if q == "voltage":
            vpk = signals[self.v_out][idx]
            return {"V_out_dBV": 20 * np.log10(max(abs(vpk), 1e-30))}
        
        ipk = signals[self.i_out][idx]
        return {"I_out_dBmA": 20 * np.log10(max(abs(ipk) / 1e-3, 1e-30))}

    def _settle_window(self, N: int, dt: float) -> tuple[int, int]:
        if self.settle_time > 0.0:
            start = round(self.settle_time / dt)
        elif self.settle_fraction > 0.0:
            start = round(N * self.settle_fraction)
        else:
            return 0, N

        if start >= N:
            raise ValueError("settle_time/settle_fraction discards the entire signal window.")
        n_keep = N - start

        if self.freq_fundamental:
            samples_per_period = (1.0 / self.freq_fundamental) / dt
            n_periods = int(n_keep / samples_per_period)
            if n_periods >= 1:
                n_keep = round(n_periods * samples_per_period)

        if n_keep < 2:
            raise ValueError("Insufficient sample window remaining to calculate FFT.")
        return start, n_keep

    def _harmonic(self, freq: float) -> int | None:
        if freq == 0:
            return 0
        if self.freq_fundamental:
            return round(freq / self.freq_fundamental)
        return None

    def _check_signals(self, signals: dict) -> None:
        needed = []
        if self.plot_quantity in ("power", "voltage") and self.v_out:
            needed.append(self.v_out)
        if self.plot_quantity in ("power", "current") and self.i_out:
            needed.append(self.i_out)

        if not needed:
            raise ValueError(
                f"plot_quantity='{self.plot_quantity}' requires target signals to be configured."
            )

        missing = [s for s in needed if s not in signals]
        if missing:
            raise ValueError(
                f"Signals not found in dataset: {missing}\n"
                f"Available signals: {list(signals)}"
            )