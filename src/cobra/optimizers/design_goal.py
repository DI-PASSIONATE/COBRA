from enum import Enum
from typing import Dict, Optional
import numpy as np
import skrf as rf
import matplotlib.pyplot as plt

class DesignGoalChecker:
    """
    DesignGoalChecker - This class checks if the design goals are met based on the current design state and the defined design goals.
    """

    def __init__(self, design_goals: list[DesignGoal], frequency_range: str | None = None):
        self.design_goals = design_goals
        self.frequency_range = frequency_range

    def check_goals(self, context) -> Dict:
        """
        Checks if the design goals are met based on the current design state.
        """
        design_state = calculate_electrical_parameters(context["simulated_network"], frequency_range=self.frequency_range)  # Use the instance's frequency range

        for goal in self.design_goals:
            parameter_value = design_state.get(goal.parameter.value)
            if parameter_value is None:
                raise ValueError(f"Design state does not contain value for parameter {goal.parameter.value}")
            
            # Check if the parameter value meets the goal range for any frequency point in the specified frequency range
            if goal.min_value is not None and goal.max_value is not None:
                if np.any((parameter_value < goal.min_value) | (parameter_value > goal.max_value)):
                    context["goal_achieved"] = False
                    return context
            elif goal.min_value is not None:
                if np.any(parameter_value < goal.min_value):
                    context["goal_achieved"] = False
                    return context
            elif goal.max_value is not None:
                if np.any(parameter_value > goal.max_value):
                    context["goal_achieved"] = False
                    return context
        context["goal_achieved"] = True
        return context


    # Override function when called print(DesignGoalChecker) to print the design goals in a readable format
    def __str__(self):
        goal_strs = []
        for goal in self.design_goals:
            goal_strs.append(f"{goal.parameter.value}: [{goal.min_value}, {goal.max_value}] in {self.frequency_range if self.frequency_range else 'full range'}")
        return "Design Goals:" + " | ".join(goal_strs)
    
    def loss(self, ntwk: rf.Network) -> list[float]:
        """
        Computes a list of penalties for each design goal based on the current design state. 
        The loss is calculated based on how much the current design state deviates from the defined design goals.
        """
        design_state = calculate_electrical_parameters(ntwk, frequency_range=self.frequency_range)  # Use the instance's frequency range

        penalties = []
        for goal in self.design_goals:
            parameter_values = design_state.get(goal.parameter.value)
            if parameter_values is None:
                raise ValueError(f"Design state does not contain value for parameter {goal.parameter.value}")
            penalties.append(goal.penalty(parameter_values))
        return penalties

    
class DesignGoal:
    """
    DesignGoal - This class represents a single design goal for the optimization process. It includes a minimum and maximum in a given frequency range
    """

    def __init__(self, parameter: DesignParameter, min_value: Optional[float] = None, max_value: Optional[float] = None):
        self.parameter = parameter
        self.min_value = min_value
        self.max_value = max_value
        self._eps = 1e-9  # Small epsilon to prevent division by zero in penalty calculation

    def penalty(self, values: np.ndarray) -> float:
        """
        Calculates the penalty based on the provided boundaries and values.
        """
        loss_val = 0.0

        # Case A: Both boundaries are provided
        if self.min_value is not None and self.max_value is not None:
            # Mask for values below min
            below_mask = values < self.min_value
            # Mask for values above max
            above_mask = values > self.max_value
            
            # Calculate normalized squared errors
            # Error = (Limit - Value) / Limit
            if np.any(below_mask):
                diff_below = (self.min_value - values[below_mask]) / (np.abs(self.min_value) + self._eps)
                loss_val += np.sum(diff_below**2)
            
            if np.any(above_mask):
                diff_above = (values[above_mask] - self.max_value) / (np.abs(self.max_value) + self._eps)
                loss_val += np.sum(diff_above**2)
            
            return loss_val

        # Case B: Only Minimum value provided
        elif self.min_value is not None:
            # Denominator for normalization
            denom = np.abs(self.min_value) + self._eps
            
            # Check for violations
            if np.any(values < self.min_value):
                # Penalize only violating values
                violating_values = values[values < self.min_value]
                # (Min - Value) / Min
                return np.sum(((self.min_value - violating_values) / denom)**2)
            else:
                # Reward (negative loss) for exceeding min
                # (Value - Min) / Min
                return -np.sum(((values - self.min_value) / denom)**2)

        # Case C: Only Maximum value provided
        elif self.max_value is not None:
            # Denominator for normalization
            denom = np.abs(self.max_value) + self._eps
            
            if np.any(values > self.max_value):
                # Penalize only violating values
                violating_values = values[values > self.max_value]
                # (Value - Max) / Max
                return np.sum(((violating_values - self.max_value) / denom)**2)
            else:
                # Reward (negative loss) for being below max
                # (Max - Value) / Max
                return -np.sum(((self.max_value - values) / denom)**2)
        raise ValueError("At least one of min_value or max_value must be provided for a DesignGoal.")

class DesignParameter(Enum):
    S11_dB = "S11_dB"
    S21_dB = "S21_dB"
    S31_dB = "S31_dB"
    S41_dB = "S41_dB"
    S12_dB = "S12_dB"
    S22_dB = "S22_dB"

def calculate_electrical_parameters(ntwk: rf.Network, frequency_range: str | None = None) -> Dict:
    """
    Calculates electrical parameters such as inductance (L), resistance (R), quality factor (Q), and coupling coefficient (k) from the S-parameters of the given network within the specified frequency range.
    """

    # 1. Single-ended to Mixed-Mode Conversion
    mm_ntwk = ntwk.copy()
    if frequency_range is not None:
        mm_ntwk = mm_ntwk[frequency_range]
    if ntwk.nports >= 4:
         mm_ntwk.se2gmm(p=2)
    # print(ntwk.frequency.unit)
    # print(ntwk.frequency.unit == "GHz")
    freq_ghz = ntwk.f / 1e9  # Convert to GHz if not already in GHz
    omega = 2 * np.pi * mm_ntwk.f  # Angular frequency in radians per second

    # 2. Extract the relevant frequency range
        # print(freq_ghz)
    # print(f"Freq Mask: {freq_mask}")

    # print(f"Freq mask applied: {frequency_range[0]} GHz to {frequency_range[1]} GHz, resulting in {len(freq_ghz)} frequency points.")

    # mm_ntwk.plot_s_db()
    # plt.show()


    # Extract Differential Z-parameters for lumped metrics
    # Index 0 = Primary Diff (d1), Index 1 = Secondary Diff (d2)
    z_d11 = mm_ntwk.z[:, 0, 0]
    z_d22 = mm_ntwk.z[:, 1, 1]
    z_d21 = mm_ntwk.z[:, 1, 0]

    # Calculate Parameters
    with np.errstate(divide="ignore", invalid="ignore"):
        Lp = np.imag(z_d11) / omega * 1e9
        Ls = np.imag(z_d22) / omega * 1e9
        Rp, Rs = np.real(z_d11), np.real(z_d22)
        Qp = np.imag(z_d11) / np.real(z_d11)
        Qs = np.imag(z_d22) / np.real(z_d22)
        k = np.abs(np.imag(z_d21) / np.sqrt(np.imag(z_d11) * np.imag(z_d22)))

    srf_idx = np.where(np.diff(np.sign(np.imag(z_d11))))[0]
    srf_f = freq_ghz[srf_idx[0]] if len(srf_idx) > 0 else None

    # Extract s-parameters
    s_param_dict = {}
    for i in range(mm_ntwk.nports):
        for j in range(mm_ntwk.nports):
            # Store s-parameters in dB for easier interpretation in the design goals
            s_param_dict[f"S{i + 1}{j + 1}_dB"] = np.array(mm_ntwk.s_db[:, i, j])
            s_param_dict[f"S{i + 1}{j + 1}"] = np.array(mm_ntwk.s[:, i, j])

    return {
        "mm_ntwk": mm_ntwk,
        "freq_ghz": freq_ghz,
        "Lp": np.array(Lp),
        "Ls": np.array(Ls),
        "Rp": np.array(Rp),
        "Rs": np.array(Rs),
        "Qp": np.array(Qp),
        "Qs": np.array(Qs),
        "k": np.array(k),
        "z_d11": np.array(z_d11),
        "srf_f": srf_f,
        **s_param_dict
    }


def plot_rfic_transformer_metrics(ntwk):
    metrics = calculate_electrical_parameters(ntwk)  # Use a wide frequency range for plotting
    mm_ntwk = metrics["mm_ntwk"]
    freq = metrics["freq_ghz"]
    Lp, Ls = metrics["Lp"], metrics["Ls"]
    Rp, Rs = metrics["Rp"], metrics["Rs"]
    Qp, Qs = metrics["Qp"], metrics["Qs"]
    k = metrics["k"]
    z_d11 = metrics["z_d11"]

    # Setup Plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(
        f"RFIC Transformer Report: {ntwk.name}", fontsize=16, fontweight="bold"
    )

    # Subplot 1: S-Parameters (S11 & S21 Mixed-Mode)
    axes[0, 0].plot(
        freq,
        mm_ntwk.s_db[:, 1, 0],
        label="Sdd21 (Insertion Loss)",
        color="teal",
        lw=2.5,
    )
    axes[0, 0].plot(
        freq,
        mm_ntwk.s_db[:, 0, 0],
        label="Sdd11 (Return Loss)",
        color="darkorange",
        ls="--",
    )
    axes[0, 0].set_title("Mixed-Mode S-Parameters", fontsize=14)
    axes[0, 0].set_ylabel("Magnitude [dB]")
    axes[0, 0].legend()

    # Subplot 2: Inductance (Lp & Ls)
    axes[0, 1].plot(freq, Lp, label="Lp (Primary)", color="blue")
    axes[0, 1].plot(freq, Ls, label="Ls (Secondary)", color="cyan")
    axes[0, 1].set_title("Inductance [nH]", fontsize=14)
    axes[0, 1].set_ylabel("L [nH]")
    axes[0, 1].legend()

    # Subplot 3: Quality Factor (Qp & Qs)
    axes[1, 0].plot(freq, Qp, label="Qp (Primary)", color="red")
    axes[1, 0].plot(freq, Qs, label="Qs (Secondary)", color="magenta")
    axes[1, 0].set_title("Quality Factor (Q)", fontsize=14)
    axes[1, 0].set_ylabel("Q")
    axes[1, 0].legend()

    # Subplot 4: Resistance (Rp & Rs)
    axes[1, 1].plot(freq, Rp, label="Rp (Primary)", color="darkgreen")
    axes[1, 1].plot(freq, Rs, label="Rs (Secondary)", color="lime")
    axes[1, 1].set_title("Loss / Resistance [Ω]", fontsize=14)
    axes[1, 1].set_ylabel("R [Ω]")
    axes[1, 1].legend()

    # Subplot 5: Coupling Coefficient (k)
    axes[2, 0].plot(freq, k, color="purple", lw=2)
    axes[2, 0].set_title("Coupling Coefficient (k)", fontsize=14)
    axes[2, 0].set_ylabel("k")
    axes[2, 0].set_ylim(0, 1.1)

    # Subplot 6: Reactance & SRF Identification
    axes[2, 1].plot(freq, np.imag(z_d11), label="Im(Zdd11)", color="brown")
    axes[2, 1].axhline(0, color="black", lw=1)  # y=0 line to find zero-crossing
    srf_f = metrics["srf_f"]
    if srf_f is not None:
        axes[2, 1].axvline(
            srf_f, color="red", linestyle=":", label=f"SRF: {srf_f:.2f} GHz"
        )
    axes[2, 1].set_title("Primary Reactance & SRF", fontsize=14)
    axes[2, 1].set_ylabel("Im(Z) [Ω]")
    axes[2, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel("Frequency [GHz]")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    dg1 = DesignGoal(parameter=DesignParameter.S21, min_value=-3, max_value=-0.5)
    dg2 = DesignGoal(parameter=DesignParameter.S11, max_value=-10)