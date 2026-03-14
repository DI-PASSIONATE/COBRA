from enum import Enum
from typing import Dict, Optional
import numpy as np
import skrf as rf
import matplotlib.pyplot as plt


class DesignParameter(Enum):
    S11_dB = "S11_dB"
    S21_dB = "S21_dB"
    S31_dB = "S31_dB"
    S41_dB = "S41_dB"
    S12_dB = "S12_dB"
    S22_dB = "S22_dB"
    Lp = "Lp"
    Ls = "Ls"
    Rp = "Rp"
    Rs = "Rs"
    Qp = "Qp"
    Qs = "Qs"
    k = "k"
    SRF = "SRF"
    COMING_SOON = "CUSTOM"  # Placeholder for future custom metrics that may be added

class DesignGoal:
    """
    DesignGoal - This class represents a single design goal for the optimization process. It includes a minimum and maximum in a given frequency range
    """

    def __init__(self, parameter: DesignParameter, frequency_range: Optional[str] = None, min_value: Optional[float] = None, max_value: Optional[float] = None, weight: float = 1.0):
        self.parameter = parameter
        self.frequency_range = frequency_range
        self.min_value = min_value
        self.max_value = max_value
        self.weight = weight
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

class DesignGoalChecker:
    """
    DesignGoalChecker - This class checks if the design goals are met based on the current design state and the defined design goals.
    """

    def __init__(self, design_goals: list[DesignGoal]):
        self.design_goals: list[DesignGoal] = design_goals

    def check_goals(self, context) -> Dict:
        """
        Checks if the design goals are met based on the current design state.
        """
        ntwk = context["simulated_network"]
        params, penalties = self.loss(ntwk)  # Calculate penalties based on the current design state
        
        # Check if any penalty is greater than zero, which indicates a violation of at least one design goal
        context["goal_achieved"] = all(penalty <= 0.0 for penalty in penalties)
        context["electrical_parameters"] = params  # Store calculated electrical parameters in context for later use (e.g., optimization feedback)
        context["penalties"] = penalties  # Store penalties in context for later use (e.g., optimization feedback)
        return context


    # Override function when called print(DesignGoalChecker) to print the design goals in a readable format
    def __str__(self):
        goal_strs = []
        for goal in self.design_goals:
            goal_strs.append(f"{goal.parameter.value}: [{goal.min_value}, {goal.max_value}] in {goal.frequency_range if goal.frequency_range else 'full range'} (weight: {goal.weight})")
        return "Design Goals:" + " | ".join(goal_strs)
    
    def loss(self, ntwk: rf.Network) -> tuple[Dict, list[float]]:
        """
        Computes a list of penalties for each design goal based on the current design state. 
        The loss is calculated based on how much the current design state deviates from the defined design goals.
        
        Args:
            ntwk: The skrf Network object
        Returns:
            A tuple containing the design state (a dictionary of calculated parameters) and a list of penalties
        """
        
        penalties = []
        design_state = {}

        for goal in self.design_goals:
            parameter_values = self.calculate_parameter(ntwk, goal.parameter, goal.frequency_range)
            design_state[goal.parameter.value] = parameter_values
            if parameter_values is None:
                raise ValueError(f"Design state does not contain value for parameter {goal.parameter.value}")
            penalties.append(goal.penalty(parameter_values) * goal.weight)
        return design_state, penalties

    def calculate_parameter(self, ntwk: rf.Network, parameter: DesignParameter, frequency_range: str | None = None, custom_func_coming_soon=None) -> np.ndarray:
        """
        Calculates a specific electrical parameter from the S-parameters of the given network within the specified frequency range.
        """
        if frequency_range is not None:
            try:
                ntwk = ntwk[frequency_range]
            except ValueError:
                # skrf might fail with "could not convert string to float: ''" for malformed ranges like "10-"
                # Or invalid frequency strings. We re-raise with context.
                raise ValueError(f"Invalid frequency range format: '{frequency_range}'. Expected format e.g. '10-20ghz'.")
                
        if parameter.value.startswith("S") and parameter.value.endswith("_dB"):
            # Extract port indices from parameter name, e.g., S21_dB -> i=2, j=1
            i, j = int(parameter.value[1]), int(parameter.value[2])
            return ntwk.s_db[:, i-1, j-1]  # Convert to 0-based index
        else:
            # For lumped parameters, we need to calculate them from the Z-parameters
            mm_ntwk = ntwk.copy()
            if mm_ntwk.nports >= 4:
                mm_ntwk.se2gmm(p=2)

            z_d11 = mm_ntwk.z[:, 0, 0]
            z_d22 = mm_ntwk.z[:, 1, 1]
            z_d21 = mm_ntwk.z[:, 1, 0]
            freq_ghz = ntwk.f / 1e9  # Convert to GHz
            omega = 2 * np.pi * mm_ntwk.f  # Angular frequency in radians

            if parameter == DesignParameter.Lp:
                return np.imag(z_d11) / omega * 1e9
            elif parameter == DesignParameter.Ls:
                return np.imag(z_d22) / omega * 1e9
            elif parameter == DesignParameter.Rp:
                return np.real(z_d11)
            elif parameter == DesignParameter.Rs:
                return np.real(z_d22)
            elif parameter == DesignParameter.Qp:
                return np.imag(z_d11) / np.real(z_d11)
            elif parameter == DesignParameter.Qs:
                return np.imag(z_d22) / np.real(z_d22)
            elif parameter == DesignParameter.k:
                return np.abs(np.imag(z_d21) / np.sqrt(np.imag(z_d11) * np.imag(z_d22)))
            elif parameter == DesignParameter.SRF:
                srf_idx = np.where(np.diff(np.sign(np.imag(z_d11))))[0]
                return freq_ghz[srf_idx[0]] if len(srf_idx) > 0 else None
            elif parameter == DesignParameter.COMING_SOON:
                return custom_func_coming_soon(ntwk)
            else:
                raise ValueError(f"Unsupported design parameter: {parameter.value}")