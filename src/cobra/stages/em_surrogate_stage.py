import os
from typing import TYPE_CHECKING

import numpy as np
import skrf as rf
from onnxruntime import InferenceSession

from cobra.configuration.configuration import ConfigurationError
from cobra.stages.base_stage import COBRABaseStage
from cobra.stages.surrogate_metadata import (
    FeasibilityConstraints,
    SurrogateMetadata,
    surrogate_port_count,
)

if TYPE_CHECKING:
    from cobra.optimization_context import OptimizationContext

#: Spacing of the inference grid, in Hz.
FREQUENCY_STEP = 1e9


def inference_frequencies(frequency_range: tuple[float, float]) -> np.ndarray:
    """The frequencies (Hz) the surrogate is evaluated at: *frequency_range* in ``FREQUENCY_STEP`` steps.
    """
    low, high = frequency_range
    return np.arange(low, high + FREQUENCY_STEP / 2, FREQUENCY_STEP)


class EMSurrogateStage(COBRABaseStage):
    """
    EM Surrogate Stage - This stage performs the EM simulation using a surrogate model.
    It takes the current design state, runs the surrogate model, and updates the design state with the new EM results.
    """

    def __init__(self, em_surrogate_model: list[str], component_names: list[str] | None = None):
        self.em_surrogate_model = em_surrogate_model

        # Touchstone components store their file path; ONNX components an InferenceSession.
        self.session: list[str | InferenceSession] = []
        self.is_touchstone: list[bool] = []
        # Band each ONNX model is evaluated over; None for Touchstone components.
        self.frequency_ranges: list[tuple[float, float] | None] = []
        # Geometries each ONNX model was trained on; None for Touchstone components.
        self.constraints: list[FeasibilityConstraints | None] = []
        for model_path in em_surrogate_model:
            if str(model_path).lower().endswith((*tuple(f".s{i}p" for i in range(1, 10)), ".snp")):
                self.session.append(model_path)
                self.is_touchstone.append(True)
                self.frequency_ranges.append(None)
                self.constraints.append(None)
            else:
                session = InferenceSession(model_path)
                geometry_inputs = [node.name for node in session.get_inputs() if node.name != "frequency"]
                try:
                    metadata = SurrogateMetadata.from_session(session)
                    constraints = FeasibilityConstraints(metadata.constraints, geometry_inputs)
                except ConfigurationError as exc:
                    raise ConfigurationError(f"{model_path}: {exc}") from exc
                frequency_range = metadata.frequency_range
                if frequency_range is None:
                    raise ConfigurationError(
                        f"{model_path} declares no usable frequency range in its metadata; COBRA needs "
                        "input_parameter_ranges.frequency (0 <= min < max, in Hz, as written by ORCA) "
                        "to know the band the surrogate is valid over."
                    )
                self.session.append(session)
                self.is_touchstone.append(False)
                self.frequency_ranges.append(frequency_range)
                self.constraints.append(constraints)

        self.component_names = component_names or []

    def infeasible_components(self, model_parameters: dict) -> dict[str, list[str]]:
        """The violated ``input_constraints`` of every ONNX component, by component name.

        Empty when every component's geometry lies in the region its model was
        trained on.
        """
        infeasible = {}
        for constraints, comp_name in zip(self.constraints, self.component_names, strict=True):
            if constraints is None:
                continue
            violated = constraints.violated(self._component_parameters(model_parameters, comp_name))
            if violated:
                infeasible[comp_name] = violated
        return infeasible

    def touchstone_networks(self) -> dict[str, rf.Network]:
        """The fixed network of every Touchstone component, by component name."""
        return {
            comp_name: rf.Network(str(session), name=comp_name)
            for session, is_ts, comp_name in zip(
                self.session, self.is_touchstone, self.component_names, strict=True
            )
            if is_ts
        }

    @staticmethod
    def _component_parameters(params: dict, comp_name: str) -> dict:
        """The model inputs of *comp_name*: its ``comp:name`` entries and the unscoped ones."""
        comp_params = {}
        for k, v in params.items():
            if ":" in k:
                comp, p_name = k.split(":", 1)
                if comp == comp_name:
                    comp_params[p_name] = v
            else:
                comp_params[k] = v
        return comp_params

    def run(self, context: "OptimizationContext") -> "OptimizationContext":
        params = context.model_parameters
        results_dir = context.results_dir
        context.predicted_networks = []
        for session, is_ts, frequency_range, comp_name in zip(
            self.session, self.is_touchstone, self.frequency_ranges, self.component_names, strict=True
        ):
            if is_ts or frequency_range is None:  # both mean a Touchstone component
                ntwk = rf.Network(str(session))
            else:
                comp_params = self._component_parameters(params, comp_name)
                ntwk = self.inference_snp(session, comp_params, frequency_range)

            ntwk.name = comp_name
            context.predicted_networks.append(ntwk)

        for ntwk in context.predicted_networks:
            # e.g., predictions could be named X1.s2p, X2.s4p, etc. depending on components and ports
            num_ports = ntwk.number_of_ports
            ntwk.write_touchstone(os.path.join(results_dir, f"{ntwk.name}_predicted.s{num_ports}p"))
        return context


    def inference_snp(
        self,
        session,
        input_params: dict,
        frequency_range: tuple[float, float],
    ) -> rf.Network:
        """
        Runs inference on the model for the given geometry parameters over *frequency_range* and returns the predicted S-parameters as a network.
        """
        # Check compatability of input parameters with model input
        for param_name in input_params:
            if param_name not in [node.name for node in session.get_inputs()]:
                raise ValueError(f"Input parameter '{param_name}' is not compatible with the model input.")

        input_names = list(input_params)
        input_values = np.array([input_params[name] for name in input_names], dtype=np.float32)

        frequency_points = inference_frequencies(frequency_range)

        # Create batched input by repeating the input parameters for each frequency point and adding the frequency as an additional feature
        batched_input = np.repeat(input_values[np.newaxis, :], len(frequency_points), axis=0)

        # Build feed_dict
        feed_dict = {}

        # Process geometry parameters
        for i, param_name in enumerate(input_names):
            feed_dict[param_name] = batched_input[:, i].reshape(-1, 1).astype(np.float32)

        # Process frequency
        feed_dict["frequency"] = (frequency_points).reshape(-1, 1).astype(np.float32)

        # Run inference
        output_names = [node.name for node in session.get_outputs()]

        # Actual inference
        outputs = session.run(output_names, feed_dict)
        output_dict = dict(zip(output_names, outputs, strict=True))

        _, ntwk, _ = self.s_param_dict_to_network(output_dict, frequency_points)

        return ntwk

    def s_param_dict_to_network(
        self,
        s_param_dict: dict, frequencies: np.ndarray
    ) -> tuple[int, rf.Network, dict]:
        """
        Build a network from the model's ``S<i><j>_real``/``_imag`` outputs.

        Accepts the full N x N matrix and the upper triangle only (ORCA's
        ``UpperTriangleReImCodec``), whose missing lower triangle is filled from its
        transpose. See :func:`surrogate_port_count` for how the port count is found.
        """
        N = surrogate_port_count(s_param_dict)  # number of ports
        if N is None:
            raise ValueError(
                f"Model outputs {', '.join(s_param_dict)} are not the S<i><j>_real/_imag entries "
                "of a full or upper-triangle S-matrix."
            )

        num_freq = len(frequencies)

        # Check frequency length
        if num_freq < 1:
            raise ValueError("Frequency array must have at least one element.")

        # Initialize S-matrix of shape (nb_f, N, N)
        S = np.zeros((num_freq, N, N), dtype=np.complex64)

        # Fill S-matrix
        for i in range(N):
            for j in range(N):
                name = f"S{i + 1}{j + 1}" if f"S{i + 1}{j + 1}_real" in s_param_dict else f"S{j + 1}{i + 1}"
                real = np.array(s_param_dict[f"{name}_real"]).squeeze()
                imag = np.array(s_param_dict[f"{name}_imag"]).squeeze()

                if real.shape[0] != num_freq or imag.shape[0] != num_freq:
                    raise ValueError(
                        f"S{i + 1}{j + 1} length mismatch with frequency array."
                    )
                S[:, i, j] = real + 1j * imag  # note: frequency as first dimension

        frequencies = frequencies / 1e9  # convert to GHz for skrf compatibility

        # Create skrf Network object
        ntwk = rf.Network(frequency=frequencies, s=S, f_unit="GHz")

        merged_output = {}
        for i in range(N):
            for j in range(N):
                merged_output[f"S{i + 1}{j + 1}"] = S[:, i, j]


        return N, ntwk, merged_output
