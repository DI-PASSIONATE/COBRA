import os

import numpy as np
import skrf as rf
from onnxruntime import InferenceSession

from cobra.stages.base_stage import COBRABaseStage


class EMSurrogateStage(COBRABaseStage):
    """
    EM Surrogate Stage - This stage performs the EM simulation using a surrogate model.
    It takes the current design state, runs the surrogate model, and updates the design state with the new EM results.
    """

    def __init__(self, em_surrogate_model: list[str], component_names: list[str] = None):
        self.em_surrogate_model = em_surrogate_model
        
        self.session = []
        self.is_touchstone = []
        for model_path in em_surrogate_model:
            if str(model_path).lower().endswith(tuple(f".s{i}p" for i in range(1, 10)) + (".snp",)):
                self.session.append(model_path)
                self.is_touchstone.append(True)
            else:
                self.session.append(InferenceSession(model_path))
                self.is_touchstone.append(False)
                
        self.component_names = component_names or []

    def run(self, context: dict) -> dict:
        params = context["model_parameters"]
        results_dir = context.get("results_dir", ".")
        context["predicted_networks"] = []
        for session, is_ts, comp_name in zip(self.session, self.is_touchstone, self.component_names):
            if is_ts:
                ntwk = rf.Network(session)
            else:
                comp_params = {}
                for k, v in params.items():
                    if ":" in k:
                        comp, p_name = k.split(":", 1)
                        if comp == comp_name:
                            comp_params[p_name] = v
                    else:
                        comp_params[k] = v
                ntwk = self.inference_snp(session, comp_params)
            
            ntwk.name = comp_name
            context["predicted_networks"].append(ntwk)
            
        for ntwk in context["predicted_networks"]:
            # e.g., predictions could be named X1.s2p, X2.s4p, etc. depending on components and ports
            num_ports = ntwk.number_of_ports
            ntwk.write_touchstone(os.path.join(results_dir, f"{ntwk.name}_predicted.s{num_ports}p"))
        return context


    def inference_snp(self, session, input_params: dict) -> rf.Network:
        """
        Runs inference on the model for the given geometry parameters and frequency points, and saves the predicted S-parameters to a Touchstone file.
        """
        # Check compatability of input parameters with model input
        for param_name in input_params:
            if param_name not in [node.name for node in session.get_inputs()]:
                raise ValueError(f"Input parameter '{param_name}' is not compatible with the model input.")
            
        input_names = [name for name in input_params]
        input_values = np.array([input_params[name] for name in input_names], dtype=np.float32)

        # Create frequency points from 1 GHz to 200 GHz in 1 GHz steps
        frequency_points = np.arange(1e9, 201e9, 1e9)

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
        output_dict = dict(zip(output_names, outputs))

        N, ntwk, output_dict = self.s_param_dict_to_network(output_dict, frequency_points)

        return ntwk
    
    def s_param_dict_to_network(
        self,
        s_param_dict: dict, frequencies: np.ndarray
    ) -> tuple[int, rf.Network, dict]:
        N = int(np.sqrt(len(s_param_dict) // 2))  # number of ports

        num_freq = len(frequencies)

        # Check frequency length
        if num_freq < 1:
            raise ValueError("Frequency array must have at least one element.")

        # Initialize S-matrix of shape (nb_f, N, N)
        S = np.zeros((num_freq, N, N), dtype=np.complex64)

        # Fill S-matrix
        for i in range(N):
            for j in range(N):
                real = np.array(s_param_dict[f"S{i + 1}{j + 1}_real"]).squeeze()
                imag = np.array(s_param_dict[f"S{i + 1}{j + 1}_imag"]).squeeze()

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