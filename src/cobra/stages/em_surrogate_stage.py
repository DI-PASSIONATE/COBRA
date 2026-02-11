from typing import Dict
from onnxruntime import InferenceSession
import numpy as np
import skrf as rf
import matplotlib.pyplot as plt
from cobra.stages.base_stage import COBRABaseStage


class EMSurrogateStage(COBRABaseStage):
    """
    EM Surrogate Stage - This stage performs the EM simulation using a surrogate model.
    It takes the current design state, runs the surrogate model, and updates the design state with the new EM results.
    """

    def __init__(self, em_surrogate_model):
        self.em_surrogate_model = em_surrogate_model
        self.session = InferenceSession(em_surrogate_model)

    def run(self, context: Dict) -> Dict:
        params = context["parameters"]
        ntwk = self.inference_snp(params)
        ntwk.plot_s_db()

        # Comment this out later - just for debugging
        plt.show()
        return context


    def inference_snp(self, input_params: dict) -> rf.Network:
        """
        Runs inference on the model for the given geometry parameters and frequency points, and saves the predicted S-parameters to a Touchstone file.
        """
        # Check compatability of input parameters with model input
        for param_name in input_params.keys():
            if param_name not in [node.name for node in self.session.get_inputs()]:
                raise ValueError(f"Input parameter '{param_name}' is not compatible with the model input.")
            
        input_names = [name for name in input_params.keys()]
        input_values = np.array([input_params[name] for name in input_names], dtype=np.float32)

        # Create frequency points from 1 GHz to 200 GHz in 1 GHz steps
        frequency_points = np.arange(0, 201e9, 1e9)

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
        output_names = [node.name for node in self.session.get_outputs()]

        # Actual inference
        outputs = self.session.run(output_names, feed_dict)
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

        # Create skrf Network object
        ntwk = rf.Network(frequency=frequencies, s=S, f_unit="Hz")

        merged_output = {}
        for i in range(N):
            for j in range(N):
                merged_output[f"S{i + 1}{j + 1}"] = S[:, i, j]

        return N, ntwk, merged_output