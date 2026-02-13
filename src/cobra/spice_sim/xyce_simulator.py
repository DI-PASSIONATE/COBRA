import subprocess

from cobra.spice_sim.base_simulator import BaseSimulator
from cobra.spice_sim.vector_fit import vector_fit
import skrf as rf
import glob, os

class XyceSimulator(BaseSimulator):
    def __init__(self, xyce_command="Xyce", parallel=False):
        super().__init__()
        self.xyce_command = xyce_command
        self.parallel = parallel

    def preprocess_ntwk(self, ntwk):
        # Preprocess the network by vector fitting the S-parameters to create a compact model that can be included in the netlist for circuit simulation.
        return vector_fit(ntwk, name="cobra_output")

    def run_simulation(self, netlist_name) -> rf.Network:
        output_name = "xyce_output"
        parallel_command = ["mpirun", "-np", "8"] if self.parallel else []
        command = parallel_command + [self.xyce_command, netlist_name, "-o", output_name]
        
        print(f"Running Xyce simulation on {netlist_name}...")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Simulation Failed! Return code: {result.returncode}")
            print(result.stderr)
            return None
        
        # Xyce outputs a file with the extension .s*p, we need to find the exact filename
        output_files = glob.glob(f"{output_name}.s*p")
        if not output_files:
            print("No output file found!")
            return None
        output_file = output_files[0]  # Assuming there's only one output file

        # Load the output file using scikit-rf and return the network object
        ntwk = rf.Network(output_file)
        return ntwk