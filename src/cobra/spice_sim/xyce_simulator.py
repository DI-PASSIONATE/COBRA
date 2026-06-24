import subprocess

from cobra.setting import CobraSetting
from cobra.spice_sim.base_simulator import BaseSimulator
from cobra.spice_sim.netlist_parsers.netlist_parser import BaseNetlistParser
from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser
from cobra.spice_sim.vector_fit import vector_fit
import skrf as rf
import glob, os

class XyceSimulator(BaseSimulator):
    netlist_parser: BaseNetlistParser = XyceNetlistParser()

    _settings = [
        CobraSetting(
            name="xyce_command",
            dtype=str,
            default="Xyce",
            description=(
                "Command used to invoke the Xyce simulator.\n"
                "Can be a plain command name (e.g. 'Xyce') if it is on PATH,\n"
                "or an absolute path to the Xyce executable."
            ),
        ),
        CobraSetting(
            name="parallel_xyce",
            dtype=bool,
            default=False,
            description=(
                "Run Xyce in parallel using MPI (mpirun -np 8).\n"
                "Requires an MPI-enabled Xyce build and mpirun on PATH."
            ),
        ),
        CobraSetting(
            name="enforce_passivity",
            dtype=bool,
            default=False,
            description=(
                "Enforce passivity on the vector-fitted surrogate model.\n"
                "Increases pre-processing time but prevents non-physical\n"
                "active behaviour in the circuit simulator."
            ),
        ),
    ]

    def __init__(self, xyce_command: str = "Xyce", parallel_xyce: bool = False, enforce_passivity: bool = False):
        self.xyce_command = xyce_command
        self.parallel = parallel_xyce
        self.enforce_passivity = enforce_passivity

    def preprocess_ntwk(self, ntwk, name="cobra_output"):
        # Preprocess the network by vector fitting the S-parameters to create a compact model that can be included in the netlist for circuit simulation.
        return vector_fit(ntwk, name=name, enforce_passivity=self.enforce_passivity)

    def run_simulation(self, netlist_name: str) -> rf.Network:
        results_dir = os.path.dirname(netlist_name)
        output_name = "xyce_output"
        parallel_command = ["mpirun", "-np", "8"] if self.parallel else []
        command = parallel_command + [self.xyce_command, os.path.basename(netlist_name), "-o", output_name]
        
        result = subprocess.run(command, capture_output=True, text=True, cwd=results_dir)
        
        if result.returncode != 0:
            print(f"Simulation Failed! Return code: {result.returncode}")
            print(result.stderr)
            return None
        
        # Xyce outputs a file with the extension .s*p, we need to find the exact filename
        output_files = glob.glob(os.path.join(results_dir, f"{output_name}.s*p"))
        if not output_files:
            print("No output file found!")
            return None
        output_file = output_files[0]  # Assuming there's only one output file

        # Load the output file using scikit-rf and return the network object
        ntwk = rf.Network(output_file)
        return ntwk