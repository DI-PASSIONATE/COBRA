import subprocess

from cobra.spice_sim.base_simulator import BaseSimulator
from cobra.spice_sim.netlist_parsers.netlist_parser import BaseNetlistParser
from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser
from cobra.spice_sim.vector_fit import vector_fit
import skrf as rf
import glob, os

class XyceSimulator(BaseSimulator):
    netlist_parser: BaseNetlistParser = XyceNetlistParser()

    def __init__(self, xyce_command:str ="Xyce", parallel_xyce:bool=False):
        self.xyce_command = xyce_command
        self.parallel = parallel_xyce
        #self.netlist_parser.from_file("TODO")

    def preprocess_ntwk(self, ntwk):
        # Preprocess the network by vector fitting the S-parameters to create a compact model that can be included in the netlist for circuit simulation.
        return vector_fit(ntwk, name="cobra_output")

    def run_simulation(self, netlist_name: str) -> rf.Network:
        output_name = "xyce_output"
        parallel_command = ["mpirun", "-np", "8"] if self.parallel else []
        command = parallel_command + [self.xyce_command, netlist_name, "-o", output_name]
        
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