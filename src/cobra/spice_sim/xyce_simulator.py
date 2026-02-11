import subprocess

from cobra.spice_sim.base_simulator import BaseSimulator

class XyceSimulator(BaseSimulator):
    def __init__(self, xyce_command="Xyce", parallel=False):
        super().__init__()
        self.xyce_command = xyce_command
        self.parallel = parallel

    def run_simulation(self, netlist_name, output_name) -> str:
        parallel_command = ["mpirun", "-np", "8"] if self.parallel else []
        command = parallel_command + [self.xyce_command, netlist_name, "-o", output_name]
        
        print(f"Running Xyce simulation on {netlist_name}...")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Simulation Failed! Return code: {result.returncode}")
            print(result.stderr)
            return None
        return output_name