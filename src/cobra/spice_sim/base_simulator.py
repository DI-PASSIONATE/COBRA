from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import pandas as pd
    import skrf as rf

    from cobra.spice_sim.netlist_parsers.netlist import Netlist
    from cobra.spice_sim.netlist_parsers.netlist_parser import NetlistParser
    from cobra.spice_sim.simulation_type import SimulationType, SimulationTypeMetadata


#: Reference temperature of the noise figure (IEEE), in kelvin.
T0 = 290.0
#: Boltzmann constant in J/K.
BOLTZMANN = 1.380649e-23


class SimulatorError(RuntimeError):
    """Raised when the simulator itself could not be run.
    This is a technical failure (missing or unusable executable, unreadable
    netlist file)
    """


@dataclass
class SimulationResult:
    """Unified result returned by :meth:`BaseSimulator.run_simulation`.

    Attributes:
    ----------
    output_files:
        Absolute paths to every output file produced by the simulator.
    network:
        Parsed S-parameter network when the simulation yielded Touchstone
        output (AC / S-parameter analysis).  ``None`` for other analysis
        types (HB, TRAN, DC, …).
    dataframes:
        Mapping of output file path → parsed ``pandas.DataFrame`` for
        every text-based PRN/table output file found.  Empty for purely
        Touchstone results.
    """
    output_files: list[str] = field(default_factory=list)
    network: rf.Network | None = None
    dataframes: dict[str, pd.DataFrame] = field(default_factory=dict)


class BaseSimulator(ABC):
    #: Parser for this simulator's netlist dialect; every subclass sets one.
    netlist_parser: ClassVar[NetlistParser]
    #: Analyses this simulator can run; a design goal needing another one is rejected.
    supported_simulation_types: ClassVar[frozenset[SimulationType]]
    #: Name of the setting holding the simulator executable, e.g. ``"xyce_command"``.
    command_setting: ClassVar[str]
    #: Whether analyses take named arguments beyond the slots of their metadata (VACASK).
    named_analysis_parameters: ClassVar[bool] = False

    @classmethod
    def get_simulation_metadata(cls, sim_type: SimulationType) -> SimulationTypeMetadata:
        """Return simulator-specific metadata for *sim_type*.

        Parameter names, descriptions, defaults, and ``.options`` category
        information come from the simulator's netlist parser, since they
        describe the arguments of its analysis directives.
        """
        return cls.netlist_parser.analysis_metadata(sim_type)

    @abstractmethod
    def prepare_netlist(
        self,
        netlist: Netlist,
        sim_type: SimulationType,
        sim_params: dict[str, str],
        goal_band: tuple[float, float] | None = None,
    ) -> Netlist:
        """Return *netlist* ready to run a *sim_type* analysis.

        Returns *netlist* itself when it already declares that analysis;
        otherwise a new netlist with the directive (from *sim_params*, falling
        back to the metadata defaults) and the output requests the simulator
        needs to produce a result for it.  *goal_band* is the frequency span,
        in Hz, the design goals on this analysis evaluate; a simulator may use
        it for the sweep of an analysis it adds.
        """

    @abstractmethod
    def run_simulation(self, netlist_path: str, netlist: Netlist) -> SimulationResult | None:
        """Run the simulator on the file *netlist_path*, whose parsed content is *netlist*,
        and return a :class:`SimulationResult`.

        Returns ``None`` if the simulation failed (non-zero exit code or no
        output files found).

        Raises:
            SimulatorError: The simulator could not be run/found at all.
        """

    @abstractmethod
    def preprocess_ntwk(self, ntwk, name: str) -> str:
        """
        Preprocess the network by performing some operations (e.g., vector fitting)
        that the simulator requires before running the simulation.

        Args:
            ntwk: The network object containing the S-parameters and frequency information.
            name (str): The name of the network for identification purposes.

        Returns:
            A file path to the preprocessed network data (e.g., a SPICE subcircuit file) that can be included in the netlist for circuit simulation.
        """

    def equivalent_RCL(self, Z, f):
        # returns RLC values for a specific impedance
        # returns large inductor instead of negative cap values, and cap becomes zero
        w = 2 * math.pi * f

        # Parallel form
        Y = 1 / Z
        G = Y.real
        B = Y.imag

        Rpar = round(1 / G,2) if G != 0 else None

        if B > 0:
            Cpar_fF = round((B/w)*1e15,2)
            Cpar = Cpar_fF/1e15
            Lpar = 1e3 # verryy large to have no impact for Rf-freq
        elif B < 0:
            Lpar_pH = round(-1 / (w * B)*1e12,2)
            Lpar = Lpar_pH/1e12
            Cpar = 0 # keine Cap da
        else:
            Lpar = 1e3 # verryy large to have no impact for Rf-freq
            Cpar = 0 # keine Cap da

        return (Rpar, Cpar, Lpar)
