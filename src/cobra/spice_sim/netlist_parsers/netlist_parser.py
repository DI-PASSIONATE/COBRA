from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path
import math
import skrf as rf
    
@dataclass
class NetlistElement:
    name: str
    etype: str                 # R,C,L,V,I,D,Q,M,X,P,Y,MODEL,...
    subtype: Optional[str]     # for Y-devices: second token (e.g. YLIN_Trafo1); else None
    line_index: int
    raw_line: str
    inline_comment: str
    tokens: List[str]
    nodes: List[str]
    value: Optional[str]
    model: Optional[str]
    params: Dict[str, str]
        

@dataclass
class BaseNetlistParser(ABC): 
    def from_lines(self, lines: List[str]) -> "BaseNetlistParser":
        self._lines = lines[:]
        self._elements: Dict[str, NetlistElement] = {}  # includes MODEL entries too
        self.parse_netlist()
        return self

    def from_file(self, path: Union[str, Path], encoding: str = "utf-8") -> "BaseNetlistParser":
        text = Path(path).read_text(encoding=encoding, errors="replace")
        self.netlist_path = path
        return self.from_lines(text.splitlines(keepends=True))

    def to_string(self) -> str:
        return "".join(self._lines)
    
    def save(self, path: Union[str, Path], encoding: str = "utf-8") -> None:
        Path(path).write_text(self.to_string(), encoding=encoding)

    def list_elements(self, types: Optional[List[str]] = None) -> List[NetlistElement]:
        elems = sorted(self._elements.values(), key=lambda e: e.line_index)
        if types:
            want = {t.upper() for t in types}
            elems = [e for e in elems if e.etype in want]
        return elems

    def get_element(self, name: str) -> NetlistElement:
        name = name.strip()
        if name not in self._elements:
            raise KeyError(f"Element '{name}' not found.")
        return self._elements[name]
    
    @abstractmethod
    def parse_netlist(self) -> None:
        """
        Parse the netlist file and extract design parameters (e.g., C, L, R values).
        Args:
            netlist_file (str): The path to the netlist file to be parsed.
        Returns:
            dict: A dictionary containing the extracted design parameters.
        """
        pass

    @abstractmethod
    def update_parameters(self, parameters: Dict[str, float]) -> None:
        """
        Update the netlist with new design parameters.
        Args:
            parameters (dict): A dictionary containing the design parameters to be updated, where keys are parameter names and values are the new parameter values.
        Returns:
            None
        """
        pass