from typing import Dict, List, Optional, Tuple
import re
from cobra.spice_sim.netlist_parsers.netlist_parser import BaseNetlistParser, NetlistElement


class XyceNetlistParser(BaseNetlistParser):
    """
    XyceNetlistParser - An implementation of the BaseNetlistParser using Xyce for parsing netlist files.
    """
    _inline_comment_markers = (";", "$")

    _comment_re = re.compile(r"^\s*\*", re.IGNORECASE)
    _continuation_re = re.compile(r"^\s*\+", re.IGNORECASE)

    _inst_re = re.compile(r"^\s*([A-Za-z]\w*)")
    _kv_re = re.compile(r"^([A-Za-z_]\w*)\s*=\s*(.+)$", re.IGNORECASE)

    # .MODEL name TYPE key=val ...
    _model_re = re.compile(r"^\s*\.model\s+(\S+)\s+(\S+)\s*(.*)$", re.IGNORECASE)
    
    def set_value(self, name: str, new_value: str) -> None:
        e = self.get_element(name)
        tokens = e.tokens[:]

        if e.etype in ("R", "C", "L"):
            if len(tokens) < 4:
                raise ValueError("R/C/L line too short.")
            tokens[3] = new_value

        elif e.etype in ("V", "I"):
            if len(tokens) < 3:
                raise ValueError("V/I line too short.")
            tokens = tokens[:3] + [new_value]

        else:
            raise ValueError(f"set_value is intended for R/C/L (and simple V/I). Use set_param or set_model for {e.etype}.")

        self._replace_line(e.line_index, tokens, e.inline_comment, e.raw_line.endswith("\n"))
        self.parse_netlist()

    def set_model(self, name: str, new_model: str) -> None:
        e = self.get_element(name)
        tokens = e.tokens[:]

        if e.etype == "D":
            if len(tokens) < 4:
                raise ValueError("D line too short.")
            tokens[3] = new_model

        elif e.etype == "M":
            if len(tokens) < 6:
                raise ValueError("M line too short.")
            tokens[5] = new_model

        elif e.etype == "Q":
            if len(tokens) < 5:
                raise ValueError("Q line too short.")
            tokens[4] = new_model

        elif e.etype == "X":
            first_kv = None
            for i in range(1, len(tokens)):
                if self._kv_re.match(tokens[i]):
                    first_kv = i
                    break
            end = first_kv if first_kv is not None else len(tokens)
            if end < 2:
                raise ValueError("X line too short.")
            tokens[end - 1] = new_model

        elif e.etype == "Y":
            if len(tokens) < 3:
                raise ValueError("Y line too short.")
            tokens[-1] = new_model

        else:
            raise ValueError(f"set_model not supported for type '{e.etype}'")

        self._replace_line(e.line_index, tokens, e.inline_comment, e.raw_line.endswith("\n"))
        self.parse_netlist()

    def set_param(self, name: str, key: str, value: str) -> None:
        e = self.get_element(name)
        tokens = e.tokens[:]
        key = key.strip()

        start_idx = 1
        if e.etype == "MODEL":
            start_idx = 3

        found = False
        for i in range(start_idx, len(tokens)):
            m = self._kv_re.match(tokens[i])
            if m and m.group(1).lower() == key.lower():
                tokens[i] = f"{key}={value}"
                found = True
                break

        if not found:
            tokens.append(f"{key}={value}")

        self._replace_line(e.line_index, tokens, e.inline_comment, e.raw_line.endswith("\n"))
        self.parse_netlist()

    # -----------------------------
    # Parsing helpers
    # -----------------------------

    def parse_netlist(self) -> None:
        self._elements.clear()

        for idx, raw in enumerate(self._lines):
            if not raw.strip():
                continue
            if self._comment_re.match(raw) or self._continuation_re.match(raw):
                continue

            code, inline_comment = self._split_inline_comment(raw)
            stripped = code.strip()
            if not stripped:
                continue

            # .MODEL entries into list
            m_model = self._model_re.match(stripped)
            if m_model:
                model_name = m_model.group(1)
                model_type = m_model.group(2)
                tokens = stripped.split()
                params = self._collect_params(tokens[3:])

                self._elements[model_name] = NetlistElement(
                    name=model_name,
                    etype="MODEL",
                    subtype=None,
                    line_index=idx,
                    raw_line=raw,
                    inline_comment=inline_comment,
                    tokens=tokens,
                    nodes=[],
                    value=None,
                    model=model_type,
                    params=params,
                )
                continue

            # other directives ignored
            if stripped.startswith("."):
                continue

            # instance
            m = self._inst_re.match(stripped)
            if not m:
                continue

            tokens = stripped.split()
            if not tokens:
                continue

            name = tokens[0]
            etype = name[0].upper()

            nodes, value, model, params, subtype = self._parse_instance(tokens, etype)

            self._elements[name] = NetlistElement(
                name=name,
                etype=etype,
                subtype=subtype,
                line_index=idx,
                raw_line=raw,
                inline_comment=inline_comment,
                tokens=tokens,
                nodes=nodes,
                value=value,
                model=model,
                params=params,
            )

    def _parse_instance(self, tokens: List[str], etype: str) -> Tuple[List[str], Optional[str], Optional[str], Dict[str, str], Optional[str]]:
        nodes: List[str] = []
        params: Dict[str, str] = {}
        value: Optional[str] = None
        model: Optional[str] = None
        subtype: Optional[str] = None

        if etype in ("R", "C", "L"):
            if len(tokens) >= 4:
                nodes = tokens[1:3]
                value = tokens[3]
                params = self._collect_params(tokens[4:])
            return nodes, value, model, params, subtype

        if etype == "M":
            if len(tokens) >= 6:
                nodes = tokens[1:5]
                model = tokens[5]
                params = self._collect_params(tokens[6:])
            return nodes, value, model, params, subtype

        if etype == "Q":
            if len(tokens) >= 5:
                nodes = tokens[1:4]
                model = tokens[4]
                params = self._collect_params(tokens[5:])
            return nodes, value, model, params, subtype

        if etype == "D":
            if len(tokens) >= 4:
                nodes = tokens[1:3]
                model = tokens[3]
                params = self._collect_params(tokens[4:])
            return nodes, value, model, params, subtype

        if etype == "X":
            first_kv = None
            for i in range(1, len(tokens)):
                if self._kv_re.match(tokens[i]):
                    first_kv = i
                    break
            end = first_kv if first_kv is not None else len(tokens)
            if end >= 3:
                model = tokens[end - 1]
                nodes = tokens[1:end - 1]
                params = self._collect_params(tokens[first_kv:]) if first_kv is not None else {}
            return nodes, value, model, params, subtype

        if etype == "P":
            if len(tokens) >= 3:
                nodes = tokens[1:3]
                params = self._collect_params(tokens[3:])
            return nodes, value, model, params, subtype

        if etype in ("V", "I"):
            if len(tokens) >= 3:
                nodes = tokens[1:3]
                value = " ".join(tokens[3:]) if len(tokens) > 3 else None
            return nodes, value, model, params, subtype

        if etype == "Y":
            if len(tokens) >= 4:
                subtype = tokens[1]
                model = tokens[-1]
                nodes = tokens[2:-1]
            return nodes, value, model, params, subtype

        if len(tokens) >= 3:
            nodes = tokens[1:3]
        params = self._collect_params(tokens[3:])
        value = tokens[-1] if len(tokens) >= 4 else None
        return nodes, value, model, params, subtype

    def _collect_params(self, toks: List[str]) -> Dict[str, str]:
        d: Dict[str, str] = {}
        for t in toks or []:
            m = self._kv_re.match(t)
            if m:
                d[m.group(1)] = m.group(2)
        return d

    def _split_inline_comment(self, raw_line: str) -> Tuple[str, str]:
        s = raw_line.rstrip("\n")
        newline = "\n" if raw_line.endswith("\n") else ""

        best_pos = None
        for m in self._inline_comment_markers:
            pos = s.find(m)
            if pos != -1 and (best_pos is None or pos < best_pos):
                best_pos = pos

        if best_pos is None:
            return s + newline, ""

        code = s[:best_pos].rstrip()
        comment = s[best_pos:].rstrip()
        return code + newline, comment

    def _replace_line(self, line_index: int, tokens: List[str], inline_comment: str, had_newline: bool) -> None:
        rebuilt = " ".join(tokens)
        if inline_comment:
            rebuilt += " " + inline_comment.lstrip()
        if had_newline and not rebuilt.endswith("\n"):
            rebuilt += "\n"
        self._lines[line_index] = rebuilt

    def update_parameters(self, parameters: Dict[str, float]) -> None:
        for name, value in parameters.items():
            if name in self._elements:
                self.set_value(name, str(value))
            else:
                print(f"Warning: Parameter '{name}' not found in netlist elements.")