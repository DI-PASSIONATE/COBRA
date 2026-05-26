from typing import Dict, List, Optional, Tuple
import re
from cobra.spice_sim.netlist_parsers.netlist_parser import BaseNetlistParser, NetlistElement, Component, Include


class XyceNetlistParser(BaseNetlistParser):
    """
    XyceNetlistParser - An implementation of the BaseNetlistParser using Xyce for parsing netlist files.

    This parser has one extra responsibility beyond a normal netlist reader:
    it rewrites Qucs-S TSTONEFILE transformer blocks into plain X-subcircuits
    so the rest of COBRA can treat them like normal surrogate components.
    """
    # Qucs-S and Xyce use different syntaxes for comments, inline comments, and
    # instance names. These regexes let us recognize the pieces we need to keep
    # or rewrite while preserving the rest of the file.
    _inline_comment_markers = (";", "$")

    _comment_re = re.compile(r"^\s*\*", re.IGNORECASE)
    _continuation_re = re.compile(r"^\s*\+", re.IGNORECASE)

    _inst_re = re.compile(r"^\s*([A-Za-z]\w*)")
    _kv_re = re.compile(r"^([A-Za-z_]\w*)\s*=\s*(.+)$", re.IGNORECASE)

    # Match ordinary .MODEL lines so we can detect the special LIN + TSTONEFILE
    # models that need to become surrogate subcircuits.
    _model_re = re.compile(r"^\s*\.model\s+(\S+)\s+(\S+)\s*(.*)$", re.IGNORECASE)
    # Qucs-S uses TSTONEFILE on LIN models to point at Touchstone data instead
    # of an actual SPICE subcircuit, so these entries must be rewritten.
    _tstonefile_re = re.compile(r"\bTSTONEFILE\s*=\s*([^\s]+)", re.IGNORECASE)
    # After vector fitting, COBRA emits a generated .sp file, and the converted
    # netlist needs a normal SPICE include that Xyce can read.
    _include_re = re.compile(r"^\s*\.include\s+[\"']?([^\s\"']+)[\"']?\s*$", re.IGNORECASE)

    def _format_line(self, tokens: List[str], inline_comment: str, had_newline: bool) -> str:
        # Rebuild a rewritten netlist line from tokenized pieces while preserving
        # any trailing comment and whether the original line ended with a newline.
        rebuilt = " ".join(tokens)
        if inline_comment:
            rebuilt += " " + inline_comment.lstrip()
        if had_newline and not rebuilt.endswith("\n"):
            rebuilt += "\n"
        return rebuilt

    def _normalize_tstonefile_subcircuits(self) -> None:
        """Convert Qucs-S TSTONEFILE Y-devices into Xyce-compatible subcircuit instances."""
        # First pass: collect the .MODEL lines that are only placeholders for
        # Touchstone data. We need this mapping before rewriting the instance
        # lines because the model directive can appear before or after the Y line.
        model_line_by_name: Dict[str, int] = {}

        for idx, raw in enumerate(self._lines):
            code, _ = self._split_inline_comment(raw)
            stripped = code.strip()
            if not stripped or stripped.startswith("*"):
                continue

            m_model = self._model_re.match(stripped)
            if not m_model:
                continue

            model_name = m_model.group(1)
            model_type = m_model.group(2)
            if model_type.upper() != "LIN":
                continue
            if not self._tstonefile_re.search(stripped):
                continue

            model_line_by_name[model_name] = idx

        if not model_line_by_name:
            return

        # Second pass: rewrite each matching YLIN instance into a normal X
        # subcircuit call and replace the matching model line with .INCLUDE.
        updated_lines = self._lines[:]
        converted_models: Dict[str, str] = {}

        for idx, raw in enumerate(self._lines):
            code, inline_comment = self._split_inline_comment(raw)
            stripped = code.strip()
            if not stripped or stripped.startswith(".") or stripped.startswith("*"):
                continue

            tokens = stripped.split()
            if len(tokens) < 4:
                continue

            if tokens[0][:1].upper() != "Y":
                continue

            model_name = tokens[-1]
            model_line_index = model_line_by_name.get(model_name)
            if model_line_index is None:
                continue

            if model_name in converted_models:
                x_name = converted_models[model_name]
            else:
                # Keep the original instance name, but prefix it with X so the
                # result looks like a standard subcircuit call.
                original_instance_name = tokens[1]
                x_name = f"X{original_instance_name}"
                converted_models[model_name] = x_name

                # The TSTONEFILE model line becomes the include for the fitted
                # surrogate generated later by the vector-fitting stage.
                include_line = f'.INCLUDE "{x_name}.sp"'
                if inline_comment:
                    include_line += " " + inline_comment.lstrip()
                if raw.endswith("\n"):
                    include_line += "\n"
                updated_lines[model_line_index] = include_line

            # Qucs-S writes each port as a signal node followed by a literal 0
            # reference. The fitted SPICE subcircuit expects only the signal
            # nodes, so we drop the zeros here.
            port_nodes = [node for node in tokens[2:-1] if node != "0"]
            instance_tokens = [x_name] + port_nodes + [f"{x_name}_subct"]
            updated_lines[idx] = self._format_line(instance_tokens, inline_comment, raw.endswith("\n"))

        self._lines = updated_lines
    
    def set_value(self, name: str, new_value: str) -> None:
        # Update a scalar element value in place, then reparse so the cached
        # element/component tables stay consistent with the modified text.
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
        # Rewrite the model name for device lines whose last token is the model
        # reference, then rebuild the parser state.
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
        # Update or append a key=value pair on a model or instance line. This is
        # used for parameters that are not simple positional values.
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
        # Normalize Qucs-S transformer placeholders before the generic parser
        # runs so the rest of COBRA only sees X-subcircuits and include lines.
        self._normalize_tstonefile_subcircuits()
        self._elements.clear()
        self._components.clear()
        self._includes.clear()

        for idx, raw in enumerate(self._lines):
            if not raw.strip():
                continue
            if self._comment_re.match(raw) or self._continuation_re.match(raw):
                continue

            code, inline_comment = self._split_inline_comment(raw)
            stripped = code.strip()
            if not stripped:
                continue

            # Record .INCLUDE lines so the rest of the application can inspect
            # which fitted subcircuit files are referenced by the netlist.
            m_include = self._include_re.match(stripped)
            if m_include:
                include_file = m_include.group(1)
                self._includes.append(Include(file_path=include_file, line_index=idx))
                continue

            # Preserve ordinary .MODEL statements in the element table for later
            # editing, but they are not treated as simulation components.
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

            # Other dot-directives are not needed for component extraction, so
            # they are left in the text but skipped during parsing.
            if stripped.startswith("."):
                continue

            # Everything else is treated as a device or instance line.
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

            # Only X instances are considered surrogate-capable components in the
            # current COBRA workflow.
            if etype == "X":
                component = Component(
                    name=name,
                    nodes=nodes,
                    model=model if model else "",
                    params=params,
                )
                self._components[name] = component

    def _parse_instance(self, tokens: List[str], etype: str) -> Tuple[List[str], Optional[str], Optional[str], Dict[str, str], Optional[str]]:
        # Start with the generic empty shape and fill in only the fields each
        # device type actually uses.
        nodes: List[str] = []
        params: Dict[str, str] = {}
        value: Optional[str] = None
        model: Optional[str] = None
        subtype: Optional[str] = None

        # Passive two-terminal elements encode their value positionally.
        if etype in ("R", "C", "L"):
            if len(tokens) >= 4:
                nodes = tokens[1:3]
                value = tokens[3]
                params = self._collect_params(tokens[4:])
            return nodes, value, model, params, subtype

        # MOSFETs carry four nodes and then a model name.
        if etype == "M":
            if len(tokens) >= 6:
                nodes = tokens[1:5]
                model = tokens[5]
                params = self._collect_params(tokens[6:])
            return nodes, value, model, params, subtype

        # BJTs use three nodes plus a model name.
        if etype == "Q":
            if len(tokens) >= 5:
                nodes = tokens[1:4]
                model = tokens[4]
                params = self._collect_params(tokens[5:])
            return nodes, value, model, params, subtype

        # Diodes are a simpler model-backed device with two nodes.
        if etype == "D":
            if len(tokens) >= 4:
                nodes = tokens[1:3]
                model = tokens[3]
                params = self._collect_params(tokens[4:])
            return nodes, value, model, params, subtype

        # X devices are the form we want after surrogate conversion: nodes first,
        # then the subcircuit name, then optional key=value parameters.
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

        # Generic ports use two nodes and then optional parameters.
        if etype == "P":
            if len(tokens) >= 3:
                nodes = tokens[1:3]
                params = self._collect_params(tokens[3:])
            return nodes, value, model, params, subtype

        # Voltage and current sources keep their expression or waveform tail as
        # a single value field.
        if etype in ("V", "I"):
            if len(tokens) >= 3:
                nodes = tokens[1:3]
                value = " ".join(tokens[3:]) if len(tokens) > 3 else None
            return nodes, value, model, params, subtype

        # The original Qucs-S YLIN syntax stores a subtype token and then a model
        # name at the end. We keep that structure only long enough to normalize
        # it into an X subcircuit earlier in parse_netlist.
        if etype == "Y":
            if len(tokens) >= 4:
                subtype = tokens[1]
                model = tokens[-1]
                nodes = tokens[2:-1]
            return nodes, value, model, params, subtype

        # Fallback for any remaining line shapes: capture the common node/value
        # layout so the caller can still inspect the line later.
        if len(tokens) >= 3:
            nodes = tokens[1:3]
        params = self._collect_params(tokens[3:])
        value = tokens[-1] if len(tokens) >= 4 else None
        return nodes, value, model, params, subtype

    def _collect_params(self, toks: List[str]) -> Dict[str, str]:
        # Pull key=value tokens into a dictionary so parameter updates can be
        # done without manually reparsing the whole line later.
        d: Dict[str, str] = {}
        for t in toks or []:
            m = self._kv_re.match(t)
            if m:
                d[m.group(1)] = m.group(2)
        return d

    def _split_inline_comment(self, raw_line: str) -> Tuple[str, str]:
        # Split a raw line into code and trailing inline comment while keeping
        # the original newline behavior intact.
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
        # Low-level text rewrite helper used by the public mutators above.
        rebuilt = " ".join(tokens)
        if inline_comment:
            rebuilt += " " + inline_comment.lstrip()
        if had_newline and not rebuilt.endswith("\n"):
            rebuilt += "\n"
        self._lines[line_index] = rebuilt

    def update_parameters(self, parameters: Dict[str, float]) -> None:
        # Bulk-update multiple named parameters by reusing the single-element
        # setter, which keeps the parser state synchronized after each change.
        for name, value in parameters.items():
            if name in self._elements:
                self.set_value(name, str(value))
            else:
                print(f"Warning: Parameter '{name}' not found in netlist elements.")