from typing import Dict, List, Optional, Tuple
import re

from cobra.spice_sim.netlist_parsers.netlist_parser import (
    BaseNetlistParser,
    Component,
    Include,
    NetlistElement,
)


class XyceNetlistParser(BaseNetlistParser):
    """
    Xyce-flavoured SPICE netlist parser.

    Responsibilities beyond plain reading/writing:
    - Rewrites Qucs-S ``TSTONEFILE`` transformer blocks (``YLIN`` instances)
      into ordinary ``X`` subcircuit calls so the rest of COBRA can treat them
      as standard surrogate components.
    """

    # -------------------------------------------------------------------------
    # Compiled regular expressions (class-level, shared by all instances)
    # -------------------------------------------------------------------------

    # Qucs-S / Xyce use ";" or "$" to start an inline comment.
    _inline_comment_markers = (";", "$")

    # Lines that are pure comments or continuation lines — skipped during parsing.
    _comment_re      = re.compile(r"^\s*\*",                          re.IGNORECASE)
    _continuation_re = re.compile(r"^\s*\+",                          re.IGNORECASE)

    # The first token on a device/instance line gives us its name.
    _inst_re         = re.compile(r"^\s*([A-Za-z]\w*)")

    # key=value parameter tokens.
    _kv_re           = re.compile(r"^([A-Za-z_]\w*)\s*=\s*(.+)$",    re.IGNORECASE)

    # .MODEL directives — we look for LIN + TSTONEFILE to detect surrogates.
    _model_re        = re.compile(r"^\s*\.model\s+(\S+)\s+(\S+)\s*(.*)$", re.IGNORECASE)

    # .SUBCKT / .ENDS markers for tracking subcircuit nesting depth.
    _subckt_re       = re.compile(r"^\s*\.subckt\b",                  re.IGNORECASE)
    _ends_re         = re.compile(r"^\s*\.ends\b",                    re.IGNORECASE)

    # TSTONEFILE=... on a LIN model marks a Qucs-S surrogate placeholder.
    _tstonefile_re   = re.compile(r"\bTSTONEFILE\s*=\s*([^\s]+)",    re.IGNORECASE)

    # .INCLUDE directives that reference fitted-subcircuit files.
    _include_re      = re.compile(
        r"^\s*\.include\s+[\"']?([^\s\"']+)[\"']?\s*$",               re.IGNORECASE
    )

    # -------------------------------------------------------------------------
    # Constructor
    # -------------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()
        # Maps rewritten X-instance names → original TSTONEFILE paths so the
        # GUI can display them even after the Y→X conversion.
        self._tstonefile_map: Dict[str, str] = {}

    # -------------------------------------------------------------------------
    # Public mutators
    # -------------------------------------------------------------------------

    def set_value(self, name: str, new_value: str) -> None:
        """Replace the positional value token of an R, C, L, V, or I element."""
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
            raise ValueError(
                f"set_value is for R/C/L and simple V/I. "
                f"Use set_param or set_model for '{e.etype}'."
            )

        self._replace_line(e.line_index, tokens, e.inline_comment, e.raw_line.endswith("\n"))
        self.parse_netlist()

    def set_model(self, name: str, new_model: str) -> None:
        """Replace the model-reference token on a device or subcircuit instance line."""
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
            # Subcircuit name sits just before the first key=value param (or at end).
            first_kv = next(
                (i for i in range(1, len(tokens)) if self._kv_re.match(tokens[i])),
                None,
            )
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
        """Update or append a ``key=value`` parameter on any element line."""
        e = self.get_element(name)
        tokens = e.tokens[:]
        key = key.strip()

        # .MODEL lines have two positional tokens before parameters start.
        start_idx = 3 if e.etype == "MODEL" else 1

        for i in range(start_idx, len(tokens)):
            m = self._kv_re.match(tokens[i])
            if m and m.group(1).lower() == key.lower():
                tokens[i] = f"{key}={value}"
                break
        else:
            tokens.append(f"{key}={value}")

        self._replace_line(e.line_index, tokens, e.inline_comment, e.raw_line.endswith("\n"))
        self.parse_netlist()

    def update_parameters(self, parameters: Dict[str, float]) -> None:
        """
        Bulk-update multiple parameters by name.

        Parameter names follow two conventions:

        * ``"ElementName"`` — updates the positional value of an R/C/L/V/I
          element via :meth:`set_value`.
        * ``"InstanceName:ParamKey"`` — updates a key=value parameter on an
          instance via :meth:`set_param`.
        """
        for name, value in parameters.items():
            if ":" in name:
                instance_name, param_key = name.split(":", 1)
                if instance_name in self._elements:
                    self.set_param(instance_name, param_key, str(value))
                else:
                    print(f"Warning: Instance '{instance_name}' not found in netlist elements.")
            elif name in self._elements:
                self.set_value(name, str(value))
            else:
                print(f"Warning: Parameter '{name}' not found in netlist elements.")

    # -------------------------------------------------------------------------
    # Parsing entry point
    # -------------------------------------------------------------------------

    def parse_netlist(self) -> None:
        """
        Parse ``self._lines`` and populate ``_elements``, ``_components``, and
        ``_includes``.

        Qucs-S TSTONEFILE placeholders are normalised to plain X-subcircuits
        before the main pass runs.
        """
        self._normalize_tstonefile_subcircuits()
        self._reset_state()

        subckt_depth = 0  # > 0 while inside a .SUBCKT definition block

        for idx, raw in enumerate(self._lines):
            if not raw.strip():
                continue
            if self._comment_re.match(raw) or self._continuation_re.match(raw):
                continue

            code, inline_comment = self._split_inline_comment(raw)
            stripped = code.strip()
            if not stripped:
                continue

            # ------------------------------------------------------------------
            # Track .SUBCKT / .ENDS nesting so inner devices are not parsed as
            # top-level components.
            # ------------------------------------------------------------------
            if self._subckt_re.match(stripped):
                tokens_sc = stripped.split()
                if len(tokens_sc) >= 2:
                    self._inline_subckt_names.add(tokens_sc[1])
                subckt_depth += 1
                continue

            if self._ends_re.match(stripped):
                if subckt_depth > 0:
                    subckt_depth -= 1
                continue

            if subckt_depth > 0:
                continue

            # ------------------------------------------------------------------
            # .INCLUDE directives
            # ------------------------------------------------------------------
            m_include = self._include_re.match(stripped)
            if m_include:
                self._includes.append(Include(file_path=m_include.group(1), line_index=idx))
                continue

            # ------------------------------------------------------------------
            # .MODEL directives — kept in the element table for later editing
            # but not treated as simulation components.
            # ------------------------------------------------------------------
            m_model = self._model_re.match(stripped)
            if m_model:
                model_name = m_model.group(1)
                model_type = m_model.group(2)
                tokens = stripped.split()
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
                    params=self._collect_params(tokens[3:]),
                )
                continue

            # Other dot-directives are preserved in the text but not parsed.
            if stripped.startswith("."):
                continue

            # ------------------------------------------------------------------
            # Device and instance lines
            # ------------------------------------------------------------------
            if not self._inst_re.match(stripped):
                continue

            tokens = stripped.split()
            if not tokens:
                continue

            name  = tokens[0]
            etype = name[0].upper()
            nodes, value, model, params, subtype = self._parse_instance(tokens, etype)

            # Carry the TSTONEFILE path into the params dict so the GUI can find it.
            if etype == "X" and name in self._tstonefile_map:
                params["TSTONEFILE"] = self._tstonefile_map[name]

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

            # X instances whose model is *not* defined inline are external
            # surrogates; add them to _components for the model-selector.
            if etype == "X" and model not in self._inline_subckt_names:
                self._components[name] = Component(
                    name=name,
                    nodes=nodes,
                    model=model if model else "",
                    params=params,
                )

    # -------------------------------------------------------------------------
    # Private: Qucs-S TSTONEFILE → X-subcircuit normalisation
    # -------------------------------------------------------------------------

    def _normalize_tstonefile_subcircuits(self) -> None:
        """
        Convert Qucs-S ``YLIN``/``TSTONEFILE`` device blocks into
        Xyce-compatible subcircuit instances (``X...``) in-place on
        ``self._lines``.

        The transformation is two-pass:

        1. Collect every ``.MODEL <name> LIN TSTONEFILE=...`` entry.
        2. Rewrite the matching ``Y`` instance into an ``X`` call and replace
           the model line with a ``.INCLUDE`` for the vector-fitted ``.sp``
           file.
        """
        self._tstonefile_map.clear()

        # --- Pass 1: collect LIN models that carry a TSTONEFILE reference ----
        model_line_index: Dict[str, int] = {}
        model_tstonefile: Dict[str, str] = {}

        for idx, raw in enumerate(self._lines):
            code, _ = self._split_inline_comment(raw)
            stripped = code.strip()
            if not stripped or stripped.startswith("*"):
                continue

            m = self._model_re.match(stripped)
            if not m or m.group(2).upper() != "LIN":
                continue

            tstone_match = self._tstonefile_re.search(stripped)
            if not tstone_match:
                continue

            model_name = m.group(1)
            model_line_index[model_name] = idx
            model_tstonefile[model_name] = tstone_match.group(1)

        if not model_line_index:
            return

        # --- Pass 2: rewrite Y-instances and replace matching model lines ----
        updated_lines = self._lines[:]
        converted_models: Dict[str, str] = {}

        for idx, raw in enumerate(self._lines):
            code, inline_comment = self._split_inline_comment(raw)
            stripped = code.strip()

            if not stripped or stripped.startswith(".") or stripped.startswith("*"):
                continue

            tokens = stripped.split()
            if len(tokens) < 4 or tokens[0][0].upper() != "Y":
                continue

            model_name = tokens[-1]
            mdl_idx    = model_line_index.get(model_name)
            if mdl_idx is None:
                continue

            if model_name in converted_models:
                x_name = converted_models[model_name]
            else:
                # Prefix the original instance name with "X" to produce a
                # standard subcircuit call that Xyce understands.
                x_name = f"X{tokens[1]}"
                converted_models[model_name] = x_name

                # Replace the placeholder .MODEL line with the .INCLUDE for
                # the surrogate .sp file generated by vector fitting.
                include_line = f'.INCLUDE "{x_name}.sp"'
                if inline_comment:
                    include_line += " " + inline_comment.lstrip()
                if raw.endswith("\n"):
                    include_line += "\n"
                updated_lines[mdl_idx] = include_line

            # Qucs-S emits each port as <signal_node> 0; drop the literal zeros
            # because the fitted subcircuit only expects the signal nodes.
            port_nodes      = [n for n in tokens[2:-1] if n != "0"]
            instance_tokens = [x_name] + port_nodes + [f"{x_name}_subct"]

            tfile = model_tstonefile.get(model_name)
            if tfile:
                self._tstonefile_map[x_name] = tfile

            updated_lines[idx] = self._format_line(instance_tokens, inline_comment, raw.endswith("\n"))

        self._lines = updated_lines

    # -------------------------------------------------------------------------
    # Private: instance line parsing
    # -------------------------------------------------------------------------

    def _parse_instance(
        self, tokens: List[str], etype: str
    ) -> Tuple[List[str], Optional[str], Optional[str], Dict[str, str], Optional[str]]:
        """
        Decode a device/instance token list into
        ``(nodes, value, model, params, subtype)``.
        Each device type has its own positional layout.
        """
        nodes:   List[str]       = []
        params:  Dict[str, str]  = {}
        value:   Optional[str]   = None
        model:   Optional[str]   = None
        subtype: Optional[str]   = None

        if etype in ("R", "C", "L"):
            # <name> <n+> <n-> <value> [params...]
            if len(tokens) >= 4:
                nodes  = tokens[1:3]
                value  = tokens[3]
                params = self._collect_params(tokens[4:])

        elif etype == "M":
            # <name> <drain> <gate> <source> <bulk> <model> [params...]
            if len(tokens) >= 6:
                nodes  = tokens[1:5]
                model  = tokens[5]
                params = self._collect_params(tokens[6:])

        elif etype == "Q":
            # <name> <collector> <base> <emitter> <model> [params...]
            if len(tokens) >= 5:
                nodes  = tokens[1:4]
                model  = tokens[4]
                params = self._collect_params(tokens[5:])

        elif etype == "D":
            # <name> <anode> <cathode> <model> [params...]
            if len(tokens) >= 4:
                nodes  = tokens[1:3]
                model  = tokens[3]
                params = self._collect_params(tokens[4:])

        elif etype == "X":
            # <name> [nodes...] <subckt_name> [key=value...]
            # The subcircuit name is the last positional token before any key=value.
            first_kv = next(
                (i for i in range(1, len(tokens)) if self._kv_re.match(tokens[i])),
                None,
            )
            end = first_kv if first_kv is not None else len(tokens)
            if end >= 3:
                model  = tokens[end - 1]
                nodes  = tokens[1 : end - 1]
                params = self._collect_params(tokens[first_kv:]) if first_kv is not None else {}

        elif etype == "P":
            # <name> <n1> <n2> [params...]
            if len(tokens) >= 3:
                nodes  = tokens[1:3]
                params = self._collect_params(tokens[3:])

        elif etype in ("V", "I"):
            # <name> <n+> <n-> [value or waveform...]
            if len(tokens) >= 3:
                nodes = tokens[1:3]
                value = " ".join(tokens[3:]) if len(tokens) > 3 else None

        elif etype == "Y":
            # Qucs-S format: <name> <subtype> [port_nodes...] <model>
            # Only seen before _normalize_tstonefile_subcircuits runs.
            if len(tokens) >= 4:
                subtype = tokens[1]
                model   = tokens[-1]
                nodes   = tokens[2:-1]

        else:
            # Fallback for unrecognised types.
            if len(tokens) >= 3:
                nodes = tokens[1:3]
            params = self._collect_params(tokens[3:])
            value  = tokens[-1] if len(tokens) >= 4 else None

        return nodes, value, model, params, subtype

    def _collect_params(self, toks: List[str]) -> Dict[str, str]:
        """Parse a list of tokens and return only the ``key=value`` pairs as a dict."""
        result: Dict[str, str] = {}
        for tok in toks or []:
            m = self._kv_re.match(tok)
            if m:
                result[m.group(1)] = m.group(2)
        return result

    # -------------------------------------------------------------------------
    # Private: line-level text helpers
    # -------------------------------------------------------------------------

    def _split_inline_comment(self, raw_line: str) -> Tuple[str, str]:
        """
        Split *raw_line* into ``(code, comment)``.

        The split point is the earliest occurrence of any inline-comment marker.
        The original trailing newline (if any) is preserved on the code part.
        """
        s       = raw_line.rstrip("\n")
        newline = "\n" if raw_line.endswith("\n") else ""

        best_pos = min(
            (s.find(m) for m in self._inline_comment_markers if s.find(m) != -1),
            default=None,
        )

        if best_pos is None:
            return s + newline, ""

        code    = s[:best_pos].rstrip()
        comment = s[best_pos:].rstrip()
        return code + newline, comment

    def _replace_line(
        self, line_index: int, tokens: List[str], inline_comment: str, had_newline: bool
    ) -> None:
        """Overwrite ``self._lines[line_index]`` with a rebuilt line."""
        self._lines[line_index] = self._format_line(tokens, inline_comment, had_newline)

    def _format_line(
        self, tokens: List[str], inline_comment: str, had_newline: bool
    ) -> str:
        """
        Assemble *tokens* back into a single text line, re-attaching any inline
        comment and restoring the trailing newline if the original had one.
        """
        rebuilt = " ".join(tokens)
        if inline_comment:
            rebuilt += " " + inline_comment.lstrip()
        if had_newline and not rebuilt.endswith("\n"):
            rebuilt += "\n"
        return rebuilt