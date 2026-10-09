---
title: VACASK Simulator – Native VACASK Netlists in COBRA
description: >-
  Use the VACASK circuit simulator as COBRA's simulation backend: netlist conventions, ports, surrogate blocks, analysis parameters, result files, and noise-figure goals.
---

# VACASK Simulator

[VACASK](https://codeberg.org/arpadbuermen/VACASK) is an alternative to Xyce.
COBRA reads **native VACASK netlists** (there is no Xyce-to-VACASK translation),
vector-fits the surrogate models into a VACASK subcircuit, runs `vacask`, and
reads its raw files back into the same results the Xyce backend produces. S-parameter,
harmonic-balance and transient goals work as documented elsewhere; VACASK adds
noise-figure goals.

## Install and Select

Install VACASK from its repository and make `vacask` available on `PATH`, or
give its full path in `vacask_command`. `cobra doctor` lists `vacask` as an
optional executable. Select the backend in the configuration:

```json
"simulator": {
  "name": "VacaskSimulator",
  "settings": {
    "vacask_command": "vacask",
    "vacask_threads": 1,
    "vector_fit_max_order": 12,
    "enforce_passivity": false
  }
}
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `vacask_command` | `"vacask"` | Command name on `PATH`, or an absolute path |
| `vacask_threads` | `1` | Threads for VACASK's device evaluation and linear solver (`vacask -n N`); `0` lets VACASK pick (honours `OMP_NUM_THREADS`). Keep `1` when `parallel_trials` is above 1 |
| `vector_fit_max_order` | `12` | Highest model order snp2le may use for a surrogate (at least 2); see [Surrogate file](#surrogate-file) |
| `enforce_passivity` | `false` | Enforce passivity on the vector-fitted surrogate; slower pre-processing, no non-physical gain |

COBRA runs `vacask -sp -qp -n N deck.sim`, so the `postprocess` steps of the
`control` block are skipped. Use `--simulator VacaskSimulator` with
`cobra init` and `cobra parse` when the input is a netlist (see
[Command Line](../user-guide/cli.md)).

## A Minimal Netlist

`examples/netlists/VACASK/trafo_acsp.sim` is a transformer with one surrogate
block. Its configuration is `examples/configs/vacask_trafo_acsp.json`.

```text
Transformer with a surrogate block        // title line, ignored by VACASK
ground 0

load "resistor.osdi"                      // devices need load + model lines
load "capacitor.osdi"
model r resistor
model c capacitor
model vsource vsource                     // not predefined without builtins.inc

include "X1.inc"                          // written by COBRA (vector fit)
X1 (Port1 Port2 Port3 Port4 _net0 _net1) s_equivalent

C1 (Port2 Port1) c c=7.94f
C3 (0 Port4) c c=10.0f

// Differential ports: a 0 V source in series with the reference resistor.
vp1 (a1 Port2) vsource dc=0 mag=2.72029 type="sine" ampl=2.72029 freq=130G
rp1 (a1 Port1) r r=925
vp2 (a2 Port4) vsource dc=0 mag=0.894427 type="sine" ampl=0.894427 freq=130G
rp2 (a2 Port3) r r=100

control
  analysis sp1 acsp ports=["vp1", "rp1", "vp2", "rp2"] from=100G to=170G mode="lin" points=500
endc
```

(Abridged; the file also has the inductors and the second capacitor.)

## Two Netlist Styles

Two ways of writing a VACASK netlist work. Both are native VACASK; they differ
in where the ports and the surrogate blocks come from. Common rules:

- The first line is a title that VACASK ignores. Names are case-sensitive.
- Every master needs a `model` line, builtins included (`model vsource vsource`);
  Verilog-A devices also need their module loaded (`load "resistor.osdi"`).
- Analyses, `options` and `save` requests sit in the `control ... endc` block.

### Style 1: `acsp` ports and placeholder instances

The style of the COBRA and Qucs-S examples above: the ports are listed in an
`acsp` analysis and each surrogate is a placeholder instance.

#### Ports

Port *k* is the *k*-th (vsource, resistor) pair of an `acsp` analysis,
`ports=["vp1", "rp1", "vp2", "rp2"]`:

- the resistor `r` is the port impedance `z0`;
- the vsource drives the port: `mag` is the AC amplitude, and
  `type="sine" ampl=... freq=...` the harmonic-balance drive;
- differential ports work, as in the example above.

Without an `acsp` list, ports are detected instead (see Style 2). A deck that
needs S-parameters must contain an `acsp` analysis.

#### Surrogate instances

An instance whose master is defined neither in the deck nor in an included file
is a surrogate component. The deck includes the vector fit the way a Xyce deck
writes `.INCLUDE "X1.sp"`:

```text
include "X1.inc"
X1 (Port1 Port2 Port3 Port4 _net0 _net1) s_equivalent
```

`s_equivalent` is any placeholder. COBRA writes `X1.inc` into the results
directory (subcircuit `X1_subct`, see [Surrogate file](#surrogate-file)) and
renames the master to `X1_subct`.

COBRA scans included files so PDK devices are not mistaken for surrogates. It
looks next to the deck, in the working directory, in `SIM_INCLUDE_PATH`, and with
the `include_path_prefix`/`include_path_suffix` of `.vacaskrc.toml`. SPICE files
included with `lang=ngspice` (or `xyce`, ...) are scanned too, and `.model X`
counts as `m_X`. Foreign includes need a Cadnip-enabled VACASK build:

```text
include "/path/to/pdk/models.lib" lang=ngspice section=tt
```

### Style 2: xschem netlists

xschem writes Spectre-like netlists in which passive blocks are often snp2le
lumped-element (LE) models, defined in `.inc` files and instantiated inside
other subcircuits (for example `ISM_le_new` or `Balun_le`), and which have no
`acsp` analysis. COBRA handles them with three features.

**Subcircuit surrogates.** A `component_models` key may name a *subcircuit*
(defined in the netlist or an included file) instead of an instance. COBRA
renames every instance of that subcircuit, at any depth, to `<name>_subct` and
adds `include "<name>.inc"` (Xyce: `.INCLUDE "<name>.sp"`), where the simulator
writes the fitted surrogate. The subcircuit's pins, in order, are the ports of
the surrogate's S-parameters, so the model must have that many ports. The
original definition stays included but unused.

```text
"simulator": {"name": "VacaskSimulator"},
"component_models": {"ISM_le_new": "ism.onnx"}
```

**Detected ports.** If the deck has no `acsp ports=[...]` list, every top-level
vsource from a ground node (declared with `ground`, default `0`) to a node
shared with exactly one resistor is a port, numbered in netlist order. xschem's
`Vin (net7 GND) vsource ...` with `Rin (RF_in net7) resistor r=50` is port 1;
supply and bias sources, whose node touches more elements, are not ports. An
`acsp` list takes precedence.

**Port probes.** A port is a probe point named after its source. The HB and
transient tables get `V(<src>) = R*I(<src>)` and `I(V<src>) = I(<src>)`, so no
0 V probe source is needed. `Power_dBm[Vout]` is the power into the port
resistor (`2 R |I|^2` with COBRA's half-amplitude phasors), and a gain goal
works between two ports:

```json
{"parameter": "Gain_dB[Vin@Vout]", "frequency_range": "130ghz", "min_value": 10.0,
 "max_value": null, "weight": 1.0, "kind": "gain_db", "node": "Vout", "port": "Vin",
 "source_amplitude": 0.0316, "impedance": 50.0}
```

`probe_nodes` lists the nodes saved with `v()` and `i()` first, then the port
sources. A 3 dB attenuator surrogate gives `Gain_dB` = -3.0 dB end to end.
Instance-style surrogates (an undefined master such as `s_equivalent`) still work
in this style.

Checked with real xschem LNA netlists (one netlist per analysis, `ground GND`,
`save default`): they parse and the ports `Vin` and `Vout` are detected.

### Surrogate file

VACASK surrogates are written by
[snp2le](https://github.com/iic-jku/snp2le) (Apache-2.0, JKU Linz), a required
dependency. COBRA runs its universal mode (vector fit, resistors `noisy=0`) and
makes the file self-contained. It begins with

```text
ground GND
load "capacitor.osdi"
load "inductor.osdi"
load "resistor.osdi"

subckt X1_subct (...)
  model resistor resistor
  ...
```

`ground GND` is snp2le's ground name; VACASK accepts several, so the file also
works in decks with `ground 0`. The `load` lines and local `model` bindings
mean the netlist's own `model` lines (for example a PDK resistor) do not apply
to the surrogate. Include it at the top level. The setting `vector_fit_max_order`
(default 12) is the highest model order snp2le may use: snp2le's GUI default of 6
is too coarse, but at very high orders (e.g. 24 on a simple 2-port) poles land
far out of band and VACASK solves the subcircuit inaccurately (|S| error about
0.3 against the model).

Broadband models need more. The octagonal-inductor model of the LNA preset spans
0 to 500 GHz: at order 12 its fit is off by up to 0.17 in |S| at 110 to 170 GHz,
at order 40 by 0.02, and orders 40 and 60 give the same circuit results, so
`examples/configs/vacask_lna_inductor.json` sets 40. Check a new model by
re-simulating one design at two orders.

A fit that does not converge (scikit-rf's `auto_fit` can cycle at the order cap)
stops after 1000 pole relocations; COBRA logs a warning and penalises that trial.

### PDK models (IHP SG13G2)

Use the PDK converted for VACASK rather than a foreign include of its ngspice
models. The ngspice files do not load as they are: `sg13g2_hbt_mod.lib` is
Latin-1 (VACASK reads UTF-8 only), the HBT cards carry ngspice-only parameters
such as `vbe_max`, and the resistors use an `r3_cmc` model Cadnip does not map.
VACASK's `sg13g2tovc` script converts the PDK once:

```bash
PDK_ROOT=/path/to/IHP-Open-PDK PDK=ihp-sg13g2 python3 /path/to/vacask/lib/vacask/python/sg13g2tovc.py
```

It writes `libs.tech/vacask/models`, the compiled models in `libs.tech/vacask/osdi`
and a `.vacaskrc.toml`. If it stops on a missing `sg13g2_io.spice`, the PDK names
the file `sg13g2_io.spi`; copy it to the expected name. The netlist then
includes the converted libraries by name:

```text
include "sg13g2_vacask_common.lib"
include "cornerHBT.lib" section=hbt_typ
xq1 (c b e sub) npn13G2 nx=8.0
```

Point VACASK to them with `SIM_INCLUDE_PATH` and `SIM_MODULE_PATH` (the latter
replaces the default module path, so list VACASK's own `lib/vacask/mod` too), or
with the PDK's `.vacaskrc.toml` copied to `~/.vacaskrc.toml`. A missing module
path shows up as `File 'mosvar.osdi' not found`.

!!! warning "Write `nx` as a real number"
    The IHP HBT computes series resistances from `(4/nx)`. VACASK evaluates an
    integer division for an integer `nx`, so `nx=8` zeroes them and the gain
    jumps (by about 9 dB in the LNA example). Write `nx=8.0`; values COBRA's
    optimizer writes are always real numbers.

`examples/netlists/VACASK/lna_trafo_hb.sim` (configuration
`examples/configs/vacask_lna_trafo_hb.json`) is the LNA example ported this way.
Against the Xyce version with the same surrogate, its 130 GHz gain is 2.6 dB
against Xyce's 3.4 dB; the difference comes from the PDK's VBIC 1.15 (Xyce) and
VBIC 1.3 (VACASK) models, not from COBRA.

!!! note
    Relative includes resolve from the copy of the deck in
    `results/<timestamp>_<name>/`, as with Xyce. A `.vacaskrc.toml` next to the
    original deck is not next to that copy: use `~/.vacaskrc.toml`,
    `SIM_INCLUDE_PATH` or absolute paths.

### Probe nodes

For HB and transient goals, a node is a probe node when it is saved with both
voltage and current, `save v(n) i(vn)`, or when a 0 V vsource named `v<node>` or
`V<node>` has the node as its first terminal (the Qucs-S convention). Detected
ports also serve as probes under their source's name (see Style 2).

## Analyses and `simulation_parameters`

| VACASK analysis | COBRA | Config key |
| --- | --- | --- |
| `acsp` | AC | `"AC"` |
| `hb` | HB | `"HB"` |
| `tran` | TRAN | `"TRAN"` |
| `noise` | NOISE | `"NOISE"` |
| `hbnoise` | HBNOISE | `"HBNOISE"` |

Other analyses (`op`, `ac`, ...) are ignored. For each analysis type the goals
need, COBRA switches off the other analyses (commenting them out with
`//cobra-off `) or injects `analysis cobra_<type> ...` when the deck has none.
Parameters use VACASK's names:

| Key | Parameters |
| --- | --- |
| `AC` | `mode`, `points`, `from`, `to` |
| `NOISE` | `mode`, `points`, `from`, `to`, `in`, `out` |
| `HBNOISE` | as `NOISE`, plus `freq`, `nharm`, `inspur`, `outspur` |
| `HB` | `freq` (space-separated tones, e.g. `"95G 10G"`), `nharm`, ... |
| `TRAN` | `step`, `stop`, `start`, `maxstep` |

```json
"simulator": {"name": "VacaskSimulator"},
"simulation_parameters": {
  "AC": {"points": "500", "from": "1G", "to": "200G"},
  "NOISE": {"out": "out", "from": "1G", "to": "10G", "points": "50"}
}
```

`points` with `mode="lin"` counts intervals: `points=500` gives 501
frequencies. `in` defaults to the first port's vsource; `out` is required when
COBRA injects a noise analysis.

When COBRA adds a `noise` or `hbnoise` analysis and `simulation_parameters` set
neither `from` nor `to`, the sweep spans the frequency ranges of the goals on it
(`NF`, or `NF_SSB`/`NF_DSB`, whose frequencies are the output offset). A single
goal frequency gives a one-point sweep. `cobra parse` warns when a noise goal lies
outside the sweep the netlist or `simulation_parameters` set.

## Result Files

Written next to the netlist copy in the results directory:

| Analysis | Result |
| --- | --- |
| `acsp` | A `skrf.Network` (per-port `z0`), also `<netlist>.sNp`; Touchstone 2.0 when the port impedances differ |
| `hb` | `<netlist>.HB.FD.csv`: `FREQ`, `Re/Im(V(n))`, `Re/Im(I(inst))` |
| `tran` | `<netlist>.csv` and the spectrum `<netlist>.TRAN.FD.csv` |
| `noise` | `<netlist>.NOISE.csv`: `FREQ`, `ONOISE`, `GAIN`, `NF` |
| `hbnoise` | `<netlist>.HBNOISE.csv`: `FREQ`, `ONOISE`, `GAIN_SIG`, `NF_DSB`, and with the image twin `GAIN_IMG`, `NF_SSB` |

VACASK reports peak amplitudes. COBRA halves the HB phasors so they follow the
Xyce convention (amplitude/2): a 0.2 V source with 50 ohm into 50 ohm gives
exactly -10 dBm.

## Noise Figure

Goals `NF` (from `noise`), and `NF_SSB` and `NF_DSB` (from `hbnoise`) are
catalogue goals in dB, for example:

```json
{"parameter": "NF", "frequency_range": "1-10ghz", "min_value": null, "max_value": 3.0,
 "weight": 1.0, "kind": "catalogue"}
```

The noise factor is

$$F = 1 + \frac{(n_{out} - n_{src} - \sum n_{term})\,T}{n_{src}\,T_0}$$

with `n_out` the total output noise, `n_src` the noise the source-port resistor
contributes to it, `n_term` that of the other ports' resistors, `T0` = 290 K and
`T` the simulation temperature (`options temp=` in degrees Celsius, default 27).
The other ports' resistors are terminations and are excluded, as in Spectre and
ADS. A matched 6 dB pad gives NF = 6.000 dB. The source-port resistor must be
noisy (the default).

!!! warning "Surrogate losses are noiseless"
    Vector-fit resistors are written with `noisy=0`, so passive losses inside
    surrogate blocks add no noise. The NF is optimistic by roughly the loss of
    the surrogate blocks in front of the first gain stage. snp2le's structure
    models (balun, inductor-pi, MIM cap, t-line RLGC, Wilkinson, branch-line)
    keep physically noisy losses; using them is planned as a per-component
    option and is not implemented yet.

### Mixers (`hbnoise`)

For `NF_SSB`, COBRA adds an image-sideband twin analysis `<name>_img` with the
input spur negated (`inspur` tone weights, e.g. `[1]` becomes `[-1]`).
Defaults are `inspur=[1]` (signal in the upper sideband of LO harmonic 1) and
`outspur=[0]` (output at the offset/IF). Then

- `F_DSB` is the formula above,
- `F_SSB = F_DSB * (g_sig + g_img) / g_sig`, with the gains of the two analyses.

If `inspur` is given in Hz instead of tone weights, `NF_SSB` is unavailable and
COBRA warns.

!!! note "Version"
    `hbnoise` exists only on VACASK `main` builds after release 0.3.4 (checked
    with `0.3.4-98-g5abdba1f`); an older binary fails with "Analysis type
    'hbnoise' not found". The noise figures are checked against an ideal
    multiplying mixer: with a noiseless mixer `NF_DSB` = 0 dB and `NF_SSB` = 3.01 dB,
    and with a noisy 50 ohm shunt at the RF node 3.01 dB and 6.02 dB, matching the
    analytic values to 0.001 dB.

## Units

VACASK's SI prefixes are case-sensitive: `f p n u m k M G T`. `M` is mega, and
`F`, `P`, `U` are not prefixes. An optimization parameter's `unit` is appended
verbatim, so use `"f"` (not `"F"`) for femtofarads. `cobra parse` warns when a
VACASK `unit` has no VACASK scale prefix.

## Tests

`tests/test_vacask_*.py`. The integration tests run when `vacask` is on `PATH`
or `COBRA_VACASK` names the executable.
