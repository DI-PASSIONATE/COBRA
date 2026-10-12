## Examples
### Qucs-S Examples
Some examples provided in here require Qucs-S elements. You need to have Qucs-S installed and specify the path to the Qucs-S library in the `.cir` files that include it. Most of the time, this doesn't require any change, but if you have installed Qucs-S in a different location, you will need to change the path in the `.cir` files.

`.INCLUDE "/usr/share/qucs-s/spicelibrary/xfmr.cir"`

has to be changed to 

`.INCLUDE "/path/to/qucs-s/spicelibrary/xfmr.cir"`

### IHP-Open-PDK Examples

Note that for some examples, you may need to download the [IHP-Open-PDK](https://github.com/IHP-GmbH/IHP-Open-PDK) to run them, and then specify the path where it is located in the `.cir` files that include the PDK. 

Example:

`.LIB "/home/david/Documents/git/IHP-Open-PDK/ihp-sg13g2/libs.tech/xyce/models/cornerRES.lib" res_typ`

has to be changed to 

`.LIB "/path/to/IHP-Open-PDK/ihp-sg13g2/libs.tech/xyce/models/cornerRES.lib" res_typ`
### LNA With an Inductor Surrogate
`configs/lna_inductor_config.json` with `netlists/LNA/lna_inductor.cir`: a 130 GHz cascode LNA (Xyce, IHP SG13G2) whose output inductor is the ORCA model `models/DavidL-11/inductor_octa`. It optimizes S11, S22, S21 and the noise figure and keeps the amplifier stable. The netlist holds an optimized design; its header lists the inductor geometry and the results.

### VACASK Examples
`configs/vacask_trafo_acsp.json` with `netlists/VACASK/trafo_acsp.sim` is the transformer example for the VACASK simulator (`"simulator": {"name": "VacaskSimulator"}`). It needs `vacask` on `PATH`; see the [VACASK guide](../docs/advanced/vacask.md).

`configs/vacask_lna_trafo_hb.json` with `netlists/VACASK/lna_trafo_hb.sim` is the LNA example. It needs the IHP SG13G2 PDK converted for VACASK (`sg13g2tovc`), found through `SIM_INCLUDE_PATH`/`SIM_MODULE_PATH` or `~/.vacaskrc.toml`.

`configs/vacask_lna_inductor.json` with `netlists/VACASK/lna_inductor.sim` is the LNA-with-inductor preset for VACASK, with the same PDK requirements.
