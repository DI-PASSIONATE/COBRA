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