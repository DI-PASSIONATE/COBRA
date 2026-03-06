import os
import skrf 
import argparse
import numpy as np
from skrf.vectorFitting import VectorFitting

def vector_fit(nw: skrf.Network, name: str):
    vf = VectorFitting(nw)
    vf.auto_fit()

    # # check if input data is passive
    # nw_is_passive  = nw.is_passive()

    # if not nw_is_passive:
    #     print('Warning: Input S-parameter data is not passive. This may lead to convergence issues in circuit simulation.')

    # # get fitting error
    # rms_error = vf.get_rms_error()

    # fit_is_passive = vf.is_passive()
    # print('\nFitted data is passive = ', fit_is_passive, 'RMS error = ', rms_error)

    # # enforce passivity if required
    # if not fit_is_passive: # False: # not fit_is_passive:
    #     viol_bands = vf.passivity_test()
    #     print(f'Initial passivity violation bands = \n{viol_bands}')
    #     print(' Enforcing passivity for fitted data...')
    #     vf.passivity_enforce()
    #     fit_is_passive = vf.is_passive()
    #     print('Fitted data is passive now = ', fit_is_passive)

    # write SPICE netlist
    netlist_filename = name + '.sp'
    vf.write_spice_subcircuit_s(netlist_filename)

    # Return filename
    return netlist_filename