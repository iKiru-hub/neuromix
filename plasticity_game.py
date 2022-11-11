#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 13:30:54 2022

@author: krubeal
"""     


import neuromix as mix  
from numpy import cos, sin, exp
#from os import system
#import time


#%% DNA definition

dna = ('Cell', {'variety': 'plasticity',
                'components': [('Protein', {'variety': 'plasticity_stdp',
                                            'params': {'w': 1.,
                                                       'A+': 0.4,
                                                       'A-': -0.4,
                                                       'a+': 0.3,
                                                       'a-': 0.3,
                                                       'tau_tr': 400,
                                                       'tau_stdp': 400
                                                       },
                                            'more': {'trainable_params': ['w'],
                                                     'lr': 0.1,
                                                     'variables': ['input', 
                                                                   'output']
                                                     }
                                            }
                                ),
                               ('Protein', {'variety': 'plasticity_stdp',
                                            'params': {'w': 1.,
                                                       'A+': 0.4,
                                                       'A-': -0.4,
                                                       'a+': 0.3,
                                                       'a-': 0.5,
                                                       'tau_tr': 300,
                                                       'tau_stdp': 300
                                                       },
                                            'more': {'trainable_params': ['w'],
                                                     'lr': 0.1,
                                                     'variables': ['input', 
                                                                   'output']
                                                     }
                                           }
                                ),
                               ('Protein', {'variety': 'cond',
                                            'params': {'tau': 300,
                                                       'Eq': 0.,
                                                       'taug': 800,
                                                       'Epeak': 1
                                                       },
                                            'more': {'trainable_params': [],
                                                     'lr': 0.01,
                                                     'activation': 'crelu'
                                                     }
                                            }
                                ),
                               ('Protein', {'variety': 'spike',
                                            'params': {'scale': 10,
                                                       'rate': 0.05,
                                                       }
                                            }
                                )],
                'connections': [(0, 2), (1, 3), (2, 4), (3, 4), (4, 5)],
                'params': {},
                'more': {'nb_in': 2,
                         'nb_out': 1,
                         'trainable_params': [],
                         'cycles': 3,
                        }
                }
       )

sub = mix.brain.generate_substrate(dna=dna)
#sub.initialize(nb_inputs=1, idx=1)


#%% Graph
#sub.add_grapher()
#sub.show_graph()

#print('\nconnectivity matrix:\n', sub.connectivity_matrix)


#%% gym

in_functions = {'l': lambda x: 2e-4 * x + 0.4,
                'nl': lambda x: (sin(x / 150) / 20 + 0.3 + 0.07 * cos(x / 60)) * exp(-0.0003 * x)
                }
tr_function = {'l': lambda x: 1.5e-4 * x + 0.,
               'nl': lambda x: (sin(x / 100) / 10 + 0.8 + 0.1 * cos(x / 70)) * exp(-0.0004 * x)}

gym = mix.sim.Gym()

gym.add_input(kind='spike', freq=[5], duration=5_000, nb=2,
              function=in_functions['l'], nb_classes=5)

gym.add_target(kind='spike', freq=[2], nb=1, 
               function=tr_function['l'], decay_params=(1, 0.005))

#gym.plot_stimuli(show_input=1, show_target=1, style='raster', fix_dim=1)


#%%

gym.add_substrate(substrate=sub)

#%% short simulation
#gym.simulation(plot_style='raster', verbose=True, training=0)


#%% training
#
gym.long_simulation(epochs=10, info_freq=5, plot_style='raster', training=1,
                    rigenerate=1, early_stopping=False)