#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 13:30:54 2022

@author: krubeal
"""     


import neuromix as mix  
from numpy import cos, sin, exp


#%% DNA definition

dna = ('Cell', {'components': [('Protein', {'variety': 'plasticity_base',
                                            'params': {'w': 1.},
                                            'more': {'trainable_params': ['w'],
                                                     'lr': 0.01,
                                                     'variables': ['input', 
                                                                   'output']
                                                     }
                                            }
                                ),
                               ('Protein', {'variety': 'plasticity_base',
                                            'params': {'w': 1.},
                                            'more': {'trainable_params': ['w'],
                                                     'lr': 0.01,
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
                                            'more': {'trainable_params': ['tau'],
                                                     'lr': 0.01,
                                                     'activation': 'crelu'
                                                     }
                                            }
                                ),
                               ('Protein', {'variety': 'spike',
                                            'params': {'scale': 10,
                                                       'rate': 0.02,
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
sub.add_grapher()
sub.show_graph()

print('\nconnectivity matrix:\n', sub.connectivity_matrix)


#%% gym

in_functions = {'l': lambda x: 2e-4 * x + 0.4,
                'nl': lambda x: (sin(x / 150) / 20 + 0.3 + 0.07* cos(x / 60)) * exp(-0.0003 * x)
                }
tr_function = {'l': lambda x: 1.5e-4 * x + 0.,
               'nl': lambda x: (sin(x / 100) / 10 + 0.8 + 0.1 * cos(x / 70)) * exp(-0.0004 * x)}

gym = mix.sim.Gym()

gym.add_input(kind='spike', freq=[10], duration=1_000, nb=2,
              function=in_functions['l'], nb_classes=5)

#gym.add_target(kind='continuous', freq=[1], nb=1, 
#               function=tr_function['l'], decay_params=(1, 0.005))

gym.plot_stimuli(show_input=1, show_target=1, fix_dim=1)


#%%

gym.add_substrate(substrate=sub)

#%% short simulation
gym.simulation(plot_style='spike', verbose=True, training=0)

#%% training

gym.long_simulation(epochs=100, info_freq=20, plot_style='plot', training=1,
                    rigenerate=0)

