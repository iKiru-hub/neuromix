# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 19:05:22 2022

@author: io
"""

import neuromix as mix
from numpy import sin, cos, exp



DNAcell_mix = ('Cell', {'components': [('Protein', {'variety': 'exp',
                                                'params': {'tau': 50,
                                                               'Eq': 0,
                                                               'w': [1]
                                                               },
                                                'more': {'trainable_params': ['tau'],
                                                         'lr': 0.01,
                                                         'activation': 'relu'
                                                         }
                                                }),
                                       ('Protein', {'variety': 'exp',
                                                    'params': {'tau': 50,
                                                                   'Eq': 0,
                                                                   'w': [1]
                                                                   },
                                                    'more': {'trainable_params': ['tau'],
                                                             'lr': 0.01,
                                                             'activation': 'relu'
                                                             }
                                                    }),
                                        ('Protein', {'variety': 'base',
                                                     'params': {},
                                                     'more': {'trainable_params': ['w'],
                                                              'lr': 0.01,
                                                              'activation': 'sigMod'
                                                              }
                                                     }),
                                        ('Protein', {'variety': 'exp',
                                                     'params': {'tau': 50,
                                                                    'Eq': 0,
                                                                    'w': [1]
                                                                    },
                                                     'more': {'trainable_params': ['tau'],
                                                              'lr': 0.01,
                                                              'activation': 'none'
                                                              }
                                                     })
                                   ],

                    'connections': [(0, 1), (1, 2), (2, 3)],

                    'params': {},

                    'more': {'nb_in': 1,
                             'nb_out': 1,
                             'trainable_params': [],
                             'cycles': 3,
                             }
                    }
           )

DNAcell_base = ('Cell', {'components': [('Protein', {'variety': 'base',
                                                     'params': {},
                                                     'more': {'trainable_params': ['w'],
                                                              'lr': 0.01,
                                                              'activation': 'sigmoid'
                                                              }
                                                     }),
                                        ('Protein', {'variety': 'base',
                                                     'params': {},
                                                     'more': {'trainable_params': ['w'],
                                                              'lr': 0.01,
                                                              'activation': 'sigmoid'
                                                              }
                                                     }),
                                        ('Protein', {'variety': 'base',
                                                     'params': {},
                                                     'more': {'trainable_params': ['w'],
                                                              'lr': 0.01,
                                                              'activation': 'sigmoid'
                                                              }
                                                     }),
                                        ('Protein', {'variety': 'base',
                                                     'params': {},
                                                     'more': {'trainable_params': ['w'],
                                                              'lr': 0.01,
                                                              'activation': 'relu'
                                                              }
                                                     })
                                   ],

                    'connections': [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)],
                    'params': {},

                    'more': {'nb_in': 1,
                             'nb_out': 1,
                             'trainable_params': [],
                             'lr': 0.1,
                             'cycles': 3,
                             }
                    }
           )


DNAexp = ('Protein', {'variety': 'exp',
                      'params': {'tau': 40,
                                     'Eq': 0,
                                     'w': [1, 1]
                                     },
                      'more': {'trainable_params': ['tau'],
                               'lr': 0.01,
                               'activation': 'none'
                               }
                      })

DNAbase = ('Protein', {'variety': 'base',
                       'params': {},
                       'more': {'trainable_params': ['w'],
                                'lr': 0.01,
                                'activation': 'relu'
                                }
                       })

DNAspike = ('Protein', {'variety': 'spike',
                        'params': {'scale': 10,
                                   'rate': 0.05,
                                   }
                        }
            )

DNAplasticty = ('Protein', {'variety': 'plasticity_base',
                            'params': {'w': [0.7]
                                           },
                            'more': {'trainable_params': ['w'],
                                     'lr': 0.001,
                                     },
                            }
                )

DNAcond = ('Protein', {'variety': 'cond',
                       'params': {'tau': 300,
                                  'Eq': 0.,
                                  'taug': 800,
                                  'Epeak': 1,
                                  'w': [1]
                                  },
                      'more': {'trainable_params': ['tau'],
                               'lr': 0.01,
                               'activation': 'none'
                               }
                      }
           )


sub = mix.brain.generate_substrate(dna=DNAspike)
sub.initialize(nb_inputs=1, idx=1)

#draw_graph(connections=DNAcell_base[1]['connections'],
#           nb_input=DNAcell_base[1]['more']['nb_in'],
#           nb_output=DNAcell_base[1]['more']['nb_out'])

# print('\nconnectivity matrix:\n', cell.connectivity_matrix)

# cell.collect_input(inputs=array([0.7]))


#### SIMULATION #####

run = 2

# initialize gym
if run > 0:

    gym = mix.sim.Gym()

    gym.add_input(kind='continuous', freq=[13], duration=1_000, nb=1,
                  function=lambda x: 0. + 0*(sin(x / 150) / 15 + 0.3 + 0.2* cos(x / 60)) * exp(-0.0003 * x),
                  nb_classes=5)

    #gym.add_target(kind='continuous', freq=[1], nb=1, function=lambda x: ((sin(x / 100) / 10 + 0.5 + 0.1 * cos(x / 70)) * exp(-0.0004 * x)),
    #               decay_params=(1, 0.005))

    gym.plot_stimuli(show_input=1, show_target=1, fix_dim=1)

    
    gym.add_substrate(substrate=sub)

# short
if run == 2:

    gym.simulation(plot_style='plot', verbose=True, training=1)

# long
elif run == 3:
    gym.long_simulation(epochs=100, info_freq=10, plot_style='plot', training=1,
                        rigenerate=0)
    
    #gym.plot_classes_test()


print('\n### end Simulation ###\n')