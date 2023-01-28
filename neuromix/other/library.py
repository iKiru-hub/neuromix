"""
this is a file reference for DNA samples
"""

""" PROTEINS """

protein_library  = {
        'exp': {0: {'params': {'Eq': 0,
                               'tau': 40,
                               'w': [1]},
                    'more': {'trainable_params': [],
                             'nb_int': 1,
                             'lr': 0.01,
                             'activation': 'none'}
                    }
                },
        'linear': {0: {'params': {'b': 0.1},
                       'more': {'trainable_params': [],
                                'nb_int': 1,
                                'lr': 0.01,
                                'activation': None}
                       },
                   1: {'params': {'b': 0.1},
                       'more': {'trainable_params': ['w'],
                                'nb_int': 2,
                                'lr': 0.01,
                                'activation': 'sigmoid'}
                 },
        'cond': {0: {'params': {'Eq': 0,
                                'Epeak': 1,
                                'tau': 300,
                                'taug': 800,
                                'w': [1]
                                },
                     'more': {'trainable_params': [],
                              'lr': 0.01,
                              'activation': 'none'}
                       }
                 },
        'spike': {0: {'params': {'scale': 10,
                                 'rate': 0.05},
                      'more': {}
                      }
                  },
        'oja_1': {0: {'params': {'Eq': 800,
                                 'Epeak': 1,
                                 'tau': 800,
                                 'taug': 800,
                                 'w': [1, 0.001, 0.001, 0.001]},
                      'more': {'trainable_params': ['w'],
                               'lr': 0.05, 
                               'activation': 'none'}
                        }
                  }
        }


# return each DNA sample as an object
protein_lib = {'exp': DNAp_exp,
               'base': DNAp_base,
               'spike': DNAp_spike,
               'cond': DNAp_cond,
               'plasticity_base': DNAp_plasticty,
               'plasticity_hebb': DNAp_hebb1,
              }


""" CELLS """


cell_library = {
    'mlp': {0 : {'components': [('Protein', {'family': 'linear',
                                             'id': 1}),
                                ('Protein', {'family': 'linear',
                                             'id': 1}),
                                ('Protein', {'family': 'linear',
                                             'id': 1})
                                ],
                 'connections': [(0, 2), (1, 2), (0, 3), (1, 3), (2, 4), (3, 4)],
                 'params': [],
                 'more': {'nb_in': 2,
                          'nb_out': 1,
                          'idx_out': 2,
                          'lr': 0.01,
                          'trainable_params': [],
                          'cycles': 2}
                }
            } 
}

DNAc_c2 = ('Cell', {'variety': 'base',
                     'components': [('Protein', {'variety': 'cond',
                                                 'params': {'tau': 300,
                                                            'Eq': 0.,
                                                            'taug': 800,
                                                            'Epeak': 1,
                                                            'w': [1]
                                                            },
                                                 'more': {'trainable_params': [],
                                                          'lr': 0.01,
                                                          'activation': 'sigmoid'
                                                          }
                                                 }),
                                    ('Protein', {'variety': 'cond',
                                                 'params': {'tau': 300,
                                                            'Eq': 0.,
                                                            'taug': 800,
                                                            'Epeak': 1,
                                                            'w': [1, 0],
                                                            },
                                                 'more': {'trainable_params': [],
                                                          'lr': 0.01,
                                                          'activation': 'sigmoid'
                                                          }
                                                     }),
                                   ],
                    'connections': [(0, 2), (1, 3), (2, 3)],
                    'params': {},
                    'more': {'nb_in': 2,
                             'nb_out': 1,
                             'trainable_params': [],
                             'lr': 0.1,
                             'cycles': 2,
                             }
                    }
           )


cell_lib = {'c2': DNAc_c2}



###
# Protein template 

"""
('Protein', {'variety': str,
             'param': {'p1': float,
                       'p2': float,
                       'w': list},
             'more': {'nb_in': int,
                      'trainable_params': [],
                      'lr': float,
                      'activation': str}
            }
)


"""
