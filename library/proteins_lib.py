"""
this is a file reference for DNA samples
"""


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



