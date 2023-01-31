import json

protein_library  = {
        'exp': {0: {'params': {'Eq': 0,
                               'tau': 40},
                    'attrb': {'nb_inp': 1,
                             'trainable_params': [],
                             'lr': 0.01}
                    }
                },
        'linear': {0: {'params': {'w': [1.],
                                  'b': 0.},
                       'attrb': {'nb_inp': 1,
                                'trainable_params': [],
                                'lr': 0.01}
                        },   
                   1: {'params': {'b': 0.1},
                       'attrb': {'nb_inp': 1,
                                'trainable_params': ['w'],
                                'lr': 0.01,
                                'activation': 'sigmoid'}
                      }
                    },
        'cond': {0: {'params': {'Eq': 0,
                                'Epeak': 1,
                                'tau': 300,
                                'taug': 800
                                },
                     'attrb': {'nb_inp': 1,
                              'trainable_params': [],
                              'lr': 0.01}
                       }
                 },
        'spike': {0: {'params': {'scale': 10,
                                 'rest_rate': 0.},
                      'attrb': {'nb_inp': 1,
                               'trainable_params': [],
                               'lr': 0.01}
                    },
                  1: {'params': {'scale': 10,
                                 'rest_rate': 0.05},
                      'attrb': {'trainable_params': []}
                      }
                  },
        'oja': {0: {'params': {'Eq': 800,
                               'Epeak': 1,
                               'tau': 800,
                               'taug': 800},
                    'attrb': {'nb_inp': 3, 
                             'nb_extf': 3,
                             'trainable_params': [],
                             'lr': 0.05}
                        },
                1: {'params': {'Eq': 800,
                               'Epeak': 1,
                               'tau': 800,
                               'taug': 800,
                               'w': [1, 0.001, 0.001, 0.001]},
                    'attrb': {'nb_inp': 3, 
                             'nb_extf': 3,
                             'trainable_params': [],
                             'lr': 0.05}
                        }
                  },
        'jump': {0: {'params': {'tau': 7,
                                'jump_time': 10,
                                'var_jump': 0.02,
                                'w': [0.1]},
                    'attrb': {'nb_inp': 1,
                             'trainable_params': []}
                    },
                 1: {'params': {'tau': 7,
                                'jump_time': 10,
                                'var_jump': 0.02,
                                'w': [0.1]},
                    'attrb': {'nb_inp': 1,
                             'trainable_params': ['w']}
                    }
                },
}


with open('proteins.json', 'w') as f:
    f.write(json.dumps(protein_library))


print('<protein library saved>')
