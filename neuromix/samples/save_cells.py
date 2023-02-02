import json


cell_library = {
    'root': {0 : {'components': [('Protein', {'family': 'linear',
                                             'id': '0'}),
                                ('Protein', {'family': 'linear',
                                             'id': '0'}),
                                ('Protein', {'family': 'linear',
                                             'id': '0'})
                                ],
                 'connections': [(0, 2), (1, 2), (0, 3), (1, 3), (2, 4), (3, 4)],
                 'params': {},
                 'attrb': {'nb_inp': 2,
                          'nb_out': 1,
                          'idx_out': [2],
                          'lr': 0.01,
                          'trainable_params': [],
                          'cycles': 2}
                }
            },
    'jump_trace': {0 : {'components': [('Protein', {'family': 'jump_trace',
                                                    'params': {'tau': 7,
                                                               'tau_trace': 100,
                                                               'jump_time': 10,
                                                               'var_jump': 0.02,
                                                               'w': [0.1, 0.1]},
                                                     'attrb': {'nb_inp': 2,
                                                               'nb_intf': 1,
                                                               'trainable_params': ['w']}
                                          }),
                                       ('Protein', {'family': 'jump_trace',
                                                    'params': {'tau': 7,
                                                               'tau_trace': 100,
                                                               'jump_time': 10,
                                                               'var_jump': 0.02,
                                                               'w': [0.1, 0.1]},
                                                    'attrb': {'nb_inp': 2,
                                                              'nb_intf': 1,
                                                              'trainable_params': ['w']}
                                          }),
                                       ('Protein', {'family': 'jump_trace',
                                                    'params': {'tau': 7,
                                                               'tau_trace': 100,
                                                               'jump_time': 10,
                                                               'var_jump': 0.04,
                                                               'w': [0.1, 0.1]},
                                                    'attrb': {'nb_inp': 2,
                                                               'nb_intf': 1,
                                                    'trainable_params': ['w']}
                                          }),
                                       ('Protein', {'family': 'jump_trace',
                                                    'params': {'tau': 7,
                                                               'tau_trace': 100,
                                                               'jump_time': 10,
                                                               'var_jump': 0.04,
                                                               'w': [0.1, 0.1]},
                                                    'attrb': {'nb_inp': 2,
                                                              'nb_intf': 1,
                                                              'trainable_params': ['w']}
                                          }),
                                       ('Protein', {'family': 'jump_trace',
                                                    'params': {'tau': 7,
                                                               'tau_trace': 100,
                                                               'jump_time': 10,
                                                               'var_jump': 0.04,
                                                               'w': [0.1, 0.1]},
                                                    'attrb': {'nb_inp': 2,
                                                              'nb_intf': 1,
                                                              'trainable_params': ['w']}
                                          }),
                                       ('Protein', {'family': 'cond_sat',
                                                    'params': {'Eq': 0,
                                                               'Epeak': 1,
                                                               'tau': 50,
                                                               'taug': 70,
                                                               'thr_sat': 0.05,
                                                               'w': [4]},
                                                    'attrb': {'nb_inp': 1}
                                          })],
                'connections': [[0, 2], [1, 2],[0, 3], [1, 3], [2, 4], [3, 4], [2, 5], [3, 5], [4, 6], [5, 6]],
                'params': {'lr_est': 0.03},
                'attrb': {
                    'nb_inp': 2,
                    'nb_out': 1,
                    'nb_jumps': 2,
                    'idx_out': [4],
                    'idx_plastic': [0, 1, 2, 3, 4],
                    'idx_intf': [0, 1, 2, 3, 4, 5],
                    'idx_extf': [],
                    'cycles': 3,
                    'trainable_params': []}
                }}
}


with open('cells.json', 'w') as f:
    f.write(json.dumps(cell_library))


print('<cell library saved>')

