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
            } 
}


with open('cells.json', 'w') as f:
    f.write(json.dumps(cell_library))


print('<cell library saved>')

