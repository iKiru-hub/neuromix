import neuromix as mix
from numpy import array, sin, cos

if __name__ == '__main__':

   

    DNAcell = ('Cell', {'components': [('Protein', {'variety': 'base',
                                                     'params': {},
                                                     'more': {'trainable_params': ['w'],
                                                              'lr': 0.01,
                                                              'acvtivation': 'sigmoid'
                                                              }
                                                     }),
                                       ('Protein', {'variety': 'base',
                                                     'params': {},
                                                     'more': {'trainable_params': ['w'],
                                                              'lr': 0.01,
                                                              'acvtivation': 'sigmoid'
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

                        'connections': [(0, 1), (0, 2), (2, 3), (1, 3)],

                        'params': {},

                        'more': {'nb_in': 1,
                                 'nb_out': 1,
                                 'trainable_params': [],
                                 'lr': 0.001,
                                 'cycles': 2,
                                 }
                        }
               )


    cell = mix.brain.generate_substrate(dna=DNAcell)

    # print('\nconnectivity matrix:\n', cell.connectivity_matrix)

    # cell.collect_input(inputs=array([0.7]))
    
    cell.add_grapher()
    cell.set_livestream(state=0)
    #cell.show_graph()


    #### SIMULATION #####

    run = 3

    # initialize gym
    if run > 0:

        gym = mix.sim.Gym()

        gym.add_input(kind='classes', freq=3, duration=3000, nb=1,
                      function=lambda x: sin(x / 50) / 2 + 0.5, nb_classes=2)

        gym.add_target(kind='classes', function=lambda x: cos(x / 50) / 2 + 0.5)

        # gym.plot_stimuli(show_input=1, show_target=1)

        gym.add_substrate(substrate=cell)

    # short
    if run == 2:

        gym.simulation(plot_style='plot', verbose=True, training=True)

    # long
    elif run == 3:

        gym.long_simulation(epochs=10, info_freq=2, plot_style='', 
                            training=1, rigenerate=0)
        
        #gym.substrate.set_livestream(1)
        #gym.simulation(plot_style='plot', verbose=True, training=True)
        
    elif run == 4:
        
        gym.substrate.set_livestream(1)
        gym.simulation(plot_style='plot', verbose=True, training=True)


    print('\n### end Simulation ###\n')
