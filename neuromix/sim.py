import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from neuromix import brain
import warnings
from tqdm import tqdm 


stored_functions = {'parabola': lambda x: max((0. + 0.01 * x - \
                                               0.00006 * x ** 2, 0)),
                    'periodic': lambda x: 0.5 + 0.5 * np.cos(x / 60)}


########### GAMES and GYM ###########

class TestMap:
    
    """ test of a substrate performance on some input-output mapping """
    
    
    def __init__(self):
        
        """
        Returns
        -------
        None
        """

        # stimuli
        self.inputs = None
        self.targets = np.array(0)

        # checklist
        self.nb_input_channels = 1
        self.checklist = {'substrate': False,
                          'input': False,
                          'target': False}

        # stimuli data
        self.stimuli_data = {'input': {'kind': str,
                                       'freq': int,
                                       'duration': int,
                                       'nb': int,
                                       'function': None,
                                       'nb_classes': 1},

                             'target': {'kind': str,
                                        'delay': int,
                                        'random_delay': bool,
                                        'nb': 1,
                                        'function': None,
                                        'decay_params': (1, 0.005),
                                        'freq': 1}}

        self.sim_record = None 

        print('\n@TestMap')

    def add_input(self, kind='spike', freq=[3], duration=1000, nb=1, 
                  function=None, nb_classes=1, overwrite=True, verbose=True):

        """
        generate an input train for the simulation
        
        Parameters
        ----------
        kind: str
            kind of input from ('spike', 'time', 'spike_time', 'spike_random', 'continuous',
            'classes'), default 'spike'
        freq: list
            list of frequencies in Hz of the input, valid for 'spike' kind, default [3]
        duration: int 
            duration in ms of the input train default 1000
        nb: int
            number of input channels, default 1
        function: lambda expression 
            function generation of the time-dependant input for duration,valid for kind 
            'continuous', default None
        nb_classes: int
            number of classes for kind 'classes', default=1
        overwrite: bool
            if True the previous inputs are overwritten [if present], default True
        verbose: str 

        Returns
        -------
        numpy.ndarray 
            activity record of the simulation 
        
        -----------------------------------------------------------------------
        kind 'spike': spike train at the specified frequency
        kind 'time': constant increment, i.e. diagonal y=x
        kind 'spike_time': constant increment starting at each spike time
        kind 'spike_random': like spike but uses random values for the spike 
                events from a Uniform (0, 1)
        kind 'continuous': the corresponding target is the function applied 
                to each timestep
        """

        self.checklist['input'] = False
        
        #### CHECK CALL ####
        
        # freq type #
        if not isinstance(freq, list) and not isinstance(freq, np.ndarray):
            raise TypeError(f'wrong type for "freq" parameter [{type(freq)}], '
                            'only <list> or <ndarray> allowed')
        
        # number of frequencies #
        if len(freq) != nb:
            nb = len(freq)
            warnings.warn(f"number of frequencies different from the number of specified inputs, set nb={nb}")
            
        # list to array
        freq = np.array(freq).reshape(len(freq), -1)


        #### BUILD INPUTS ####
        
        # time
        if kind == 'time':

            inputs = np.array([np.arange(duration)] * nb)

        elif kind[:5] == 'spike':

            inputs = np.random.binomial(1, freq / 1000, size=(nb, duration)).astype(float)

            if kind == 'spike_time':
                
                # self.inputs *= np.array([range(duration)] * nb)
    
                for i, row in enumerate(self):
                    counter = 0
                    for j, x in enumerate(row):
                        if x != 0:
                            counter = 0
                        else:
                            counter += 1
                        inputs[i, j] = counter

            elif kind == 'spike_random':
    
                inputs *= np.random.uniform(0, 1, size=(nb, duration)).astype(float)

        elif kind == 'continuous':

            if not function:
                raise ValueError('missing function for input kind "continuous"')

            inputs = np.array([[function(np.arange(duration))] for _ in range(nb)]).reshape(nb, duration)

        elif kind == 'classes':

            inputs = np.around(np.random.choice(range(1, nb_classes + 1), size=(nb, duration)) / nb_classes, 3)

        else:
            raise ValueError(f'"{kind}" input not supported for now')

        # save input
        if self.inputs is None or overwrite:
            self.inputs = inputs 

            # record details
            self.checklist['input'] = True
            self.stimuli_data['input']['kind'] = kind
            self.stimuli_data['input']['freq'] = freq
            self.stimuli_data['input']['duration'] = duration
            self.stimuli_data['input']['nb'] = nb
            self.stimuli_data['input']['function'] = function
            self.stimuli_data['input']['nb_classes'] = nb_classes
        
        else:
            assert self.stimuli_data['input']['kind'] == 'classes' and kind != 'classes', "existing input of kind 'classes' can not be extended with new kind {kind}"
            assert self.inputs.shape == inputs.shape, \
                    f"new input with shape={inputs.shape} incompatible with existing input with shape={self.inputs.shape}"
            assert len(inputs.shape) > 1, \
                    f"at least two dimensions are required, {len(inputs.shape)} provided"

            self.inputs = np.concatenate((self.inputs, inputs))

            # record details
            self.nb_input_channels += 1
            input_name = f"input_{self.nb_input_channels}"

            self.stimuli_data[input_name] = {}
            self.stimuli_data[input_name]['kind'] = kind
            self.stimuli_data[input_name]['freq'] = freq
            self.stimuli_data[input_name]['duration'] = duration
            self.stimuli_data[input_name]['nb'] = nb
            self.stimuli_data[input_name]['function'] = function
            self.stimuli_data[input_name]['nb_classes'] = nb_classes

        if verbose:
            print(f'+ input [{self.nb_input_channels}] "{kind}" added | shape={self.inputs.shape}')

    def add_target(self, kind='spike', delay=200, random_delay=True, nb=0,
                   function=None, decay_params=(1, 0.005), freq=[3], verbose=True):

        """
        generate a target train after a certain delay from the inputs for the 
                simulation
        
        Parameters
        ----------
        kind: str
            kind of input from ('spike', 'spike_random', 'spike_decay', 'cont_decay', 
            'continuous', 'cont_onset', 'classes'), default 'spike'
        delay: float
            delay after the input onset, default 100 [ms]
        random_delay: bool
            delay drawn from a Normal(mean=delay std=50), default True
        nb: int
            number of targets, if left the default 0 then the same input number will be taken
        function: lambda expression
            function generation of the time-dependant input for duration, valid for 'continuous' 
            and 'decay' kind, default None
        decay_params: tuple
            (height steepness) as: height * exp(-steepness * x), default (1, 0.005)
        freq: float
            frequency in Hz, used for continuous input kind
        
        Returns
        -------
        None

        ----------------------------------------------------------------------
        kind 'spike': spike after a input spike plus a delay
        kind 'spike_random': like spike but uses random values for the spike 
                events from a Uniform (0, 1)
        kind 'spike_decay': like spike but uses values according to the 
                provided function
        kind 'cont_decay': like 'spike_decay' but it covers all the timestep
        kind 'cont_onset': the corresponding target is the function applied 
                to each input value
        kind 'continuous': the corresponding target is the function applied 
                to each timestep
        """

        # check input presence
        if not self.checklist['input']:
            raise ValueError('target not generated, missing input')

        # settings
        self.checklist['target'] = False
        

        #### CHECK CALL ###

        # load input data
        input_kind = self.stimuli_data['input']['kind']
        nb = self.stimuli_data['input']['nb'] if nb == 0 else nb
        nb_classes = self.stimuli_data['input']['nb_classes']
        
        # check kind
        if input_kind == 'time':
            warnings.warn('target not generated for input kind "time"')
            return

        elif input_kind == 'classes' and kind != 'classes':
            raise ValueError(f'target of kind "{kind}" can not be generated '
                             'for input kind "classes"')
            
        elif input_kind != 'classes' and kind == 'classes':
            raise ValueError(f'target of kind "classes" can not be generated '
                             f'for input kind "{input_kind}"')

        elif kind == 'continuous' or kind == 'cont_onset':

            if function is None:
                raise NotImplementedError(f'missing function for kind "{kind}"')
                
        # check parameters
        
        # freq type # <--- not really relevant, frequency is never used here
        if not isinstance(freq, list) and not isinstance(freq, np.ndarray):
            raise TypeError(f'wrong type for "freq" parameter [{type(freq)}], '
                             'only <list> or <ndarray> allowed')
        
        # number of frequencies #
        if len(freq) > nb:
            warnings.warn('number of frequencies greater than the number of '
                          'specified targets, adapting parameter "nb"')
            nb = len(freq)
            
        # list to array
        freq = np.array(freq).reshape(len(freq), -1)
                

        duration = self.inputs.shape[1]
        self.targets = np.zeros((nb, duration)).astype(float)

        ############################################################ 

        # classes data
        if kind == 'classes':
            self.targets = np.zeros((nb, duration)).astype(float)

            classes = np.array([round(i / nb_classes, 3) for i in range(1, nb_classes + 1)])
            np.random.shuffle(classes)
            pairs = {}
            y_idx = 0
            for i, row in enumerate(self.inputs):
                for j, x in enumerate(row):

                    # unmatched input class
                    if x not in tuple(pairs.keys()):
                        pairs[x] = classes[y_idx]  # record next target class
                        y_idx += 1  # step target index count

                    self.targets[i, j] = pairs[x]

        else:

            # for each input dimension
            for i, row in enumerate(self.inputs):

                if (i + 1) > nb:
                    break

                # when there is a non-zero event (i.e. a spike)
                for j, x in enumerate(row):

                    # continuous | no delay
                    if kind == 'continuous':
                        self.targets[i, j] = function(j)
                        continue

                    # <time> input ###########################################
                    if input_kind == 'spike_time' or input_kind == 'time':

                        # zero x is relevant
                        if x == 0:

                            # define a distance at which place a target value 
                            # with proportional height
                            if random_delay:
                                d = int(abs(np.random.normal(delay, 50, 1)))
                            else:
                                d = delay

                            # ignore if the distance overshoot the input length
                            if (j + d) > (duration - 1):
                                continue

                            # spike | intensity 1
                            if kind == 'spike':
                                self.targets[i, j + d] = 1
                                continue

                            # spike_random | intensity randomly from (0, 1)
                            elif kind == 'spike_random':
                                self.targets[i, j + d] = np.random.random()
                                continue

                            # spike_decay | intensity as function(delay)
                            elif kind == 'spike_decay':
                                self.targets[i, j + d] = decay_params[0] * \
                                    np.exp(-decay_params[1] * d)
                                continue

                            # continuous decay | intensity as function(t)
                            elif kind == 'cont_decay':
                                for t in range(j, duration):
                                    self.targets[i, t] = decay_params[0] * \
                                        np.exp(-decay_params[1] * t)
                                    
                            # cont_decay | intensity as function(t) 
                            # specified as : for t in range(j, duration) 
                            elif kind == 'cont_onset':
                                for t, k in enumerate(range(j, duration)):
                                    self.targets[i, k] = function(t)
                                continue

                            else:
                                raise NotImplementedError(f'{kind} target not '
                                                          'supported with input'
                                                          ' "{input_kind}" for now')

                    # <spike> input ##########################################
                    elif input_kind[:5] == 'spike':

                        # non-zero x is relevant
                        if x != 0:

                            # continuous decay | intensity as function(t)
                            # input scaling
                            if kind == 'cont_decay':
                                for t in range(duration - j):
                                    self.targets[i, j + t] = x * \
                                        decay_params[0] * np.exp(-decay_params[1] * t)
                                continue
                            
                            # cont_decay | intensity as function(t) 
                            # as for t in range(j, duration) 
                            # input scaling
                            elif kind == 'cont_onset':
                                for t, k in enumerate(range(j, duration)):
                                    self.targets[i, k] = x * function(t)
                                continue
                            

                            ############ delay ############

                            # define a distance at which place a target value with proportional height
                            if random_delay:
                                d = int(abs(np.random.normal(delay, 50, 1)))
                            else:
                                d = delay

                            # ignore if the distance overshoot the input length
                            if (j + d) > (duration - 1):
                                continue

                            # spike | intensity 1
                            if kind == 'spike':
                                self.targets[i, j + d] = 1
                                continue

                            # spike_random | intensity randomly from (0, 1)
                            elif kind == 'spike_random':
                                self.targets[i, j + d] = np.random.random()
                                continue

                            # spike_decay | intensity as function(delay) | input scaling
                            elif kind == 'spike_decay':
                                self.targets[i, j + d] = x * decay_params[0] * np.exp(-decay_params[1] * d)
                                continue

                            else:
                                raise NotImplementedError(
                                    f'"{kind}" target not supported with input "{input_kind}" for now')
                                
                                
                    # <continuous> input #####################################
                    elif input_kind == 'continuous':
                        
                        if not np.random.binomial(1, freq/1000):
                            continue
                        
                        # spike | intensity 1 | input scaling
                        if kind == 'spike':
                            self.targets[i, j] = x
                            continue

                        # spike_random | intensity randomly from (0, 1)
                        elif kind == 'spike_random':
                            self.targets[i, j] = np.random.random()
                            continue
                        
                        
                        # continuous decay | intensity as function(t) | input scaling
                        elif kind == 'cont_decay':
                            for t in range(duration - j):
                                self.targets[i, j + t] = x * decay_params[0] * np.exp(-decay_params[1] * t)
                        
                        # cont_decay | intensity as function(t) for t in range(j, duration) 
                        # input scaling
                        if kind == 'cont_onset':
                            if x != 0:                        
                                for t, k in enumerate(range(j, duration)):
                                    self.targets[i, k] = x * function(t)
                            continue 
                    
                
        self.checklist['target'] = True
        self.stimuli_data['target']['kind'] = kind
        self.stimuli_data['target']['delay'] = delay
        self.stimuli_data['target']['random_delay'] = random_delay
        self.stimuli_data['target']['nb'] = nb
        self.stimuli_data['target']['function'] = function
        self.stimuli_data['target']['decay_params'] = decay_params
        self.stimuli_data['target']['freq'] = freq

        if verbose:
            print(f'+ target "{kind}" added')
         
    def rigenerate_data(self, new_input=False, new_target=False):
        
        """
        generate a new set of inputs or targets with the same settings
        
        Parameters
        ----------
        new_input: bool
            if True new inputs will be generated, default False
        new_target: bool
            if True new targets will be generated, default False
        
        Returns
        -------
        None
        """
        
        if new_input: 
            
            self.add_input(kind=self.stimuli_data['input']['kind'],
                           freq=self.stimuli_data['input']['freq'], 
                           duration=self.stimuli_data['input']['duration'], 
                           nb=self.stimuli_data['input']['nb'], 
                           function=self.stimuli_data['input']['function'], 
                           nb_classes=self.stimuli_data['input']['nb_classes'],
                           verbose=False)
            
        if new_target and self.checklist['target']:
            
            self.add_target(kind=self.stimuli_data['target']['kind'],
                            delay=self.stimuli_data['target']['delay'],
                            random_delay=self.stimuli_data['target']['random_delay'],
                            nb=self.stimuli_data['target']['nb'],
                            function=self.stimuli_data['target']['function'],
                            decay_params=self.stimuli_data['target']['decay_params'],
                            freq=self.stimuli_data['target']['freq'],
                            verbose=False)

    def plot_stimuli(self, show_input=True, show_target=False, style='plot', 
                     fix_dim=False):
        
    
        """
        plot the stimuli
        
        Parameters
        ----------
        show_input: bool
            if True it also plots the inputs, default False
        show_target: bool
            if True it also plots the targets, default False
        style: str
            style of the plot from ('plot', 'raster'), default 'plot'
        fix_dim: bool
            if True ylim=(0, 1), default False
        
        Returns
        -------
        None
        """

        # check input / target
        if not self.checklist['input']:
            raise NotImplementedError('target not generated, missing input')

        elif not self.checklist['target']:
            if show_target:
                warnings.warn('missing target, it will not be plotted')
                show_target = False

        # check applicable style
        if style == 'raster' and self.stimuli_data['input']['kind'] == 'continuous':
            style = 'plot'
            warnings.warn('"raster" style can not be used with continuous '
                          'inputs, "plot" will be used')
            print()

        nb, duration = self.inputs.shape

        # classes data kind
        if self.stimuli_data['input']['kind'] == 'classes':
            
            nb_classes = self.stimuli_data['input']['nb_classes']

            print('\n--- Classes data ---')
            
            # absent target
            if not self.checklist['target'] or not show_target:
                
                for i in range(nb):
                    print(f'\nstream {i+1}:')

                    for t, x in zip(range(duration), self.inputs[i]):
                        print(f'{x}')

                        if t == 30:
                            break
                        
                    plt.scatter(self.inputs[i], np.zeros(len(self.inputs[i])))
                plt.xlabel('input')
                plt.xticks(np.arange(1, nb_classes+1)/nb_classes, range(nb_classes))
                plt.title('Classes')
                plt.yticks(())
                plt.show()
                
            # present target
            else:
                
                # for each number of input streams
                for i in range(nb):
                    print(f'\nstream {i+1}:')
    
                    for t, x, y in zip(range(duration), self.inputs[i], self.targets[i]):
                        print(f'({x}, {y})')
    
                        if t == 30:
                            break
                        
                    plt.scatter(self.inputs[i], self.targets[i])
                plt.xlabel('input classes')
                plt.xticks(np.arange(1, nb_classes+1)/nb_classes, range(nb_classes))
                plt.ylabel('target classes')
                plt.yticks(np.arange(1, nb_classes+1)/nb_classes, range(nb_classes))
                plt.title('Classes')
                plt.show()

            print()

            return

        # non-classes data kind

        x = range(duration)

        if style == 'plot':

            if show_input:
                for i, inp in enumerate(self.inputs):
                    plt.plot(x, inp, alpha=0.7, label=f'input {i + 1}')

            if show_target:
                for j, tar in enumerate(self.targets):
                    plt.plot(x, tar, alpha=0.7, label=f'target {j + 1}')

            plt.xlabel('time [ms]')
            plt.ylabel('stimulus intensity')
            plt.title('Stimulation')
            if fix_dim:
                plt.ylim((0, 1))
            plt.legend()
            plt.show()

        elif style == 'raster':
            
            self.plot_raster(show_target=show_target) 
        
    def plot_raster(self, ax=None, show_input=True, show_target=True, show_activity=True, record_idx=0, ylims=(0, 1), alpha=1):
        
        """
        raster plot for spike data

        Parameters 
        ----------
        ax : plt.Axes
            axes to plot on, default None
        show_input: bool 
        show_target: bool
        show_activity: bool 
        record_idx: int 
            index of the record to plot, default last record index
        ylims: tuple
            ylims for the plot, default (0, 1)
        alpha: float
            alpha value for the plot, default 1
        
        Returns 
        -------
        None
        """
            
        # number of rows of data 
        nb = 0
        
        # check
        if show_input:
            if self.stimuli_data['input']['kind'] not in ('spike', 'spike_random'):
                warnings.warn(f"input kind <{self.stimuli_data['input']['kind']}> not supported for raster plot, ignored")
                show_input = False
            elif not self.checklist['input']:
                warnings.warn("input will not be plotted, not found")
                show_input = False
            else:
                nb += len(self.inputs)

        if show_target:
            if self.stimuli_data['target']['kind'] not in ('spike', 'spike_random', 'spike_decay'):
                warnings.warn(f"target kind <{self.stimuli_data['target']['kind']}> not supported for raster plot, ignored")
                show_target = False
            elif not self.checklist['target']:
                warnings.warn("target will not be plotted, not found")
                show_target = False
            else:
                nb += len(self.targets)

        if show_activity:
            assert record_idx in self.sim_record.keys(), f"record index {record_idx} not found, available={self.sim-record.keys()}"
            
            if self.sim_record is None:
                warnings.warn("recorded activity will not be plotted, not found")
                show_activity = False
            else:
                spikes = self.sim_record[len(self.sim_record.keys())-1]['activity']
                nb += len(spikes)

        if (not show_input) and (not show_target) and (not show_activity):
            warnings.warn("no data to plot has been selected")
            return

        ### Prepare Data ###
        #stimuli = np.zeros((nb, duration))
        #positions = np.ones((nb, duration))*-1
        data = []
        names = []
        colors = []
        i = 0
        j = 0
        k = 0

        # y-axis scale
        y_scale = (ylims[1] - ylims[0]) / 1.5


        # input spikes
        if show_input:
            for i, row in enumerate(self.inputs):
                x = []
                y = []
                for t, v in enumerate(row):
                    if v > 0:
                        #stimuli[i, t] = v
                        #positions[i, t] = 1.5 - (i + 1) / nb
                        x += [t]
                        y += [(1.5 - (i + 1) / nb) * y_scale + ylims[0]]

                data += [(x, y)]
     
                names += [f'input {i + 1}']
                colors += ['black']

        # target spikes
        if show_target:
            for j, row in enumerate(self.targets):
                x = []
                y = []
                for t, v in enumerate(row):
                    if v > 0:
                        #stimuli[i + j + 1, t] = v
                        #positions[i + j + 1, t] = 1.5 - (i + j + 2) / nb
                        x += [t]
                        y += [(1.5 - (i + j + 1 + int(show_input)) / nb) * y_scale + ylims[0]]

                data += [(x, y)]
    
                names += [f'target {j + 1}']
                colors += ['orange']
            
        # other spikes
        if show_activity:
            for k, row in enumerate(spikes):
                x = []
                y = []
                for t, v in enumerate(row):
                    if v > 0:
                        #stimuli[i + j + k + 2, t] = v
                        #positions[i + j + k + 2, t] = 1.5 - (i + j + k + 3) / nb
                        x += [t]
                        y += [(1.5 - (i + j + k + 1 + int(show_input) + int(show_target)) / nb) * y_scale + ylims[0]]
                data += [(x, y)]

                names += [f'other {k+1}']
                colors += ['green']
        
        ### Plotting ###
        
        # setting 
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        # plot
        for i, stimulus in enumerate(data):
            ax.scatter(stimulus[0], stimulus[1], marker='|', lw=1, color=colors[i], alpha=alpha)

        # raster
        # for l, stimulus in enumerate(stimuli):
            # plt.subplot(nb, 1, k+1)
            #plt.plot(x, positions[l], 'o', color=colors[l], label=f'{names[l]}')
            #plt.scatter(x, positions[l], c=stimulus, cmap=colors[l])
            #plt.scatter(duration + 100, positions[l, -1], c=0., cmap=colors[l], label=f'{names[l]}')

        ax.set_ylim(ylims)
        
        #ax.set_yticks(())
        ax.set_xlabel('time [ms]')
        #plt.legend(loc='lower left')
        ax.set_title('Stimulation raster plot')
    
    def testing(self, candidate: object, verbose=False, 
                return_obj=True, plotting=False):
        
        """
        test an class.AgentEvo object on a previously added mapping
        
        Parameters
        ----------
        candidate: class.AgentEvo
        verbose: bool
            default False
        return_obj: bool
            if True the fitted candidate is returned, default True
        
        Returns
        -------
        candidate : object
            if return_obj else None
        """
        

        if not self.checklist['input']:
            raise NotImplementedError('missing input')

        elif not self.checklist['target']:
            raise NotImplementedError('missing target')

        # settings
        duration = self.inputs.shape[1]
        
        # record
        loss = 0.
        
        if plotting:
            outputs = np.zeros(duration)
        
        if verbose:
            print(f'\nTesting candidate: {candidate.get_name()}')
            
        # run
        for ms in range(duration):

            # collect input
            candidate.collect_input(inputs=self.inputs[:, ms].T)

            # agent step
            candidate.step()

            # error
            error = self.targets[:, ms] - candidate.get_output()
            if error > 1e6:
                break
            
            loss += error.item() ** 2
            
            # record
            if plotting:
                outputs[ms] = candidate.get_output().item()
            
            
        if verbose:
            print(f'result: {loss/duration:.4f} error/ms')
        
        # if the candidate is an Agent type it has its fitness updates
        if hasattr(candidate, 'set_fitness') and callable(getattr(candidate, 
                                                        'set_fitness', None)):
            candidate.set_fitness(fitness= -1 * round(loss, 4))
            
        # plot
        if plotting:
            
            
            plt.plot(range(duration), self.inputs[0, :], '--k', alpha=0.5, 
                     label='input')

            plt.plot(range(duration), self.targets[0, :], '--', color='orange', 
                     alpha=0.8, label='target')

            plt.plot(range(duration), outputs, '-g', alpha=0.85, label='output')

            plt.xlabel('time [ms]')
            plt.legend()
            plt.title('candidate test')

        if return_obj:
            return candidate
    
    def is_complete(self):
        
        """
        Returns
        -------
        checklist : bool
            True if there are both targets and inputs, False otherwise
        """
        
        return self.checklist['input'] and self.checklist['target']

    def get_record(self):

        """
        return the record of the last simulation 

        Returns
        -------
        dict 
            keys ('activity', 'record')
        """

        return self.sim_record


class Gym(TestMap):
    
    """ a TestMap subclass 
    
    it extend the simple test to include a one-epoch simulation method
    as well as a multiple-epochs simulation ideal for training and long-term
    dynamics with similar stimulation
    """

    def __init__(self):
        
        """
        Returns
        -------
        None
        """
        
        super().__init__()

        # substrate
        self.substrate: object
        self.substrate_name = ""
        
        print('\n@Gym')
       
    def add_substrate(self, substrate: object):

        """
        add a class.Substrate object to use
        
        Parameters
        ----------
        substrate: class.Substrate object
        
        Returns
        -------
        None
        """

        self.substrate = substrate

        self.checklist['substrate'] = True
        print(f'+ substrate "{self.substrate.get_substrate_name()}" added')
        
        if not substrate.is_initialized():
            warnings.warn('substrate not initialized')

    def simulation(self, plot_style=None, verbose=True, training=False, return_=False):

        """
        simulation one epoch of activity
        
        Parameters
        ----------
        plot_style: str
            plotting style from (None, 'plot', 'raster'), default None
        verbose: bool
            print runtime information, default True
        training: bool
            allow training of the substrate, default False
        
        Returns
        -------
        None
        """

        if not self.checklist['substrate']:
            raise NotImplementedError('missing substrate')

        elif not self.checklist['input']:
            raise NotImplementedError('missing input')

        elif not self.checklist['target']:
            warnings.warn('missing target, no supervised training possible')

        # settings
        duration = self.inputs.shape[1]
        iter_duration = range(duration)  # to loop 
        loss_flag = self.checklist['target']  # compute loss
        trainable = self.substrate.is_trainable() 
        backward_flag = trainable * training  # backpropagate

        if verbose:
            print('\n### Simulation ###\n')
            print(f'duration: {duration}ms')
            print(f'compute loss : {bool(loss_flag)}')
            print(f'update: {bool(backward_flag)}')

            # tqdm progress bar
            iter_duration = tqdm(iter_duration)

        # stimuli of interest
        nb_inputs = len(self.inputs)
        if self.checklist['target']:
            nb_targets = len(self.targets)
        else:
            nb_targets = 0

        nb_outputs = self.substrate.get_nb_output()
        nb_stimuli = nb_inputs + nb_targets + nb_outputs

        if verbose:
            print('\n--- start training  ---')
            

        # record
        activity = np.zeros((nb_stimuli, duration))
        nb_metrics = self.substrate.get_nb_trainable()
        metrics = np.zeros((nb_metrics, duration))
        loss = 0

        for ms in iter_duration:

            # collect input
            self.substrate.collect_input(inputs=self.inputs[:, ms].T)

            # agent step
            self.substrate.step()

            # loss pass
            if loss_flag:
                
                # loss
                error = self.targets[:, ms] - self.substrate.get_output()
                loss += (error ** 2).sum().item()

                # backward pass
                if np.any(self.targets[:, ms] != 0) and backward_flag:
                    self.substrate.add_loss(backpropagated_loss=error)
            
            # trainable substrate
            if trainable and training:
                self.substrate.update()

            # record
            if plot_style is not None or return_:

                # track stimuli during epoch
                activity[:nb_inputs, ms] = self.inputs[:, ms].T
                if self.checklist['target']:
                    activity[nb_inputs: nb_inputs + nb_targets, 
                             ms] = self.targets[:, ms].T

                activity[nb_inputs + nb_targets:, ms] = self.substrate.get_output()
                
                if trainable:  # no loss record
                    metrics[:, ms] = self.substrate.get_trainable_params()

        if verbose:
            print('\n--- end training ---')
            print(f'\nloss: {loss / duration:.3f}')
        
        # record
        if self.sim_record is None:
            self.sim_record = {}
            idx = 0
        else:
            idx = len(self.sim_record.keys())
        self.sim_record[idx] = {'activity': activity[-nb_outputs:, :],
                                'metrics': metrics}

        # plot
        if plot_style is not None and self.stimuli_data['input']['kind'] != 'classes':

            x = range(duration)
            i, j, k = 0, -1, 0
            if plot_style == 'plot':

                for i in range(nb_inputs):
                    plt.plot(x, activity[i, :], label=f'input {i + 1}')

                for j in range(nb_targets):
                    plt.plot(x, activity[i + j + 1, :], label=f'target {j + 1}')

                for k in range(nb_outputs):
                    plt.plot(x, activity[i + j + k + 2, :], 
                             label=f'output {k + 1}')

                plt.xlabel('time [ms]')
                plt.legend()
                plt.title(f'Plot of the activity for {duration}ms')
                plt.show()
                
            elif plot_style == 'raster':
                self.plot_raster(show_target=0, spikes=activity[-nb_outputs:, :])

        if return_:
            return activity, metrics 

    def long_simulation(self, epochs=10, info_freq=1, plot_style=None, 
                        verbose=True, training=False, rigenerate=False,
                        early_stopping=True, return_=False):

        """
        simulation of all the epochs of activity
        
        Parameters
        ----------
        epochs: int
            number of epochs
        info_freq: int
            number of epochs after which an epoch-info are printed, default 1
        plot_style: str
            plotting style from (None, 'plot', 'raster'), default None
        verbose: bool
            print runtime information, default True
        training: bool
            if True the substrate is trained, default False
        rigenerate: bool
            if True the input\target are rigenerated at each epoch, default False
        
        Returns
        -------
        metrics : np.ndarray
            if True the simulation statistics are returned, default False
        """
        

        if not self.checklist['substrate']:
            raise NotImplementedError('missing substrate')

        elif not self.checklist['input']:
            raise NotImplementedError('missing input')

        elif not self.checklist['target']:
            warnings.warn('missing target, no supervised training possible')

        # settings
        duration = self.inputs.shape[1]
        loss_flag = self.checklist['target']  # compute loss
        trainable = self.substrate.is_trainable()
        backward_flag = trainable * training  # backpropagate
        

        if verbose:
            print('\n### Long Simulation ###\n')
            print(f'duration: {duration} ms')
            print(f'compute loss : {bool(loss_flag)}')
            print(f'backward pass: {bool(backward_flag)}')

        # stimuli of interest
        nb_inputs = len(self.inputs)
        if self.checklist['target']:
            nb_targets = len(self.targets)
        else:
            nb_targets = 0

        nb_output = self.substrate.get_nb_output()
        nb_stimuli = nb_inputs + nb_targets + nb_output

        # record metrics : trainable parameters + loss
        nb_metrics = self.substrate.get_nb_trainable() + int(training)
        metrics = np.zeros((nb_metrics, epochs))
        trainable_names = self.substrate.get_trainable_names()

        activity_zero = np.zeros((1, duration))

        # early stopping
        done = False
        error = 0.
        activity = np.ndarray
        
        if verbose:
            print('\n--- start training ---')

        for epoch in range(epochs):
                        
            # new data
            if rigenerate:
                self.rigenerate_data(new_input=1, new_target=1)

            # record
            activity = np.zeros((nb_stimuli, duration))
            
            if self.substrate.verbose:
                print('\n', epoch,  '-'*33)

            for ms in range(duration):
                
                #print(ms, 'ms')
                # collect input
                self.substrate.collect_input(inputs=self.inputs[:, ms])

                # agent steps
                self.substrate.step()

                # loss pass
                if loss_flag:

                    # loss
                    target = self.targets[:, ms]
                    error = target - self.substrate.get_output()
                    
                    if self.substrate.verbose:
                        print(f'\ny: {self.targets[:, ms]}')

                    # backward pass
                    if np.all(self.targets[:, ms] != 0) and backward_flag:
                        self.substrate.add_loss(backpropagated_loss=error)
                
                # trainable substrate
                if trainable:
                    self.substrate.update()

                # record
                if (plot_style is not None or return_) and self.stimuli_data['input'][
                        'kind'] != 'classes':

                    # track stimuli during epoch
                    activity[:nb_inputs, ms] = self.inputs[:, ms].T
                    
                    if self.checklist['target']:
                        activity[nb_inputs: nb_inputs + nb_targets, 
                                 ms] = self.targets[:, ms].T

                    activity[nb_inputs + nb_targets:, ms] = self.substrate.get_output()

                    if epoch == 0:
                        activity_zero[0, ms] = self.substrate.get_output()

            # record metrics
            if loss_flag:
                metrics[0, epoch] = error.sum().item() ** 2

            if backward_flag:
                metrics[1:, epoch] = self.substrate.get_trainable_params()
                            
            if epoch % info_freq == 0 and verbose:
                print(f'\nEpoch {epoch} | {epochs}', end=' ')

                if loss_flag:
                    print(f' - loss: {error.sum().item() ** 2:.4f}')

                    if early_stopping:
                        done = (error.sum().item() ** 2) == 0

            if done:
                break

        if verbose:
            print('\n\n--- end training ---')
            

        #######################################################################

        ### PLOT ACTIVITY ###
        
        if plot_style is not None:
            

            # activity
            x = range(duration)
            i, j, k = 0, -1, 0
            
            ### simple plot
            if plot_style == 'plot':
                
                # not classes data kind
                if not self.stimuli_data['input']['kind'] == 'classes':

                    plt.figure()
                    plt.plot(x, activity_zero[0, :], '--g', alpha=0.2, 
                             label='output epoch 0')
    
                    for i in range(nb_inputs):
                        plt.plot(x, activity[i, :], '-k', alpha=0.5, 
                                 label=f'input {i + 1}')
    
                    for j in range(nb_targets):
                        plt.plot(x, activity[i + j + 1, :], '-', color='orange', 
                                 alpha=0.5, label=f'target {j + 1}')
    
                    for k in range(nb_output):
                        plt.plot(x, activity[i + j + k + 2, :], '-g', 
                                 alpha=0.85, label=f'output {k + 1}')
    
                    plt.ylim((0, 1))
                    plt.xlabel('time [ms]')
                    plt.legend()
                    plt.title(f'Plot of the activity for {duration}ms')

            ### raster plot
            elif plot_style == 'raster':
                
                self.plot_raster(show_target=1, spikes=activity[-nb_output:, :])
                
                
            ### PLOT METRICS ###
            
            plt.figure()
            plt.subplot(nb_metrics, 1, 1)
            plt.plot(range(epochs), metrics[0, :], '-r', label='loss')
            plt.ylim((0, max(metrics[0, :])))
            plt.xlabel('epochs')
            plt.legend(loc='upper right')
            plt.title('Training metrics')

            for n in range(nb_metrics - 1):
                plt.subplot(nb_metrics, 1, n + 2)
                plt.plot(range(epochs), metrics[n + 1, :], '-k',
                         label=f'param "{trainable_names[n]}"')
                plt.xlabel('epochs')
                plt.legend(loc='upper right')                
    
            plt.show()

        if return_:
            return metrics 

    def test_substrate(self):
        
        """
        test the substrate
        
        Returns
        -------
        None
        """
        
        if not isinstance(self.substrate, object):
            raise ValueError(f'{self.substrate} is not testable ')
        
        # create candidate
        candidate = AgentEvo(substrate=self.substrate, name='candidate',
                         gen=0, kind='new')
        
        # test
        self.testing(candidate=candidate, verbose=True, return_obj=False,
                     plotting=(True))        
        
    def plot_classes_test(self):
        
        """
        for stimuli of classes "classes", plot the performance of the substrate
        
        Returns
        -------
        None
        """
        
        # inputs
        nb = self.stimuli_data['input']['nb']
        duration = self.stimuli_data['input']['duration']
        nb_classes =  self.stimuli_data['input']['nb_classes']
                
        # output
        outputs = np.zeros((nb, duration))
        for ms in range(duration):

            # collect input
            self.substrate.collect_input(inputs=self.inputs[:, ms].T)

            # agent step
            self.substrate.step()

            outputs[:, ms] = self.substrate.get_output()
            
        # plot
        for i in range(nb):
            
            plt.scatter(self.inputs[i], self.targets[i], color='black', alpha=0.5, label='stimuli')
            plt.plot(self.inputs[i], outputs[i], '^g', alpha=0.5, label='output')
    
        plt.xlabel('input')
        plt.xticks(np.arange(1, nb_classes+1)/nb_classes, np.around(np.arange(1, nb_classes+1)/nb_classes, 2))
        plt.ylabel('target')
        plt.yticks(np.arange(1, nb_classes+1)/nb_classes, np.around(np.arange(1, nb_classes+1)/nb_classes, 2))
        plt.title(f'{nb_classes} classes of stimuli')
        plt.legend()
        plt.show()

    def get_input(self):

        """
        Returns
        -------
        inputs : np.ndarray
            shape=(nb_inputs, duration)
        stimuli_data : dict 
            input stimulation information
        checklist : bool 
            True if inputs have been loaded, default False
        """

        return self.inputs, self.stimuli_data['input'], self.checklist['input']

    def get_target(self):

        """
        Returns
        -------
        targets : np.ndarray
            shape=(nb_targets, duration)
        stimuli_data : dict 
            target stimulation information
        checklist : bool 
            True if targets have been loaded, default False
        """

        return self.targets, self.stimuli_data['target'], self.checklist['target']


########### AGENTS ###########

class AgentEvo:
    
    """ Agent class for evolving substrates """
    
    def __init__(self, substrate: list, name: str, gen: int, kind: str):
        
        """
        :param substrate: class.Substrate object
        :param name: str
        :param gen: int, generation of the agent
        :param kind: str, kind of individual
        :return: None
        """
        
        self.substrate = substrate
        self.name = name
        self.gen = gen
        self.kind = kind
        
        
        self.output = 0.
        self.fitness = 0.
        
    def step(self):
        
        """
        step the dynamics
        :return: None
        """
        
        self.substrate.step()
        self.output = self.substrate.get_output()
        
    def collect_input(self, inputs: np.ndarray):
        
        """
        receive an external input
        :param inputs: ndarray
        :return: None
        """
        
        self.substrate.collect_input(inputs=inputs)
    
    def get_output(self):
        
        """
        :return: output, np.ndarray
        """
        
        return self.output
    
    def get_fitness(self):
        
        """
        :return: fitness, float
        """
        
        return self.fitness
    
    def get_dna(self):
        
        """
        :return: list
        """
        
        return self.substrate.get_dna()
    
    def get_substrate(self):
        
        """
        :return: class.Substrate object
        """
        
        return self.substrate
    
    def get_info(self):
        
        """
        :return: info, dict, information and status of the agent
        """
        
        return {'name': self.name,
                'gen': self.gen,
                'kind': self.kind,
                'fitness': self.fitness,
                'dna': self.substrate.get_dna()}
    
    def get_name(self):
        
        """
        :return: str, name
        """
        
        return self.name
        
    def set_fitness(self, fitness: float):
        
        """
        set a new fitness value
        :param fitness: float
        :return: None
        """
        
        self.fitness = fitness
        
    def reset(self):
        
        """
        reset of the agent variables
        :return: None
        """
        
        # reset substrate
        self.substrate.reset()
        
        # agent variables
        self.output = 0.
        self.fitness = 0.
        
                
########### DNA GENERATOR ###########
            
  
class DNA_generator:
    
    """ generate and manipulate dna and produce a functional Substrate object """
    
    def __init__(self, dna_params: dict, substrate_name: str):
        
        """
        :param dna_params, dict, parameters for building a dna
        :param substrate_name, str, name of the substrate to build
        :return: None
        """
        
        self.params = dna_params
        self.substrate_name = substrate_name
        self.dna = [substrate_name]
        
        # mutations
        self.mutation_rate = lambda: np.random.normal(1, 
                                    dna_params['settings']['mutation_std'])
        self.mutation_prob = lambda: np.random.binomial(1, 
                                    dna_params['settings']['mutation_prob'])
        
    def sample_protein(self, variety_pool: tuple, probabilities: tuple):
        
        """
        :param variety_pool: tuple, set of protein varieties to sample from
        :param probabilities: tuple, set of probabilities for each protein 
        variety to be picked
        :return: dict, the protein picked
        """
        
        variety = np.random.choice(variety_pool, p=probabilities)
        
        if variety == 'exp':
            
            protein_content = {'variety': variety,
                               'attributes': {'tau': self.params['substrate'][
                                   'protein']['variety'][variety]['attributes'][
                                       'tau'](),
                                              'Eq': self.params['substrate'][
                                                  'protein']['variety'][
                                                variety]['attributes']['Eq']()
                                              },
                               'more': {'trainable_params': [],
                                        'activation': self.params['substrate'][
                                            'protein']['variety'][variety][
                                                'more']['activation']()}
                               }
                       
            
        elif variety == 'base':
            
            protein_content = {'variety': variety,
                               'attributes': {},
                               'more': {'trainable_params': [],
                                        'activation': self.params['substrate'][
                                            'protein']['variety'][variety][
                                                'more']['activation']()}
                               }
                       
        return ('Protein', protein_content)

    def new_dna(self):
        
        """
        build a new complete dna
        :return: None
        """
        
        
        if self.substrate_name == 'Protein':
            
            # get hyperparameters
            variety_pool = tuple(self.params['substrate']['protein']['variety'].keys())
            probabilities = self.params['substrate']['protein']['probabilities']
            self.dna += [self.sample_protein(variety_pool, probabilities)]
    
        elif self.substrate_name == 'Cell':
                        
            # get hyperparameters
            variety_pool = tuple(self.params['substrate']['protein']['variety'].keys())
            probabilities = self.params['substrate']['protein']['probabilities']
            nb_proteins = self.params['substrate']['protein']['n']()
            
            # define components
            dna_content = {'components': [self.sample_protein(variety_pool=variety_pool,
                                                       probabilities=probabilities)
                                   for _ in range(nb_proteins)]
                           }

            # cell params
            dna_content['more'] = self.params['substrate']['cell']['more']
            dna_content['more']['trainable_params'] = []
            
            # probably too harsh computing the cycles this way
            dna_content['more']['cycles'] = round(np.random.normal(nb_proteins, 
                                                                   1))  
            
            # define connections
            connections = [(0, 1)]
            nb_conn = round(np.random.normal(nb_proteins, 2))            
            for _ in range(nb_conn):
                i, j = np.random.choice(range(nb_proteins + 1), replace=False, 
                                        size=2)
                
                # avoid backward connections to the input source
                if i == 0:
                    continue
                
                # avoid repetition of connections
                if (i, j) not in connections:
                    connections += [(j, i)]    
                
            dna_content['connections'] = connections
            dna_content['attributes'] = {}
            
            # merge
            self.dna += [dna_content]
            
        
        # finalize
        self.dna = tuple(self.dna)
        
    def generate(self, kind='new', dna1=False, dna2=False):
        
        """
        handle the request of a new dna
        :param kind, str, type of dna to build from ("new", "mutant", "crossed"),
        default "new"
        :param dna1, list, parent dna required for kind "mutant" and "crossed",
        default False
        :param dna2, list, second parent dna required for kind "crossed",
        default False
        :return: class.Substrate object
        """
        
        # reset
        self.dna = [self.substrate_name]
        
        if kind == 'new':
            
            self.new_dna()
            
        elif kind == 'mutant':
            
            if not dna1:
                warnings.warn('parent dna not provided, returning a new dna')
                self.new_dna()
                
            else:
                self.mutate(dna=dna1)
                
        elif kind == 'crossed':
            
            if not dna1 or not dna2:
                warnings.warn('one or more parents dna not provided, ',
                              'returning a new dna')
                self.new_dna()
                
            else:
                self.cross(dna1=dna1, dna2=dna2)
                
        else:
            raise ValueError(f'kind "{kind}" is invalid, allowed are "new", ',
                             '"mutant", "crossed"')
                
        
        return brain.generate_substrate(dna=self.dna, verbose=False)
                
    def mutate(self, dna: list):
        
        """
        mutate a parent dna and produce a new mutant one
        :param dna, list
        :return: None
        """
                
        # mutate proteins
        if dna[0] == 'Protein':
            
            dna = self.mutate_protein(protein_gene=dna)
            
        # mutate cell
        elif dna[0] == 'Cell':
            
            # mutate_components
            new_components = []
            
            for protein_gene in dna[1]['components']:
                
                if self.mutation_prob():
                    new_components += [self.mutate_protein(protein_gene=protein_gene)]
                    continue
                
                new_components += [protein_gene]
                
            dna[1]['components'] = new_components
            
        # update
        self.dna = dna
                
    def mutate_protein(self, protein_gene: list):
        
        """
        mutate a protein given its corresponding dna dict
        :param protein_gene: list, protein dna
        :return: list, the mutated protein dna
        """
    
                    
        # assuming only one protein kind
        if protein_gene[1]['variety'] == 'exp':
            
            if self.mutation_prob():
                protein_gene[1]['attributes']['tau'] *= self.mutation_rate()
                
            if self.mutation_prob():
                
                # if Eq = 0 -> 1, if Eq = 1 -> 0
                protein_gene[1]['attributes']['Eq'] =  \
                    -1 * protein_gene[1]['attributes']['Eq'] + 1
    
            protein_gene[1]['attributes']['w'] = np.array(protein_gene[1]
                                ['attributes']['w']) * self.mutation_rate()
                
        elif protein_gene[1]['variety'] == 'base':
            
            if self.mutation_prob():
                protein_gene[1]['attributes']['w'] = np.array(protein_gene[1]
                                    ['attributes']['w']) * self.mutation_rate()
                
            if self.mutation_prob():
                protein_gene[1]['attributes']['bias'] *= self.mutation_rate()
                
        else:
            warnings.warn('can not mutate protein ',
                          f'"{protein_gene[1]["variety"]}"')
            
        return protein_gene
                
    def cross(self, dna1: list, dna2: list):
        
        """
        apply crossing over to two parents dna
        :param dna1, first parent dna
        :param dna2, second parent dna
        :return: None
        """
        
        self.new_dna()  # <--- TO EDIT
    
    def get_dna(self):
        
        """
        :return: list, last dna produced
        """
            
        return self.dna
    

########### EVOLUTION ###########
    
class SimpleEvolution:
    
    """ simple genetic algorithm to evolve DNA on a defined stimulation """
    
    def __init__(self, testmap: object, proportions: dict, dna_generator: object, verbose=True):
        
        """
        :param testmap: class.TestMap object
        :param proportions: dict, dictionary of values in [0, 1] indicating the
        proportion in the population of different kind of individuals, keys
        are : {'nb_new': x, 'nb_mut': x, 'nb_cro': x, 'nb_cop': x}
        :param dna_generator: class.DNA_generator object
        :param verbose: bool, default True
        :return: None
        """
        
        # params
        self.nb_new = proportions['nb_new']
        self.nb_mut = proportions['nb_mut']
        self.nb_cro = proportions['nb_cro']
        self.nb_cop = proportions['nb_cop']
        self.n = self.nb_new + self.nb_mut + self.nb_cro + self.nb_cop + 2
        self.gen = -1
        self.mutation_params = {}
        self.core_density = 0
        
        self.verbose = verbose
        
        # game
        if not testmap.is_complete():
            raise ValueError(f'testmap is incomplete [checklist: {testmap.checklist}]')
        self.testmap = testmap
        
        # dna generator
        self.dna_generator = dna_generator
        
        # var
        self.population = 0
        self.fittests = ()
        
        # record
        self.curr_fitness = 0.
        
        print('\n@SimpleEvolution')      
        
    def evolve(self, epochs=10, reset=False):
        
        """
        evolve the popoulation as a cycle of new generation and fitting
        :param epochs: int, number of epochs to run (i.e. generations), default 10
        :param reset: bool, if True the generation restart from 0, default False
        :return: None
        """
        
        print(f'\n------------- evolving [{epochs}] ---------------\n')
        
        if reset:
            self.gen = -1
        
        for epoch in range(epochs):
            
            self.new_generation()
            
            self.fit()
            
            self.record()
            
        print('\n', '----------------------------------\n')
                
    def new_generation(self):
        
        """
        building of a new population
        :return: None
        """
        
        self.gen += 1
        
        # generation zero
        if self.gen == 0:
            
            self.population = [AgentEvo(substrate=self.dna_generator.generate(kind='new'),
                                        name=self.random_name(gen=0, kind='new'),
                                        gen=0,
                                        kind='new') for _ in range(self.n)]
            
            
        # other generations
        else:
            
            self.fittests[0].reset()
            self.fittests[1].reset()
            
            # fittests
            new_population = [self.fittests[0], self.fittests[1]]
            
            # copies
            new_population += [AgentEvo(substrate=self.fittests[0].get_substrate(),
                                        name=self.fittests[0].get_name(),
                                        gen=self.gen,
                                        kind='copy') for _ in range(self.nb_cop)]
            
            # new
            self.population += [AgentEvo(substrate=self.dna_generator.generate(kind='new'),
                                        name=self.random_name(gen=self.gen, 
                                                              kind='new'),
                                        gen=self.gen,
                                        kind='new') for _ in range(self.nb_new)]
            
            # mutant
            self.population += [AgentEvo(substrate=self.dna_generator.generate(kind='mutant', 
                                                                                    dna1=self.fittests[0].get_dna()),
                                        name=self.inherit_name(name=self.fittests[0].get_name(),
                                                               gen=self.gen, 
                                                               tag='mut'),
                                        gen=self.gen,
                                        kind='mutant') for _ in range(self.nb_mut)]
            
            # crossing over
            self.population += [AgentEvo(substrate=self.dna_generator.generate(kind='new',
                                                                                    dna1=self.fittests[0].get_dna(),
                                                                                    dna2=self.fittests[1].get_dna()),
                                        name=self.inherit_name(name=self.fittests[0].get_name(),
                                                               gen=self.gen, 
                                                               tag='cro'),
                                        gen=0,
                                        kind='crossed') for _ in range(self.nb_cro)]
        
        if self.verbose:
            print(f'\n### generation {self.gen} ###')
    
    def fit(self):
        
        """
        fit population to testmap  | edit to multiprocessing
        :return: None
        """
        
        self.population = self.shell(batch=self.population)
        
    def shell(self, batch: list):
        
        """
        fit a batch of agents over the game, run sequentially on one core
        :param batch: list, set of agents 
        :return: list, set of fitted agents in order of fitness
        """
        
        fitted_batch = []
        for candidate in batch:
            fitted_batch += [self.testmap.testing(candidate=candidate)]
            
        return sorted(fitted_batch, key=lambda x: x.fitness, reverse=True)
    
    def record(self):
        
        """
        record fitness and fittests agents [2]
        :return: None
        """
        
        self.fittests = self.population[:2]
        self.curr_fitness = self.population[0].get_fitness()
        
        if self.verbose:
            print(f'\n- fittest: {self.fittests[0].get_name()} [{self.fittests[0].get_fitness():.3f}]')
    
    def get_fitted_dna(self):
        
        """
        :return: dict, information about the fittest
        """
        
        return self.fittest.get_info()
        
    @staticmethod
    def random_name(gen: int, kind: str):
        
        """
        generate a random name
        :param gen: int
        :param kind: str
        :return: str, name
        """
        
        vowels = 'a e i o u'.split(' ')
        consonants = 'b c d f g h j k l m n p q r s t v w x y z'.split(' ')
        
        name = f'{str(gen)}_' + "".join([np.random.choice(vowels) if np.random.random() > 0 else 
                        np.random.choice(consonants) for _ in range(5)]) + f'_{kind}'
            
        return name
    
    @staticmethod
    def inherit_name(self, name: str, gen: int, tag: str):
        
        """
        generate a new name given a base one and updating its generation and tag
        :param name: str,
        :param gen: int,
        :param tag: str,
        :return: str
        """
        
        _, midname, _ = name.split('_')
        
        return f'{str(gen)}_' + midname + f'_{tag}'
    
    def show_fittest(self):
        
        """
        show the performance of the fittest
        :return: None
        """
        
        self.testmap.testing(candidate=self.fittests[0], return_obj=0, plotting=1)
        print(f'name: {self.fittests[0].get_name()}')
        print(f'\nscore: {self.fittests[0].get_fitness():.4f}\n\ndna: {self.fittests[0].get_dna()}')
    

if __name__ == '__main__':

    print('%sim')
