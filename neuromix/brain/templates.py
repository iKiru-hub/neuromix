import numpy as np

import networkx as nx
import pprint

import warnings 


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)



def set_random_seed(seed: int):

    """
    set the random seed for numpy and random
    
    Parameters
    ----------
    seed : int
    
    Returns
    -------
    None
    """

    global RANDOM_SEED
    RANDOM_SEED = seed
    np.random.seed(seed)

    print(f"-- Set {RANDOM_SEED=} --")

def get_random_seed():
    
    """
    get the random seed
    
    Returns
    -------
    RANDOM_SEED : int
    """

    return RANDOM_SEED

def generate_unique_id(size=7):

    """
    generate a unique id for the Substrate
    
    Parameters
    ----------
    size : int [optional]
        length of the unique id, by default 7

    Returns
    -------
    unique_id : int
    """

    symbols = [chr(l) for l in range(65, 91)] + [str(i) for i in range(10)]
    return "".join(np.random.choice(symbols, size=size).tolist())

def print_dna(dna: dict, depth=5):

    """
    print the DNA of a Substrate
    
    Parameters
    ----------
    dna : dict
    depth : int [optional]
        depth of the print, by default 5

    Returns
    -------
    None
    """
    
    pretty_printer = pprint.PrettyPrinter(depth=depth)
    pretty_printer.pprint(dna)


#### ACTIVATION FUNCTIONS ####

activation_functions = {'sigmoid': (lambda x: 1 / (1 + np.exp(-x)),
                                    lambda x: (1 / (1 + np.exp(-x))) * \
                                        (1 - (1 / (1 + np.exp(-x))))),
                        'sigMod': (lambda x: 1 / (1 + np.exp(-x/0.09 + 6)),
                                   lambda x: (1 / (1 + np.exp(-x/0.09 + 6))) * \
                                       (1 - (1 / (1 + np.exp(-x/0.09 + 6))))),
                        'softmax': (lambda x: np.exp(x) / np.sum(np.exp(x)),
                                    None),
                        'relu': (lambda x: x * (x > 0),
                                 lambda x: 1 * (x > 0)),
                        'crelu': (lambda x: min((x * (x > 0), 1)),
                                  lambda x: 1 * (x > 0)),
                        'step': (lambda x: 1 * (x > 0),
                                   lambda x: 1),
                        'none': (lambda x: x,
                                 lambda x: 1)
                        }

# derivative of the softmax function as lambda expression 


#### SUBSTRATE ####

class Substrate:

    """
    base Substrate class:
        an object endowed with internal dynamics and input-output channels,
        whose definition is encoded in a DNA
    """

    def __init__(self, dna: dict, verbose=False):

        """
        Parameters
        ----------
        dna: dict 

        Returns 
        ------- 
        None 
        """

        # DNA
        self.DNA = dna
        self.substrate_class = 'Substrate'
        self.substrate_family = 'root'
        self.substrate_id = '0'

        # 'I' for input, 'O' for output, 'H' for hidden, '.' for undefined
        self.substrate_role = "."
        self.unique_id = generate_unique_id()
        self.index = 0

        # hyper-parameters
        self._lr = 0.

        # numbers
        self.nb_inputs = 0
        
        # training
        self.trainable_names = []
        self.nb_trainable = 0
        self.trainable_params = np.zeros(0)
        self.trainable = False
        self.backprop_enabled = False

        self._original_params = {}

        # variables
        self._ext_inputs = None
        self.activity = 0.
       
        # backward
        self._feedback = np.zeros((1, 1))
        
        # initialization
        self.initialization_flag = False

        # check 
        self._substrate_initialization()
        
        self.verbose = verbose
        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}.{self.substrate_role}', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()

    def _substrate_initialization(self):

        """
        minimal DNA check of the class.Substrate to set hyperparameters like the learning rate [lr]

        Returns
        -------
        None
        """

        ### KEY CHECK ###

        # check presence of general keys
        keys = tuple(self.DNA.keys())
        assert 'params' in keys, "missing general key 'params' in DNA"
        assert isinstance(self.DNA['params'], dict), "general key 'params' must be a dict"
        assert 'attrb' in keys, "missing general key 'attrb' in DNA"
        assert isinstance(self.DNA['attrb'], dict), "general key 'attrb' must be a dict"

        print('training params: ', self.DNA['attrb'])
        # set trainable params if present 
        if 'trainable_params' in self.DNA['attrb'].keys():

            # check 
            assert isinstance(self.DNA['attrb']['trainable_params'], list), \
                "trainable_params must be a list of strings"
            for param in self.DNA['attrb']['trainable_params']:
                self.trainable_names += [param]
        else:
            self.DNA['attrb']['trainable_params'] = []
            self.trainable_names = []

        self.nb_trainable = len(self.trainable_names)
        print(f'Substrate - nb_trainable: {self.nb_trainable}')

        # set backrpopagation flag
        if 'backprop' in self.DNA['attrb'].keys():
            self.backprop_enabled = self.DNA['attrb']['backprop']
        else:
            self.DNA['attrb']['backprop'] = False

        # if the Substrate is trainable, initialize the trainable params container
        if self.nb_trainable > 0:
            self.trainable_params = np.zeros(self.nb_trainable)
            self.trainable = True

        # optionally available <more> keys
        self._lr = self.DNA['attrb']['lr'] if 'lr' in self.DNA['attrb'].keys() else 0.

        # idendfiers
        if 'more' not in self.DNA.keys():
            self.DNA['more'] = {}
        self.DNA['more']['idx'] = self.index
        self.DNA['more']['uid'] = self.unique_id
        self.DNA['more']['role'] = self.substrate_role

        ### INITIALIZATION ###
        self.nb_inputs = self.DNA['attrb']['nb_inp'] if 'nb_inp' in self.DNA['attrb'].keys() else 1
        self._ext_inputs = np.zeros((self.nb_inputs, 1))

    def _update_substrate_dna(self):
        
        """
        update the dna with the new parameters
        Returns
        -------
        None
        """
       
        self.DNA['attrb']['trainable_params'] = self.trainable_names
        self.DNA['attrb']['lr'] = self._lr
        self.DNA['attrb']['backprop'] = self.backprop_enabled
        self.DNA['attrb']['nb_inp'] = self.nb_inputs
    
    def step(self):

        """
        receive an input and the state is updated [to edit]
        
        Returns
        -------
        None
        """

        self.activity = self._ext_inputs.copy()

    def update(self):

        """
        the trainable parameters are updated [to edit]
        
        Returns
        -------
        None
        """

        self._feedback *= 0

    def collect_input(self, inputs: np.ndarray):
        
        """
        receive and store the inputs
        
        Parameters
        ----------
        inputs : np.ndarray
        
        Returns
        -------
        None
        """

        self._ext_inputs = inputs.copy() if isinstance(inputs, np.ndarray) else inputs

    def add_feedback(self, ext_feedback: np.ndarray):
        
        """
        record an external feedback
        
        Parameters
        ----------
        ext_feedback : np.ndarray
            external feedback
        
        Returns 
        -------
        None
        """

        self._feedback = ext_feedback
    
    def add_idx(self, idx: int):

        """
        add the index of the Substrate
        
        Parameters 
        ----------
        idx : int
            id of the component
        
        Returns 
        -------
        None
        """

        self.index = idx

    def set_id(self, _id: str):

        """
        set the id of the dna used 

        Parameters
        ----------
        _id : str 

        Returns
        -------
        None
        """

        self.substrate_id = _id

    def set_role(self, role: str):

        """
        set the role of the Substrate

        Parameters
        ----------
        role : str

        Returns
        -------
        None
        """

        self.substrate_role = role
        self.DNA['more']['role'] = role

    def set_backprop(self, flag: bool):
    
            """
            set the backpropagation flag
    
            Parameters
            ----------
            flag : bool
    
            Returns
            -------
            None
            """
    
            self.backprop_enabled = flag

    def get_loss(self) -> float:
        

        """
        compute and return the loss at this node
        
        Returns
        -------
        loss : np.ndarray
        """

        return self._feedback

    def get_output(self):

        """
        Returns
        -------
        output : float
            default activity
        """

        return self.activity
    
    def get_trainable_params(self):

        """
        [to edit]
        
        Returns
        -------
        None
        """

        return self.trainable_params
    
    def get_trainable_names(self):

        """
        Returns
        -------
        trainable_parameters : list
            names of the trainable parameters
        """

        return self.trainable_names

    def get_nb_trainable(self):

        """
        Returns 
        -------
        trainable_params : int
            number of trainable params
        """

        return self.nb_trainable

    def get_nb_output(self):

        """
        Returns
        -------
        number_of_output : int
            default 1 
        """

        return 1
    
    def is_trainable(self):

        """
        Returns 
        -------
        is_trainble : bool
            trainable molecule or not
        """

        return self.trainable

    def is_backprop(self):
    
            """
            Returns
            -------
            is_backprop : bool
                backpropagation flag
            """
    
            return self.backprop_enable

    def get_substrate_identity(self, full=False):

        """
        Returns
        -------
        substrate_identity : dict
        """
        if full:
            return {'class': self.substrate_class, 'family': self.substrate_family, 'id': self.substrate_id, 'uid': self.unique_id, 'role': self.substrate_role, 'idx': self.index}

        return {'class': self.substrate_class, 'family': self.substrate_family, 'id': self.substrate_id}

    def print_substrate_identity(self, return_str=False):

        """
        Returns
        -------
        substrate_identity : str
        """

        if not return_str:
            print(f"{self.substrate_class}.{self.substrate_family}.{self.substrate_id}", end='')
            print(f", uid : {self.unique_id}, role: {self.substrate_role}, idx :{self.index}")
    
        return f"{self.substrate_class}.{self.substrate_family}.{self.substrate_id}" + \
                f", uid : {self.unique_id}, role: {self.substrate_role}, idx :{self.index}"

    def get_unique_id(self):

        """
        Returns
        -------
        substrate_unique_id : str
        """

        return self.unique_id

    def get_role(self):
        
        """
        Returns
        -------
        role : str
        """

        return self.substrate_role

    def get_nb_inout(self):

        """
        Returns
        -------
        nb_inputs : int
        """

        return self.nb_inputs

    def get_dna(self, show=False, full=False, depth=5):
        
        """
        update the DNA with the current parameters and returns it

        Parameters
        ----------
        show : bool
            print the DNA
        full : bool
            if True, print the full DNA as [$substrate_class, [dna]]
        depth : int
            depth of the print
        
        Returns
        -------
        DNA : dict
        """
        
        self._update_substrate_dna()

        if show:
            print(f"{self.substrate_class}.{self.substrate_family}.{self.substrate_id}.{self.substrate_role}")
            print_dna(dna=self.DNA, depth=depth)
            return

        if full:
            return [self.substrate_class, self.DNA]
        return self.DNA
    
    def is_initialized(self):
        
        """
        Returns
        -------
        bool
        """
        
        return self.initialization_flag

    def reset(self):

        """
        reset the value of the variables [to edit]
        Returns
        -------
        None
        """

        self._ext_inputs *= 0
        self.activity *= 0.


#### SUBSTRATE STRUCTURE ####

class SubstrateStructure(Substrate):
    
    """ base SubstrateStructure class
    
    a Substrate object having a structure made of more elementary substrates, 
    its components, which made up the internal dynamics (and input-output) of 
    the object through pre-defined connections 
    """

    def __init__(self, dna: dict, built_components: list, verbose=False):

        """
        Parameters 
        ----------
        dna: dict
        built_components: list,
            list of already built components
        verbose: bool
        
        Returns
        -------
        None
        """

        # set up
        print('-Structure-')
        Substrate.__init__(self, dna=dna)
        self.substrate_class = 'Structure'
        
        # hyperparams
        self._cycles = 1
        
        # structure
        self.components = None 
        self.connections = None
        self.connectivity_matrix = np.zeros(0)
        self.nb_components = 0
        self.idx_out = 0
        self.nb_outputs = 0
        
        self.trainable_components = []
        
        # graph
        self.graph = nx.DiGraph()
        self.graph_color = []
        
        # variables
        self.activity = np.zeros(self.nb_inputs + self.nb_components)
        self.output = np.zeros(self.nb_outputs)

        # check 
        self._substrate_structure_initialization()
        self._build_structure(built_components=built_components)
        self._update_structure_dna()
        self._build_graph()

        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()

    
            print(f'\nnb_components: {self.nb_components}\nnb_inputs: '
                  f'{self.nb_input}\nnb_outputs: {self.nb_output}'
                  f'\nnb_trainable: {self.nb_trainable}')

    def _substrate_structure_initialization(self):

        """
        minimal DNA check of the class.SubstrateStructure and initialization of \
        some parameters 

        Returns
        -------
        None
        """

        # Structure keys check  
        structure_keys = tuple(self.DNA.keys())

        assert 'components' in structure_keys, "missing key 'components'"
        assert isinstance(self.DNA['components'], list), \
            "'components' must be a list"
        assert 'connections' in structure_keys, "missing key 'connections'"

        self.connections = self.DNA['connections']

        # attributes_keys check 
        attributes_keys = tuple(self.DNA['attrb'].keys())
        assert 'idx_out' in attributes_keys, "missing attribute 'idx_out'"
        assert isinstance(self.DNA['attrb']['idx_out'], list), "attribute 'idx_out' must be a list"
        assert 'nb_out' in attributes_keys, "missing attribute 'nb_out'"
        assert 'cycles' in attributes_keys, "missing attribute 'cycles'"
        
        self.idx_out = self.DNA['attrb']['idx_out']
        self.nb_outputs = self.DNA['attrb']['nb_out']
        self._cycles = self.DNA['attrb']['cycles']

        # 
        self.initialization_flag = True

    def _build_structure(self, built_components: list):

        """
        create each component from the DNA and store them in a list

        Parameters
        ----------
        built_components: list,
            list of already built components
        
        Returns
        -------
        None
        """

        # check 
        assert isinstance(built_components, list), "built_components must be a list"
        
        self.components = []

        # alphabet for the parameters name
        alphabet = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 'k')

        # register each already built component
        for idx, a_component in enumerate(built_components):

            self.components += [a_component]
            
            # register trainable components
            if a_component.is_trainable():
                self.trainable_components += [idx]

        # trainable
        self.trainable = self.trainable_components.__len__() > 0

        # number of inputs and components
        self.nb_components = len(self.components)  # components
        tot = len(self.components) + self.nb_inputs   # total

        # build a connectivity matrix
        self.connectivity_matrix = np.zeros((tot, tot))

        # NB: the number of external inputs shall be considered; e.g. one input has index 0
        for j, i in self.connections:
            if i >= tot:
                raise ValueError(f'source index {i} is too big')
            if j >= tot:
                raise ValueError(f'sink index {j} is too big')
            self.connectivity_matrix[i, j] = 1

        # define input indices
        self.idx_inp = []
        for input_row in (self.connectivity_matrix.T)[:self.nb_inputs]:
            self.idx_inp += (np.where(input_row == 1)[0] - self.nb_inputs).tolist()

        # unique input indices
        self.idx_inp = list(set(self.idx_inp))
        self.DNA['attrb']['idx_inp'] = self.idx_inp

        # initialize the _weights for each component
        nb_i, nb_h, nb_o = 1, 1, 1
        for i in range(self.nb_components):

            # re-initialize each component and add its index
            self.components[i].re_initialize(nb_inputs=int(self.connectivity_matrix[i + self.nb_inputs].sum()))
            self.components[i].add_idx(idx=i)

            # set the role of the component

            if i in self.idx_inp:
                self.components[i].set_role(role=f'I{nb_i}')
                nb_i += 1
            elif i in self.idx_out:
                self.components[i].set_role(role=f'O{nb_o}')
                nb_o += 1
            else:
                self.components[i].set_role(role=f'H{nb_h}')
                nb_h += 1

            # if the components is within the substrate structure's trainable parameters
            # book space in track for the tracking of params of the internal components
            self.nb_trainable += self.components[i].get_nb_trainable()
            
            # register the trainable parameters name
            param_values = self.components[i].get_trainable_names()
        
            # set name
            self.trainable_names += [f'c_{alphabet[k]}{i}' for k
                                     in range(len(param_values))]
        print(f'after built - nb trainable: {self.nb_trainable}')

        # adjust 
        assert self.nb_trainable >= 0, 'negative number of trainable parameters'
            
        self.trainable_params = np.zeros(self.nb_trainable)  

        # var
        self.activity = np.zeros(self.nb_inputs + self.nb_components)
        self.output = np.zeros(self.nb_outputs)

    def _update_structure_dna(self):

        """
        returns possible warnings on the cell structure

        Returns
        -------
        None
        """

        # update DNA with each components' DNA
        components_gene = []
        for component in self.components:
            components_gene += [component.get_dna(full=True)]

        self.DNA['components'] = components_gene

        # update DNA
        self._update_substrate_dna()

        # warning
        if self.DNA['components'].__len__() == 0:
            warnings.warn('Empty Structure: no components detected')

        if self.DNA['connections'].__len__() == 0:
            warnings.warn('Disconnected components: no connections detected')

        if self.DNA['attrb']['nb_inp'] == 0:
            warnings.warn('Autonomous Structure: zero inputs specified')

        if self.DNA['attrb']['nb_out'] == 0:
            warnings.warn('Isolated Structure: zero outputs specified')

    def _build_graph(self):

        """
        build the graph of the structure

        Returns
        -------
        None
        """

        # define nodes with their names
        nodes = [f'E{i+1}' for i in range(self.nb_inputs)] + \
            [components.get_role() for components in self.components]

        # add nodes 
        for node in nodes:
            self.graph.add_node(node)

        for (i, j) in self.connections:
            self.graph.add_edge(nodes[i], nodes[j])

        # set colors 
        for node in self.graph.nodes:
            if node[0] == 'E':
                self.graph_color += ['grey']
            elif node[0] == 'I':
                self.graph_color += ['red']
            elif node[0] == 'O':
                self.graph_color += ['yellow']
            else:
                self.graph_color += ['orange']

    def step(self):

        """
        the internal components of the cell have their state updated

        Returns
        -------
        None
        """

        for cycle in range(self._cycles):

            # get the activity of each component, input to its downstream neighbours
            for idx, component in enumerate(self.components):
                self.activity[self.nb_inputs + idx] = component.get_output()

            # each component loads the inputs from its 
            # inputs sources + each component steps
            # print('\n-load and step-')
            for i in range(self.nb_components):

                # indexes of the inputs to component idx
                inputs_j = np.where(self.connectivity_matrix[i + self.nb_inputs] \
                                    != 0)[0]

                if len(inputs_j) == 0:
                    continue

                # load
                self.components[i].collect_input(
                    inputs=self.activity.take(inputs_j).T)

                # step
                self.components[i].step()
                
                
                # live graph
                if self.grapher and self.livestream:
                        
                    self.grapher.live_graph(activations=self.activity)
                

            # define the input as the activity of the output components
            #for idx, component in enumerate(self.components[-self.nb_outputs:]):
            #    self.output[idx] = component.get_output()
            self.output = self.activity[self.idx_out]

            # reset inputs
            try:
                self.activity *= 0
            except RuntimeWarning:
                print('runtime warning: ', self.activity)
                input()

    def update(self):

        """
        the trainable parameters are updated [to edit]
        
        Returns
        -------
        None
        """

        if self._feedback.sum() == 0:
            return

        # BACKPROPAGATION # [if backpropagation is enabled]
        if self.backprop_enabled:

            # loss at the output nodes
            for k in range(self.nb_outputs):
                self.components[-k-1].add_feedback(ext_feedback=self._feedback[k])

            # starting from the output nodes, loop over all the components
            for i in range(self.nb_components - 1, 0, - 1):

                loss_i = self.components[i].get_loss()[0]

                # loop over the input nodes to i and add its loss
                for k, j in enumerate(np.where(self.connectivity_matrix[i + \
                                                        self.nb_inputs] == 1)[0]):
                    self.components[j - self.nb_inputs].add_feedback(ext_feedback=loss_i[k])  # indexed at j

            # parameter update
            for idx in range(self.nb_components):
                self.components[idx].update()
        
        # reset
        self._feedback *= 0

    def collect_input(self, inputs: np.ndarray):
        
        """
        receive and store the inputs
        
        Parameters
        ----------
        inputs: np.ndarray
        
        Returns
        -------
        None
        """

        # external inputs
        self.activity[:self.nb_inputs] = inputs

    def get_output(self):

        """
        Returns
        -------
        output : float
        """

        return self.output
    
    def get_activity(self):
        
        """
        Returns
        -------
        activity : np.ndarray
        """
        
        return self.activity

    def get_trainable_params(self):

        """
        Returns
        -------
        trainable_params : np.ndarray
            shape=(nb_trainable,)
        """
        
        base_idx = 0
        
        # loop over the component to get the parameters from
        for idx in self.trainable_components:

            param_values = self.components[idx].get_trainable_params()
                            
            # store
            self.trainable_params[base_idx: base_idx + 
                                  len(param_values)] = param_values
            base_idx += len(param_values)

                

        return self.trainable_params

    def get_trainable_names(self):

        """
        Returns
        -------
        trainable_name : list
            names of the trainable parameters
        """

        return self.trainable_names

    def is_trainable(self):

        """
        Returns
        -------
        is_trainable : bool
            trainable molecule or not
        """

        return self.trainable
    
    def get_nb_trainable(self):
        
        """
        Returns
        -------
        number_of_trainable : int
            number of trainable parameters
        """
        
        return self.nb_trainable

    def get_nb_output(self):

        """
        Returns
        -------
        number_of_output : int
            number of outputs provided
        """

        return self.nb_outputs
    
    def get_connections(self):
       
        """
        Returns
        -------
        connections : list
        """
        
        return self.connections
        
    def get_nb_inout(self):
        
        """
        Returns
        -------
        number_of_input_output : tuple
            number of input and output components
        """
        
        return self.nb_inputs, self.nb_outputs
    
    def set_livestream(self, state=False):
        
        """
        set the state of the livestream
        Parameters
        ----------
        state : bool
        
        Returns
        -------
        None
        """
        
        self.livestream = state
        
        # initialize
        if self.grapher and self.livestream:
            self.grapher.initialize()
        
    def get_graph(self, show=False):

        """
        Returns
        -------
        graph : nx.Graph
        graph_color : list
        """

        if show:
            nx.draw_networkx(self.graph, pos=nx.shell_layout(self.graph), with_labels=True, node_color=self.graph_color, node_size=600)
            return

        return self.graph, self.graph_color

    def reset_structure(self):

        """
        reset the value of the variables [to edit]
        
        Returns
        -------
        None
        """

        self.activity *= 0.
        self.output *= 0
        
        for i in range(self.nb_components):
            self.components[i].reset()


#### PLASTICITY layer ####

class PlasticityLayer:

    def __init__(self, dna: dict):
            
        """
        Parameters
        ----------
        dna: dict
            dictionary containing the specifics of the substrate

        Returns
        -------
        None
        """

        self.nb_intf = 0
        self.nb_extf = 0

        self._externals = np.array([])
        self._internals = np.array([])
        self.plasticity = False

        self._plasticity_initialization(DNA=dna)

    def _plasticity_initialization(self, DNA: dict):

        """
        initialize the plasticity rule [to edit]

        Parameters
        ----------
        DNA: dict
            dictionary containing the specifics of the substrate
        
        Returns
        -------
        None
        """

        # check the presence of the required attributes
        if 'nb_intf' in DNA['attrb'].keys():
            self.nb_intf = DNA['attrb']['nb_intf']
            self._internals = np.zeros(self.nb_intf)
            self.plasticity = True

        if 'nb_extf' in DNA['attrb'].keys():
            self.nb_extf = DNA['attrb']['nb_extf']
            self._externals = np.zeros(self.nb_extf)
            self.plasticity = True

    def _collect_internals(self):

        """
        collect internal variables relevant for the plasticity rule [to edit]
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """

        pass
            
    def _reset_plasticity(self):

        """
        reset the value of the variables 
        
        Returns
        -------
        None
        """

        self._externals *= 0.
        self._internals *= 0.

    def collect_externals(self, externals: np.ndarray):

        """
        collect externals variables relevant for the plasticity rule
        Parameters
        ----------
        externals: np.ndarray
            array of variables of interest
        
        Returns
        -------
        None
        """

        self._externals = externals.copy()

    def get_internals(self):

        """
        Returns
        -------
        internals : np.ndarray
        """

        return self._internals


#### PROTEINS ####

class Protein(Substrate):

    """ base Protein class
    
    it serves as superclass for all Proteins, the building blocks of Structures
    and possessing clearly defined dynamics, whose parameters are trainable
    with gradient descent [and genetic algorithms]
    """

    def __init__(self, dna: dict, verbose=False):

        """
        Parameters
        ----------
        dna: dict
        verbose: bool
        
        Returns
        -------
        None
        """

        # set up
        Substrate.__init__(self, dna=dna)
        self.substrate_class = 'Protein'
        self.substrate_family = 'root'

        # variables
        self._z = 0.

        # parameters
        self._weights = None
        self.weight_shape = None
        
        self._original_params['w'] = None

        # activation
        self._activation = lambda x: x
        self._activation_deriv = lambda x: 1
        
        #
        self._protein_initialization()
        self._update_protein_dna()

        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}.{self.substrate_role}', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()

    def _protein_initialization(self, new_weights=False):

        """
        minimal DNA check of the class.Protein to set hyperparameters 
        like the cycles

        Returns
        -------
        None
        """

        # activation initialization | default None
        if 'activation' in self.DNA['attrb']:
            activation_name = self.DNA['attrb']['activation'] 
            self._activation, self._activation_deriv = activation_functions[activation_name]

        ### WEIGHT INITIALIZATION ###

        # weights in the DNA and no new weights are needed 
        if 'w' in self.DNA['params'] and not new_weights:

            if len(self.DNA['params']['w']) != self.nb_inputs:
                warnings.warn(f"weights size {len(self.DNA['params']['w'])} does not match input size {self.nb_inputs}, resizing", RuntimeWarning)

                # re-launch function but now generating new _weights
                self._protein_initialization(new_weights=True)
                return

            self._weights = np.array(self.DNA['params']['w']).reshape(1, -1)

        # new weights
        else:
            self._weights = np.around(np.abs(np.random.normal(1, 1 / np.sqrt(\
                                    self.nb_inputs + 1), (1, self.nb_inputs))), 4)
        
        
        # weight original copy
        self._original_params['w'] = self._weights.copy()
        self.weight_shape = self._weights.shape

        ### TRAINABLE RECORD ###
        # if the _weights are trainable, then update the trainable record 
        # adjusting for the number of inputs
        if 'trainable_params' in self.DNA['attrb']:  # if trainable params are defined
            if 'w' in self.DNA['attrb']['trainable_params'] or 'w1' in self.DNA['attrb']['trainable_params']:

                self.nb_trainable = self.nb_inputs
                
                # number of _weights + one bias
                self.trainable_params = np.zeros(self.nb_trainable)  

                # _weights
                self.trainable_names = [f'w{i + 1}' for i in range(self.nb_inputs)]

        #
        self.trainable = self.nb_trainable > 0
        
        #
        self.initialization_flag = True

    def _update_protein_dna(self):

        """ 
        update the dna with the current parameters [_weights]

        Returns
        -------
        None
        """

        self.DNA['params']['w'] = self._weights.tolist()
        self.DNA['attrb']['nb_inp'] = self.nb_inputs

        # update the DNA
        self._update_substrate_dna()

    def re_initialize(self, nb_inputs: int):

        """
        re-initialization of the _weights and parameters with a new number 
        of inputs

        Parameters
        ----------
        nb_inputs : int 

        Returns
        -------
        None 
        """

        self.nb_inputs = nb_inputs

        self._protein_initialization(new_weights=True)

        self._update_protein_dna()

    def get_trainable_params(self):

        """
        Returns
        -------
        trainable_params : list
            if trainable is True then return a list with the current
            values for the _weights and bias
        """

        if self.trainable:
            self.trainable_params = self._weights[0].copy()
            
        return self.trainable_params
    
    def get_output(self):

        """
        Returns
        -------
        output : float
            default activity
        """

        return self.activity
    
    def reset_protein(self):

        """
        reset the value of the variables
        
        Returns
        -------
        None
        """

        self._z *= 0
        self.activity *= 0
        self._weights = self._original_params['w'].copy()


class ProteinPlasticity(Protein, PlasticityLayer):

    """ base class for Protein implementing a plasticity rule
    
    if this class is used, it implements a simple weighted input, 
    and it is trained with gradient descent
    """

    def __init__(self, dna: dict, verbose=False):

        """
        Parameters
        ----------
        dna: dict,
        verbose: bool
        
        Returns
        -------
        None
        """

        # dna
        Protein.__init__(self, dna=dna)
        PlasticityLayer.__init__(self, dna=dna)

        self.substrate_family = 'Plasticity'
        
        #
        self._update_protein_dna()

        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}.{self.substrate_role}', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()

    def step(self):

        """
        receive and input and compute the output as a weighted sum
        
        Returns
        -------
        None
        """

        self.activity = np.dot(self._weights, self._ext_inputs)

    def update(self):

        """
        Returns
        -------
        None
        """

        if self.trainable:
            self._weights += self._lr * self._feedback

            self._feedback *= 0

    def reset_protein_plasticity(self):

        """
        reset the value of run-time variables ['externals']

        Returns
        -------
        None
        """
        self.reset_protein()
        self._reset_plasticity()


#### CELLS ####


class Cell(SubstrateStructure):
    
    """ a Structure subclass
    
    it implements a network of elementary Proteins component
    
    """

    def __init__(self, dna: dict, built_components: list, verbose=False):
        
        """
        Parameters
        ----------
        dna: dict
        built_components : list
            list of already built components
        verbose, bool
        
        Returns
        -------
        None
        """

        # set up
        SubstrateStructure.__init__(self, dna=dna, built_components=built_components)
        self.substrate_family = 'Cell'
        
        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}.{self.substrate_role}', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()
    
            print(f'\nnb_components: {self.nb_components}\nnb_inputs: '
                  f'{self.nb_inputs}\nnb_outputs: {self.nb_outputs}'
                  f'\nnb_trainable: {self.nb_trainable}')


class CellPlasticity(SubstrateStructure, PlasticityLayer):
    
    """ a Structure subclass,
    
    it is like a Cell.class but implementing one or more ProteinPlasticity
    """
    
    def __init__(self, dna: dict, built_components: list, verbose=False):

        """
        Parameters
        ----------
        dna : dict 
        built_components : list
            list of already built components
        verbose : bool
            default False 

        Returns
        -------
        None
        """

        # set up
        SubstrateStructure.__init__(self, dna=dna, built_components=built_components)
        PlasticityLayer.__init__(self, dna=dna)

        self.substrate_family = 'CellPlasticity'

        # output indexes 
        if 'idx_out' in dna['attrb'].keys():
            self.idx_out = dna['attrb']['idx_out']
            if isinstance(self.idx_out, int):
                self.idx_out = [self.idx_out]
                print('tupled!')
        else:
            warnings.warn("indexes of the output nodes not provided")
            self.idx_out = ()

      
        ## plasticity variables [output, inputs..., modulators...] ##
        # index of the internal components which are plastic 
        self.idx_plastic = []

        # index of the internal components from which register the activity 
        self.idx_intf = []
        
        # index of the external components from which register the activity
        self.idx_extf = []
        
        # arrays in which to store the activity of the internal and external plasticity variables
        self._comp_internals = None
        self._comp_externals = None 
        
        # initialize
        self._cell_plasticity_initialization()
        #self._build_structure(built_components=built_components)
        self._update_structure_dna()
        
        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}.{self.substrate_role}', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()
    
            print(f'\nnb_components: {self.nb_components}\nnb_inputs: '
                  f'{self.nb_input}\nnb_outputs: {self.nb_output}'
                  f'\nnb_plastic : {len(self.idx_plastic)}\nnb_trainable: {self.trainable_components}')

    def _cell_plasticity_initialization(self):

        """
        check DNA requirements and initialize class terms 

        Returns
        -------
        None
        """

        # attributes keys check 
        attributes_keys = self.DNA['attrb']
        assert 'idx_plastic' in attributes_keys, "missing attribute 'idx_plastic'"
        assert isinstance(self.DNA['attrb']['idx_plastic'], list), "idx_plastic must be a list"
        assert 'idx_intf' in attributes_keys, "missing attribute key 'idx_intf'"
        assert isinstance(self.DNA['attrb']['idx_intf'], list), "nb_intf must be a list"
        assert 'idx_extf' in attributes_keys, "missing attribute key 'idx_extf'"
        assert isinstance(self.DNA['attrb']['idx_extf'], list), "idx_extf must be a list"

        # initialize
        self.idx_plastic = self.DNA['attrb']['idx_plastic']
        self.idx_intf = self.DNA['attrb']['idx_intf']
        self.idx_extf = self.DNA['attrb']['idx_extf']
        self._comp_internals = np.zeros(len(self.idx_intf))
        self._comp_externals = np.zeros(len(self.idx_extf))

    def _build_structure2(self, built_components: list):
        
        """
        create each component from the DNA and store them in a list

        Parameters
        ----------
        built_components : list
            list of already built components
        
        Returns
        -------
        None
        """

        print('CellPlasticity building...')
        
        # alphabet for the paramters name
        alphabet = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 'k')

        # list of components and connections
        dna_components = self.DNA['components']
        dna_connections = self.DNA['connections']
        self.components = []

        # register each already built component
        for idx, a_component in enumerate(built_components):
            self.components += [a_component]
            
            # register trainable components
            if a_component.is_trainable():
                self.trainable_components += [idx]

        # trainable
        self.trainable = self.trainable_components.__len__() > 0

        # number of inputs and components
        self.nb_input = self.DNA['attrb']['nb_inp']  # inputs
        if self.idx_out is not None:
            try:
                self.nb_output = len(self.idx_out)
            except TypeError:
                self.nb_output = 1
        else:
            self.nb_output = 0 
            warnings.warn('Isolated Cell: zero outputs specified')

        self.nb_components = len(self.components)  # components
        tot = len(self.components) + self.nb_input   # total

        # build a connectivity matrix
        self.connectivity_matrix = np.zeros((tot, tot))

        # NB: the number of external inputs shall be considered; 
        # e.g. one input has index 0
        for j, i in dna_connections:
            if i >= tot:
                raise ValueError(f'source index {i} is too big')
            if j >= tot:
                raise ValueError(f'sink index {j} is too big')
            self.connectivity_matrix[i, j] = 1

        # initialize the _weights for each component
        for i in range(self.nb_components):
            #self.components[i].initialize(nb_inputs=int(
            #    self.connectivity_matrix[i + self.nb_input].sum()), idx=i)

            # if the components is within the substrate structure's trainable
            # parameters
            # book space in track for the tracking of params of the internal components

            # re-initialize each component and add its index
            self.components[i].re_initialize(nb_inputs=int(self.connectivity_matrix[i + self.nb_inputs].sum()))
            self.components[i].add_idx(idx=i)

            self.nb_trainable += self.components[i].get_nb_trainable()
            
            # register the trainable parameters name
            param_values = self.components[i].get_trainable_names()
                        
            # set name
            self.trainable_names += [f'c_{alphabet[k]}{i}' for k
                                     in range(len(param_values))]
            
        # adjust 
        self.nb_trainable = max((0, self.nb_trainable))
                        
        self.trainable_params = np.zeros(self.nb_trainable)  
        
        # internals
        #self._internals = np.zeros(self.nb_output + self.nb_input)
        
        # var
        self.activity = np.zeros(self.nb_input + self.nb_components)
        self.output = np.zeros(self.nb_output)
        
        # 
        self.initialization_flag = True

    def _collect_internals(self):
        
        """
        collect the values of the internal variables of interests
        
        Returns
        -------
        None
        """
    
        self._comp_internals = np.concatenate([self.components[idx].get_internals() \
                                                  for idx in self.idx_plastic])
    
    def step(self):

        """
        the internal components of the cell have their state updated
        
        Returns
        -------
        None
        """

        for cycle in range(self._cycles):

            # get the activity of each component, input to its downstream neighbours
            for idx, component in enumerate(self.components):
                self.activity[self.nb_inputs + idx] = component.get_output()

            # each component loads the inputs from its inputs sources +
            # each component steps
            for i in range(self.nb_components):

                # indexes of the inputs to component idx
                inputs_j = np.where(self.connectivity_matrix[i + 
                                                             self.nb_inputs] != 0)[0]

                if len(inputs_j) == 0:
                    continue

                # load
                self.components[i].collect_input(inputs=
                                                 self.activity.take(inputs_j).T)

                # step
                self.components[i].step()
                
            # define the output as the activity of the output components
            for l, idx in enumerate(self.idx_out):
                self.output[l] = self.components[idx].get_output()

            # set plasticity variables
            self._collect_internals()

            # reset inputs
            try:
                self.activity *= 0
            except RuntimeWarning:
                print('runtime warning: ', self.activity)
                input()
                
    def update(self):

        """
        the trainable parameters are updated [to edit]
        
        Returns
        -------
        None
        """

        # if self._feedback.sum() == 0:
        #    return
        # BACKPROPAGATION # [if backpropagation is enabled]
        if self.backprop_enabled:

            # loss at the output nodes
            for k in range(self.nb_output):
                self.components[-k-1].add_feedback(ext_feedback=self._feedback[k])

            # starting from the output nodes, loop over all the components except
            # the input nodes
                for i in range(self.nb_components - 1 - self.nb_input, 0, - 1):

                    loss_i = self.components[i].get_loss()[0]

                    # loop over the input nodes to i and add its loss
                    for k, j in enumerate(np.where(self.connectivity_matrix[i + 
                                                            self.nb_input] == 1)[0]):
                        self.components[j - self.nb_input].add_feedback(
                            ext_feedback=loss_i[k])  # indexed at j

        # the plasticity proteins collect internals 
        for k in self.idx_plastic:
            self.components[k].collect_externals(externals=self._comp_internals)

        # parameter update
        for idx in range(self.nb_components):
            self.components[idx].update()
        
        # reset
        self._feedback *= 0        


### ROOT DICTIONARY ###

root_dict = {'Substrate': lambda dna, verbose: Substrate(dna=dna, verbose=verbose),
             'SubstrateStructure': lambda dna, verbose: SubstrateStructure(dna=dna, verbose=verbose),
             'Protein': lambda dna, verbose: Protein(dna=dna, verbose=verbose),
             'ProteinPlasticity': lambda dna, verbose: ProteinPlasticity(dna=dna, verbose=verbose),
             'Cell': lambda dna, verbose: Cell(dna=dna, verbose=verbose, built_components=[]),
             'CellPlasticity': lambda dna, verbose: CellPlasticity(dna=dna, verbose=verbose, built_components=[]),
            }


### empty Substrate DNA ###

root_library = {'Substrate': {'params': {},
                              'attrb': {'nb_inp': 0,
                                       'nb_out': 0}},
                'SubstrateStructure': {'components': [],
                                       'connections': [],
                                       'params': {},
                                       'attrb': {'nb_inp': 0,
                                                'nb_out': 0,
                                                'idx_out': [],
                                                'idx_plastic': [],
                                                'trainable_params': [],
                                                'cycles': 0}
                                      },
                'Protein': {'params': {},
                            'attrb': {'nb_inp': 0,
                                     'nb_out': 0}},
                'ProteinPlasticity': {'params': {},
                                      'attrb': {'nb_inp': 0, 
                                               'nb_out': 0,
                                               'nb_extf': 2}},
                'Cell': {'components': [],
                         'connections': [],
                         'params': {},
                         'attrb': {'nb_inp': 0,
                                  'nb_out': 0,
                                  'idx_out': [],
                                  'trainable_params': [],
                                  'cycles': 0}
                        },
                'CellPlasticity': {'components': [],
                                   'connections': [],
                                   'params': {},
                                   'attrb': {'nb_inp': 0,
                                            'nb_out': 0,
                                            'idx_out': [],
                                            'idx_plastic': [],
                                            'idx_intf': [],
                                            'idx_extf': [],
                                            'trainable_params': [],
                                            'cycles': 0}
                        }
                } 



if __name__ == '__main__':

    print('%templates')
