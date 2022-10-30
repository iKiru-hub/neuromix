import numpy as np
import warnings
from tools import Grapher


#### SUBSTRATE GENERATOR ####

def generate_substrate(dna: tuple, verbose=True):

    """
    generate a substrate given a DNA
    :param dna: tuple, with structure as (substrate_name, {})
    :param verbose: bool, if True each substrate and its components will print
    their stamp
    :return: substrate
    """

    substrate_name = dna[0]

    if substrate_name == 'Protein':
        substrate = protein_dict[dna[1]['variety']](dna[1], verbose=verbose)

    elif substrate_name == 'Cell':
        substrate = Cell(dna=dna[1], verbose=verbose)

    else:
        raise NotImplementedError(f'substrate <{substrate_name}> not supported for now')

    return substrate


#### ACTIVATION FUNCTIONS ####

activation_functions = {'sigmoid': (lambda x: 1 / (1 + np.exp(-x)),
                                    lambda x: (1 / (1 + np.exp(-x))) * \
                                        (1 - (1 / (1 + np.exp(-x))))),
                        'sigMod': (lambda x: 1 / (1 + np.exp(-x/0.09 + 6)),
                                   lambda x: (1 / (1 + np.exp(-x/0.09 + 6))) * \
                                       (1 - (1 / (1 + np.exp(-x/0.09 + 6))))),
                        'relu': (lambda x: x * (x > 0),
                                 lambda x: 1 * (x > 0)),
                        'crelu': (lambda x: min((x * (x > 0), 1)),
                                  lambda x: 1 * (x > 0)),
                        'step': (lambda x: 1 * (x > 0),
                                   lambda x: 1),
                        'none': (lambda x: x,
                                 lambda x: 1)
                        }


#### SUBSTRATE ####

class SubstrateCore:
    
    """
    base class for a Substrate capable of its core dynamics (step) and 
    output definition
    """
    
    def __init__(self, dna: dict, verbose=False):

        """
        :param dna: dict, the DNA of the substrate
        :param verbose: bool, if True various log will be printed in console,
        defaul False
        """

        # DNA
        self.DNA = dna
        self.check_dna()
        self.id = '0'
        self.substrate_name = 'Substrate'
        
        self.verbose = verbose

        # numbers
        self.nb_input = 0
        self.nb_output = 1

        # variables
        self.inputs = np.zeros((self.nb_input, 1))
        self.activity = 0.
        
        
    def step(self):

        """
        receive an input and the state is updated [to edit]
        :return: None
        """

        pass

    def check_dna(self):

        """
        check if the DNA contains the right keys [to edit]
        :return: None
        """

        pass

    def initialize(self, nb_inputs: int, idx: int):

        """
        initialize the weights value and check the consistency if they are already initialized
        :param nb_inputs: int, number of inputs
        :param idx: int, id of the component
        :return: None
        """

        self.nb_input = nb_inputs
        self.id = idx


    def collect_input(self, inputs: np.ndarray):
        
        """
        receive and store the inputs
        :param inputs: np.ndarray
        :return: None
        """

        self.inputs = inputs


    def get_output(self):

        """
        return the output state
        :return: float
        """

        return self.activity


    def get_nb_output(self):

        """
        :return: int, number of outputs provided
        """

        return self.nb_output
    

    @staticmethod
    def get_substrate_name():

        """
        :return: str
        """

        return 'Substrate'
    
    def get_dna(self):
        
        """
        :return list
        """
        
        return (self.substrate_name, self.DNA)
    

    def reset(self):

        """
        reset the value of the variables [to edit]
        :return: None
        """

        self.inputs *= 0
        self.activity *= 0.
        
        
class SubstrateTrain(SubstrateCore):

    """
    base SubstrateTrain class:
        an extension of SubstrateCore to include training features
    """

    def __init__(self, dna: dict, verbose=False):

        """
        :param dna: dict
        """
        
        super().__init__(dna=dna, verbose=verbose)

        # hyper-parameters
        self.lr = self.DNA['more']['lr'] if 'lr' in \
            tuple(self.DNA['more'].keys()) else 0.

        
        # training
        self.trainable_names = self.DNA['more']['trainable_params'] 
        self.nb_trainable = max((0, len(self.trainable_names)))
        self.trainable_params = np.zeros(self.nb_trainable)
        self.trainable = self.nb_trainable > 0
        
        # backward
        self.back_loss = 0.


    def update(self):

        """
        the trainable parameters are updated [to edit]
        :return: None
        """

        pass


    def add_loss(self, backpropagated_loss: np.ndarray):
        
        """
        record another loss value from back-propagated from downstream
        :param backpropagated_loss: np.ndarray, external loss
        :return: None
        """

        self.back_loss += backpropagated_loss

    def get_loss(self):
        

        """
        compute and return the loss at this node
        :return: np.ndarray
        """

        return self.back_loss * self.weights

    def get_trainable_params(self):

        """
        [to edit]
        :return: None
        """

        return self.trainable_params
    
    def get_trainable_names(self):

        """
        :return: list, names of the trainable parameters
        """

        return self.trainable_names


    def get_nb_trainable(self):

        """
        :return: int, number of trainable params
        """

        return self.nb_trainable

    def is_trainable(self):

        """
        :return: bool, trainable molecule or not
        """

        return self.trainable

    
    def get_dna(self):
        
        """
        :return list
        """
        
        return (self.substrate_name, self.DNA)
    
        

class Substrate:

    """
    base Substrate class:
        an object endowed with internal dynamics and input-output channels,
        whose definition is encoded in a DNA
    """

    def __init__(self, dna: dict, verbose=False):

        """
        :param dna: dict
        """

        # DNA
        self.DNA = dna
        self.check_dna()
        self.id = '0'
        self.substrate_name = 'Substrate'
        
        self.verbose = verbose

        # hyper-parameters
        if 'more' in tuple(self.DNA.keys()):
            self.lr = self.DNA['more']['lr'] if 'lr' in \
                tuple(self.DNA['more'].keys()) else 0.

        # numbers
        self.nb_input = 0
        self.nb_output = 1
        
        # training
        self.trainable_names = ()
        self.nb_trainable = max((0, len(self.trainable_names)))
        self.trainable_params = np.zeros(self.nb_trainable)
        self.trainable = self.nb_trainable > 0

        # variables
        self.ext_inputs = np.zeros((self.nb_input, 1))
        self.activity = 0.

        # backward
        self.back_loss = np.zeros(1)
        
        # initialization
        self.initialization_flag = False


    def step(self):

        """
        receive an input and the state is updated [to edit]
        :return: None
        """

        pass

    def update(self):

        """
        the trainable parameters are updated [to edit]
        :return: None
        """

        pass

    def check_dna(self):

        """
        check if the DNA contains the right keys [to edit]
        :return: None
        """

        pass

    def initialize(self, nb_inputs: int, idx: int):

        """
        initialize the weights value and check the consistency if they are already initialized
        :param nb_inputs: int, number of inputs
        :param idx: int, id of the component
        :return: None
        """

        self.nb_input = nb_inputs
        self.id = idx


    def collect_input(self, inputs: np.ndarray):
        
        """
        receive and store the inputs
        :param inputs: np.ndarray
        :return: None
        """

        self.inputs = inputs

    def add_loss(self, backpropagated_loss: np.ndarray):
        
        """
        record another loss value from back-propagated from downstream
        :param backpropagated_loss: np.ndarray, external loss
        :return: None
        """

        self.back_loss += backpropagated_loss

    def get_loss(self):
        

        """
        compute and return the loss at this node
        :return: np.ndarray
        """

        return self.back_loss

    def get_output(self):

        """
        return the output state
        :return: float
        """

        return self.activity

    def get_trainable_params(self):

        """
        [to edit]
        :return: None
        """

        return self.trainable_params
    
    def get_trainable_names(self):

        """
        :return: list, names of the trainable parameters
        """

        return self.trainable_names


    def get_nb_trainable(self):

        """
        :return: int, number of trainable params
        """

        return self.nb_trainable

    def is_trainable(self):

        """
        :return: bool, trainable molecule or not
        """

        return self.trainable


    def get_nb_output(self):

        """
        :return: int, number of outputs provided
        """

        return self.nb_output
    

    @staticmethod
    def get_substrate_name():

        """
        :return: str
        """

        return 'Substrate'
    
    def get_dna(self):
        
        """
        :return: list
        """
        
        return self.substrate_name, self.DNA
    
    def update_dna(self):
        
        """
        update the dna with the new parameters
        :return: None
        """
        
        pass
    
    def is_initialized(self):
        
        """
        :return: bool
        """
        
        return self.initialization_flag

    def reset(self):

        """
        reset the value of the variables [to edit]
        :return: None
        """

        self.ext_inputs *= 0
        self.activity *= 0.


#### SUBSTRATE STRUCTURE ####


class SubstrateStructureCore(SubstrateCore):
    
    """
    base SubstrateStructureCore class, a Substrate object with the particularity
    of having a structure made of more elementary substrates, its
    components, which made up the internal dynamics (and input-output) of the 
    object through pre-defined connections 
    """

    def __init__(self, dna: dict, verbose=True):

        """
        :param dna: dict
        """

        # set up
        super().__init__(dna=dna)
        self.check_dna()
        self.substrate_name = 'Structure'
        
        # hyperparams
        self.cycles = self.DNA['more']['cycles'] if 'cycles' in \
            tuple(self.DNA['more'].keys()) else 1
        
        # structure
        self.components = []
        self.connections = self.DNA['connections']
        self.connectivity_matrix = np.zeros(0)
        self.nb_components = 0
                
        self.build_structure()
        
        # grapher
        self.grapher = False
        self.livestream = False
        
        # var
        self.activity = np.zeros(self.nb_input + self.nb_components)
        self.output = np.zeros(self.nb_output)

        if verbose:
            print('\n@SubstrateStructure', end='')
    
            print(f'\nnb_components: {self.nb_components}\nnb_inputs: {self.nb_input}\nnb_outputs: {self.nb_output}'
                  f'\nnb_trainable: {self.nb_trainable}')
            
    def step(self):

        """
        the internal components of the cell have their state updated
        :return: None
        """

        for cycle in range(self.cycles):

            # get the activity of each component, input to its downstream neighbours
            for idx, component in enumerate(self.components):
                self.activity[self.nb_input + idx] = component.get_output()

            # each component loads the inputs from its inputs sources + each component steps
            # print('\n-load and step-')
            for i, component in enumerate(self.components):

                # indexes of the inputs to component idx
                inputs_j = np.where(self.connectivity_matrix[i + self.nb_input] != 0)[0]

                if len(inputs_j) == 0:
                    continue

                # load
                self.components[i].collect_input(inputs=self.activity.take(inputs_j).T)

                # step
                self.components[i].step()
                
                
                # live graph
                if self.grapher and self.livestream:
                        
                    self.grapher.live_graph(activations=self.activity)
                

            # define the input as the activity of the output components
            for idx, component in enumerate(self.components[-self.nb_output:]):
                self.output[idx] = component.get_output()

            # reset inputs
            try:
                self.activity *= 0
            except RuntimeWarning:
                print('runtime warning: ', self.activity)
                input()


    def collect_input(self, inputs: np.ndarray):
        """
        receive and store the inputs
        :param inputs: np.ndarray
        :return: None
        """

        # external inputs
        self.activity[:self.nb_input] = inputs

    def check_dna(self):

        """
        check if the DNA contains the right keys
        :return: None
        """

        # general keys
        keys = tuple(self.DNA.keys())
        if not keys.__contains__('components') or not keys.__contains__('connections') or \
                not keys.__contains__('attributes') or not keys.__contains__('more'):

            raise ValueError('DNA does not contain one or more requires general keys')

        # <more> keys
        more_keys = tuple(self.DNA['more'])
        if not more_keys.__contains__('nb_in') or not more_keys.__contains__('nb_out') or \
                not more_keys.__contains__('cycles'):

            raise ValueError('DNA does not contain one or more required <more> keys')

        # warning
        if self.DNA['components'].__len__() == 0:
            warnings.warn('Empty Structure: no components detected')

        if self.DNA['connections'].__len__() == 0:
            warnings.warn('Disconnected components: no connections detected')

        if self.DNA['more']['nb_in'] == 0:
            warnings.warn('Autonomous Structure: zero inputs specified')

        if self.DNA['more']['nb_out'] == 0:
            warnings.warn('Isolated Structure: zero outputs specified')

    def build_structure(self):

        """
        create each component from the DNA and store them in a list
        :return: None
        """
        
        # alphabet for the paramters name
        alphabet = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 'k')

        # list of components and connections
        dna_components = self.DNA['components']
        dna_connections = self.DNA['connections']

        # create each component
        for idx, protein_gene in enumerate(dna_components):
            protein = generate_substrate(dna=protein_gene, verbose=self.verbose)
            self.components += [protein]

            self.trainable_components += [idx] * int(protein_gene[1]['more']['trainable_params'].__len__() > 0)

        # trainable
        self.trainable = self.trainable_components.__len__() > 0

        # number of inputs and components
        self.nb_input = self.DNA['more']['nb_in']  # inputs
        self.nb_output = self.DNA['more']['nb_out']
        self.nb_components = len(self.components)  # components
        tot = len(self.components) + self.nb_input   # total

        # build a connectivity matrix
        self.connectivity_matrix = np.zeros((tot, tot))

        # NB: the number of external inputs shall be considered; e.g. one input has index 0
        for j, i in dna_connections:
            if i >= tot:
                raise ValueError(f'source index {i} is too big')
            if j >= tot:
                raise ValueError(f'sink index {j} is too big')
            self.connectivity_matrix[i, j] = 1

        # initialize the weights for each component
        for i in range(self.nb_components):
            self.components[i].initialize(nb_inputs=int(self.connectivity_matrix[i + self.nb_input].sum()),
                                              idx=i)

            # if the components is within the substrate structure's trainable
            # parameters
            # book space in track for the tracking of params of the internal components
            self.nb_trainable += self.components[i].get_nb_trainable()
            
            # register the trainable parameters name
            param_values = self.components[i].get_trainable_names()
        
            # set name
            self.trainable_names += [f'c_{alphabet[k]}{i}' for k
                                     in range(len(param_values))]
            
        # adjust 
        self.nb_trainable = max((0, self.nb_trainable))
            
        # define param names #
        # loop over the component to get the parameters from
        """for i, param_name in enumerate(self.trainable_params):
            
            # component params, check if the params is within the trainable parameters
            if param_name[0] == f'c{i+1}':
                param_idx = int(param_name[1:]) 
                param_values = self.components[param_idx].get_trainable_params()
            
                # set name
                self.trainable_names += [param_name + f'_{alphabet[k]}' for k
                                         in range(len(param_values))]
                
            else:
            
            self.trainable_names += [param_name]"""
            
        self.trainable_params = np.zeros(self.nb_trainable)    

    def get_output(self):

        """
        :return float
        """

        return self.output
    
    def get_activity(self):
        
        """
        :return ndarray
        """
        
        return self.activity

    def get_nb_output(self):

        """
        :return int, number of outputs provided
        """

        return self.nb_output
    
    def get_connections(self):
        
        """
        :return list, connections
        """
        
        return self.connections
        
    def get_nb_inout(self):
        
        """
        :return tuple, number of input and output components
        """
        
        return self.nb_input, self.nb_output
    
    def set_livestream(self, state=False):
        
        """
        set the state of the livestream
        :param state: bool,
        :return None
        """
        
        self.livestream = state
        
        # initialize
        if self.grapher and self.livestream:
            self.grapher.initialize()
    
    def add_grapher(self):
        
        """
        add a mix.tools.Grapher object to live stream the activiy
        :return None
        """
        
        self.grapher = Grapher(connections=self.connections,
                               nb_input=self.nb_input,
                               nb_output=self.nb_output)
        
    def show_graph(self):
        
        """
        plot the graph of connections if a grapher object has been added
        :return None
        """
        
        if not self.grapher:
            warnings.warn("no grapher object has been added, no plot generated")
            return
        
        self.grapher.draw_graph()
        
        
    def get_grapher(self):
        
        """
        :return object, Grapher object with its graph data relative to this
        substrate structure; return False if no grapher has been addedd
        """
        
        return self.grapher
        

    @staticmethod
    def get_substrate_name():

        """
        :return str
        """

        return 'SubstrateStructure'
    
    def reset(self):

        """
        reset the value of the variables [to edit]
        :return: None
        """

        self.inputs *= 0
        self.activity *= 0.
        
        for i in range(self.nb_components):
            self.components[i].reset()
        
    
    """
    base SubstrateStructure class, a Substrate object with the particularity
    of having a structure made of more elementary substrates, its
    components, which made up the internal dynamics (and input-output) of the 
    object through pre-defined connections 
    """

    def __init__(self, dna: dict, verbose=True):

        """
        :param dna: dict
        """

        # set up
        super().__init__(dna=dna)
        
        
        self.trainable_components = []
        
        if verbose:
            print('\n@SubstrateStructure', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()
    
            print(f'\nnb_components: {self.nb_components}\nnb_inputs: {self.nb_input}\nnb_outputs: {self.nb_output}'
                  f'\nnb_trainable: {self.nb_trainable}')
            
    def step(self):

        """
        the internal components of the cell have their state updated
        :return: None
        """

        for cycle in range(self.cycles):

            # get the activity of each component, input to its downstream neighbours
            for idx, component in enumerate(self.components):
                self.activity[self.nb_input + idx] = component.get_output()

            # each component loads the inputs from its inputs sources + each component steps
            # print('\n-load and step-')
            for i, component in enumerate(self.components):

                # indexes of the inputs to component idx
                inputs_j = np.where(self.connectivity_matrix[i + self.nb_input] != 0)[0]

                if len(inputs_j) == 0:
                    continue

                # load
                self.components[i].collect_input(inputs=self.activity.take(inputs_j).T)

                # step
                self.components[i].step()
                
                
                # live graph
                if self.grapher and self.livestream:
                        
                    self.grapher.live_graph(activations=self.activity)
                

            # define the input as the activity of the output components
            for idx, component in enumerate(self.components[-self.nb_output:]):
                self.output[idx] = component.get_output()

            # reset inputs
            try:
                self.activity *= 0
            except RuntimeWarning:
                print('runtime warning: ', self.activity)
                input()

    def update(self):

        """
        the trainable parameters are updated [to edit]
        :return: None
        """

        if self.back_loss.sum() == 0:
            return

        # loss at the output nodes
        for k in range(self.nb_output):
            self.components[-k-1].add_loss(backpropagated_loss=self.back_loss[k])

        # BACKPROPAGATION #
        # starting from the output nodes, loop over all the components
        for i in range(self.nb_components - 1, 0, - 1):

            loss_i = self.components[i].get_loss()[0]

            # loop over the input nodes to i and add its loss
            for k, j in enumerate(np.where(self.connectivity_matrix[i + self.nb_input] == 1)[0]):
                self.components[j - self.nb_input].add_loss(backpropagated_loss=loss_i[k])  # indexed at j

        # parameter update
        for idx in range(self.nb_components):
            self.components[idx].update()
        
        # reset
        self.back_loss *= 0


    def collect_input(self, inputs: np.ndarray):
        """
        receive and store the inputs
        :param inputs: np.ndarray
        :return: None
        """

        # external inputs
        self.activity[:self.nb_input] = inputs

    def check_dna(self):

        """
        check if the DNA contains the right keys
        :return: None
        """

        # general keys
        keys = tuple(self.DNA.keys())
        if not keys.__contains__('components') or not keys.__contains__('connections') or \
                not keys.__contains__('attributes') or not keys.__contains__('more'):

            raise ValueError('DNA does not contain one or more requires general keys')

        # <more> keys
        more_keys = tuple(self.DNA['more'])
        if not more_keys.__contains__('nb_in') or not more_keys.__contains__('nb_out') or \
                not more_keys.__contains__('cycles'):

            raise ValueError('DNA does not contain one or more required <more> keys')

        # warning
        if self.DNA['components'].__len__() == 0:
            warnings.warn('Empty Structure: no components detected')

        if self.DNA['connections'].__len__() == 0:
            warnings.warn('Disconnected components: no connections detected')

        if self.DNA['more']['nb_in'] == 0:
            warnings.warn('Autonomous Structure: zero inputs specified')

        if self.DNA['more']['nb_out'] == 0:
            warnings.warn('Isolated Structure: zero outputs specified')

    def build_structure(self):

        """
        create each component from the DNA and store them in a list
        :return: None
        """
        
        # alphabet for the paramters name
        alphabet = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 'k')

        # list of components and connections
        dna_components = self.DNA['components']
        dna_connections = self.DNA['connections']

        # create each component
        for idx, protein_gene in enumerate(dna_components):
            protein = generate_substrate(dna=protein_gene, verbose=self.verbose)
            self.components += [protein]

            self.trainable_components += [idx] * int(protein_gene[1]['more']['trainable_params'].__len__() > 0)

        # trainable
        self.trainable = self.trainable_components.__len__() > 0

        # number of inputs and components
        self.nb_input = self.DNA['more']['nb_in']  # inputs
        self.nb_output = self.DNA['more']['nb_out']
        self.nb_components = len(self.components)  # components
        tot = len(self.components) + self.nb_input   # total

        # build a connectivity matrix
        self.connectivity_matrix = np.zeros((tot, tot))

        # NB: the number of external inputs shall be considered; e.g. one input has index 0
        for j, i in dna_connections:
            if i >= tot:
                raise ValueError(f'source index {i} is too big')
            if j >= tot:
                raise ValueError(f'sink index {j} is too big')
            self.connectivity_matrix[i, j] = 1

        # initialize the weights for each component
        for i in range(self.nb_components):
            self.components[i].initialize(nb_inputs=int(self.connectivity_matrix[i + self.nb_input].sum()),
                                              idx=i)

            # if the components is within the substrate structure's trainable
            # parameters
            # book space in track for the tracking of params of the internal components
            self.nb_trainable += self.components[i].get_nb_trainable()
            
            # register the trainable parameters name
            param_values = self.components[i].get_trainable_names()
        
            # set name
            self.trainable_names += [f'c_{alphabet[k]}{i}' for k
                                     in range(len(param_values))]
            
        # adjust 
        self.nb_trainable = max((0, self.nb_trainable))
            
        # define param names #
        # loop over the component to get the parameters from
        """for i, param_name in enumerate(self.trainable_params):
            
            # component params, check if the params is within the trainable parameters
            if param_name[0] == f'c{i+1}':
                param_idx = int(param_name[1:]) 
                param_values = self.components[param_idx].get_trainable_params()
            
                # set name
                self.trainable_names += [param_name + f'_{alphabet[k]}' for k
                                         in range(len(param_values))]
                
            else:
            
            self.trainable_names += [param_name]"""
            
        self.trainable_params = np.zeros(self.nb_trainable)    

    def get_output(self):

        """
        :return float
        """

        return self.output
    
    def get_activity(self):
        
        """
        :return ndarray
        """
        
        return self.activity

    def get_trainable_params(self):

        """
        :return: ndarray shape=(nb_trainable,)
        """
        
        base_idx = 0
        
        # loop over the component to get the parameters from
        for idx in self.trainable_components:

            param_values = self.components[idx].get_trainable_params()
                            
            # store
            self.trainable_params[base_idx: base_idx + len(param_values)] = param_values
            base_idx += len(param_values)

                

        return self.trainable_params

    def get_trainable_names(self):

        """
        :return: list, names of the trainable parameters
        """

        return self.trainable_names

    def is_trainable(self):

        """
        :return: bool, trainable molecule or not
        """

        return self.trainable
    
    def get_nb_trainable(self):
        
        """
        :return int, numebr of trainable parameters
        """
        
        return self.nb_trainable

    def get_nb_output(self):

        """
        :return int, number of outputs provided
        """

        return self.nb_output
    
    def get_connections(self):
        
        """
        :return list, connections
        """
        
        return self.connections
        
    def get_nb_inout(self):
        
        """
        :return tuple, number of input and output components
        """
        
        return self.nb_input, self.nb_output
    
    def set_livestream(self, state=False):
        
        """
        set the state of the livestream
        :param state: bool,
        :return None
        """
        
        self.livestream = state
        
        # initialize
        if self.grapher and self.livestream:
            self.grapher.initialize()
    
    def add_grapher(self):
        
        """
        add a mix.tools.Grapher object to live stream the activiy
        :return None
        """
        
        self.grapher = Grapher(connections=self.connections,
                               nb_input=self.nb_input,
                               nb_output=self.nb_output)
        
    def show_graph(self):
        
        """
        plot the graph of connections if a grapher object has been added
        :return None
        """
        
        if not self.grapher:
            warnings.warn("no grapher object has been added, no plot generated")
            return
        
        self.grapher.draw_graph()
        
        
    def get_grapher(self):
        
        """
        :return object, Grapher object with its graph data relative to this
        substrate structure; return False if no grapher has been addedd
        """
        
        return self.grapher
        

    @staticmethod
    def get_substrate_name():

        """
        :return str
        """

        return 'SubstrateStructure'
    
    def reset(self):

        """
        reset the value of the variables [to edit]
        :return: None
        """

        self.inputs *= 0
        self.activity *= 0.
        
        for i in range(self.nb_components):
            self.components[i].reset()
        
        
class SubstrateStructure(Substrate):
    
    """
    base SubstrateStructure class, a Substrate object with the particularity
    of having a structure made of more elementary substrates, its
    components, which made up the internal dynamics (and input-output) of the 
    object through pre-defined connections 
    """

    def __init__(self, dna: dict, verbose=True):

        """
        :param dna: dict
        """

        # set up
        super().__init__(dna=dna)
        self.check_dna()
        self.substrate_name = 'Structure'
        
        # hyperparams
        self.cycles = self.DNA['more']['cycles'] if 'cycles' in \
            tuple(self.DNA['more'].keys()) else 1
        
        # structure
        self.components = []
        self.connections = self.DNA['connections']
        self.connectivity_matrix = np.zeros(0)
        self.nb_components = 0
        
        self.trainable_components = []
        
        self.build_structure()
        
        # grapher
        self.grapher = False
        self.livestream = False
        
        # var
        self.activity = np.zeros(self.nb_input + self.nb_components)
        self.output = np.zeros(self.nb_output)

        if verbose:
            print('\n@SubstrateStructure', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()
    
            print(f'\nnb_components: {self.nb_components}\nnb_inputs: {self.nb_input}\nnb_outputs: {self.nb_output}'
                  f'\nnb_trainable: {self.nb_trainable}')
            
    def step(self):

        """
        the internal components of the cell have their state updated
        :return: None
        """

        for cycle in range(self.cycles):

            # get the activity of each component, input to its downstream neighbours
            for idx, component in enumerate(self.components):
                self.activity[self.nb_input + idx] = component.get_output()

            # each component loads the inputs from its inputs sources + each component steps
            # print('\n-load and step-')
            for i, component in enumerate(self.components):

                # indexes of the inputs to component idx
                inputs_j = np.where(self.connectivity_matrix[i + self.nb_input] != 0)[0]

                if len(inputs_j) == 0:
                    continue

                # load
                self.components[i].collect_input(inputs=self.activity.take(inputs_j).T)

                # step
                self.components[i].step()
                
                
                # live graph
                if self.grapher and self.livestream:
                        
                    self.grapher.live_graph(activations=self.activity)
                

            # define the input as the activity of the output components
            for idx, component in enumerate(self.components[-self.nb_output:]):
                self.output[idx] = component.get_output()

            # reset inputs
            try:
                self.activity *= 0
            except RuntimeWarning:
                print('runtime warning: ', self.activity)
                input()

    def update(self):

        """
        the trainable parameters are updated [to edit]
        :return: None
        """

        if self.back_loss.sum() == 0:
            return

        # loss at the output nodes
        for k in range(self.nb_output):
            self.components[-k-1].add_loss(backpropagated_loss=self.back_loss[k])

        # BACKPROPAGATION #
        # starting from the output nodes, loop over all the components
        for i in range(self.nb_components - 1, 0, - 1):

            loss_i = self.components[i].get_loss()[0]

            # loop over the input nodes to i and add its loss
            for k, j in enumerate(np.where(self.connectivity_matrix[i + self.nb_input] == 1)[0]):
                self.components[j - self.nb_input].add_loss(backpropagated_loss=loss_i[k])  # indexed at j

        # parameter update
        for idx in range(self.nb_components):
            self.components[idx].update()
        
        # reset
        self.back_loss *= 0


    def collect_input(self, inputs: np.ndarray):
        """
        receive and store the inputs
        :param inputs: np.ndarray
        :return: None
        """

        # external inputs
        self.activity[:self.nb_input] = inputs

    def check_dna(self):

        """
        check if the DNA contains the right keys
        :return: None
        """

        # general keys
        keys = tuple(self.DNA.keys())
        if not keys.__contains__('components') or not keys.__contains__('connections') or \
                not keys.__contains__('attributes') or not keys.__contains__('more'):

            raise ValueError('DNA does not contain one or more requires general keys')

        # <more> keys
        more_keys = tuple(self.DNA['more'])
        if not more_keys.__contains__('nb_in') or not more_keys.__contains__('nb_out') or \
                not more_keys.__contains__('cycles'):

            raise ValueError('DNA does not contain one or more required <more> keys')

        # warning
        if self.DNA['components'].__len__() == 0:
            warnings.warn('Empty Structure: no components detected')

        if self.DNA['connections'].__len__() == 0:
            warnings.warn('Disconnected components: no connections detected')

        if self.DNA['more']['nb_in'] == 0:
            warnings.warn('Autonomous Structure: zero inputs specified')

        if self.DNA['more']['nb_out'] == 0:
            warnings.warn('Isolated Structure: zero outputs specified')

    def build_structure(self):

        """
        create each component from the DNA and store them in a list
        :return: None
        """
        
        # alphabet for the paramters name
        alphabet = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 'k')

        # list of components and connections
        dna_components = self.DNA['components']
        dna_connections = self.DNA['connections']

        # create each component
        for idx, protein_gene in enumerate(dna_components):
            protein = generate_substrate(dna=protein_gene, verbose=self.verbose)
            self.components += [protein]

            self.trainable_components += [idx] * int(protein_gene[1]['more']['trainable_params'].__len__() > 0)

        # trainable
        self.trainable = self.trainable_components.__len__() > 0

        # number of inputs and components
        self.nb_input = self.DNA['more']['nb_in']  # inputs
        self.nb_output = self.DNA['more']['nb_out']
        self.nb_components = len(self.components)  # components
        tot = len(self.components) + self.nb_input   # total

        # build a connectivity matrix
        self.connectivity_matrix = np.zeros((tot, tot))

        # NB: the number of external inputs shall be considered; e.g. one input has index 0
        for j, i in dna_connections:
            if i >= tot:
                raise ValueError(f'source index {i} is too big')
            if j >= tot:
                raise ValueError(f'sink index {j} is too big')
            self.connectivity_matrix[i, j] = 1

        # initialize the weights for each component
        for i in range(self.nb_components):
            self.components[i].initialize(nb_inputs=int(self.connectivity_matrix[i + self.nb_input].sum()),
                                              idx=i)

            # if the components is within the substrate structure's trainable
            # parameters
            # book space in track for the tracking of params of the internal components
            self.nb_trainable += self.components[i].get_nb_trainable()
            
            # register the trainable parameters name
            param_values = self.components[i].get_trainable_names()
        
            # set name
            self.trainable_names += [f'c_{alphabet[k]}{i}' for k
                                     in range(len(param_values))]
            
        # adjust 
        self.nb_trainable = max((0, self.nb_trainable))
            
        # define param names #
        # loop over the component to get the parameters from
        """for i, param_name in enumerate(self.trainable_params):
            
            # component params, check if the params is within the trainable parameters
            if param_name[0] == f'c{i+1}':
                param_idx = int(param_name[1:]) 
                param_values = self.components[param_idx].get_trainable_params()
            
                # set name
                self.trainable_names += [param_name + f'_{alphabet[k]}' for k
                                         in range(len(param_values))]
                
            else:
            
            self.trainable_names += [param_name]"""
            
        self.trainable_params = np.zeros(self.nb_trainable)  
        
        # 
        self.initialization_flag = True

    def get_output(self):

        """
        :return float
        """

        return self.output
    
    def get_activity(self):
        
        """
        :return ndarray
        """
        
        return self.activity

    def get_trainable_params(self):

        """
        :return: ndarray shape=(nb_trainable,)
        """
        
        base_idx = 0
        
        # loop over the component to get the parameters from
        for idx in self.trainable_components:

            param_values = self.components[idx].get_trainable_params()
                            
            # store
            self.trainable_params[base_idx: base_idx + len(param_values)] = param_values
            base_idx += len(param_values)

                

        return self.trainable_params

    def get_trainable_names(self):

        """
        :return: list, names of the trainable parameters
        """

        return self.trainable_names

    def is_trainable(self):

        """
        :return: bool, trainable molecule or not
        """

        return self.trainable
    
    def get_nb_trainable(self):
        
        """
        :return int, numebr of trainable parameters
        """
        
        return self.nb_trainable

    def get_nb_output(self):

        """
        :return int, number of outputs provided
        """

        return self.nb_output
    
    def get_connections(self):
        
        """
        :return list, connections
        """
        
        return self.connections
        
    def get_nb_inout(self):
        
        """
        :return tuple, number of input and output components
        """
        
        return self.nb_input, self.nb_output
    
    def set_livestream(self, state=False):
        
        """
        set the state of the livestream
        :param state: bool,
        :return None
        """
        
        self.livestream = state
        
        # initialize
        if self.grapher and self.livestream:
            self.grapher.initialize()
    
    def add_grapher(self):
        
        """
        add a mix.tools.Grapher object to live stream the activiy
        :return None
        """
        
        self.grapher = Grapher(connections=self.connections,
                               nb_input=self.nb_input,
                               nb_output=self.nb_output)
        
    def show_graph(self):
        
        """
        plot the graph of connections if a grapher object has been added
        :return None
        """
        
        if not self.grapher:
            warnings.warn("no grapher object has been added, no plot generated")
            return
        
        self.grapher.draw_graph()
        
        
    def get_grapher(self):
        
        """
        :return object, Grapher object with its graph data relative to this
        substrate structure; return False if no grapher has been addedd
        """
        
        return self.grapher
        

    @staticmethod
    def get_substrate_name():

        """
        :return str
        """

        return 'SubstrateStructure'
    
    def reset(self):

        """
        reset the value of the variables [to edit]
        :return: None
        """

        self.inputs *= 0
        self.activity *= 0.
        
        for i in range(self.nb_components):
            self.components[i].reset()
        
        
#### MOLECULES ####

class Protein(Substrate):

    """
    base Protein class
    """

    def __init__(self, dna: dict):

        """
        :param dna: dict
        """

        # set up
        super().__init__(dna=dna)
        self.substrate_name = 'Protein'

        # parameters
        self.weights = False

        # if weights already defined in the DNA, use those
        if 'attributes' in tuple(self.DNA.keys()):
            if 'w' in tuple(self.DNA['attributes'].keys()):
                self.weights = np.array(self.DNA['attributes']['w']).reshape(1, -1).astype(float)

        # variables
        self.z = 0.
        
        # activation
        self.activation = lambda x: x
        self.activation_deriv = lambda x: 1


    def initialize(self, nb_inputs: int, idx: int):

        """
        initialize the weights value and check the consistency if they are already initialized
        :param nb_inputs: int, number of inputs
        :param idx: int, id of the component
        :return: None
        """

        self.nb_input = nb_inputs
        self.id = idx
        self.ext_inputs = np.zeros((nb_inputs, 1))

        # weights not initialized yet -> create and record
        if not isinstance(self.weights, np.ndarray):
            self.weights = np.abs(np.random.normal(1, 1 / np.sqrt(nb_inputs + 1), (1, nb_inputs)))

        # weights already initialized, check
        elif self.weights.shape[1] != nb_inputs:
            raise ValueError(f'weights of shape {self.weights.shape} do not match the number of inputs [{nb_inputs:d}]')

        # if the weights are trainable, then update the trainable record adjusting for the number of inputs
        if 'w' in self.DNA['more']['trainable_params']:
            self.nb_trainable = self.nb_input + 1
            self.trainable_params = np.zeros(self.nb_trainable)  # number of weights + one bias

            # weights
            self.trainable_names = [f'w{i + 1}' for i in range(nb_inputs)]

            # tuple
            self.trainable_names = tuple(self.trainable_names)

        #
        self.update_dna()
        #
        self.initialization_flag = True
        
        
    def collect_input(self, inputs: np.ndarray):

        """
        receive and store the inputs
        :param inputs: np.ndarray
        :return: None
        """

        # dont store absent inputs
        if np.any(inputs != 0):
            self.ext_inputs[:] = inputs


    def get_trainable_params(self):

        """
        :return: list, if trainable is True then return a list with the current
        values for the weights and bias
        """

        if self.trainable:
            self.trainable_params[:-1] = self.weights[0]


    @staticmethod
    def get_substrate_name():

        """
        :return: str, "Protein"
        """

        return 'Protein'

    def reset(self):

        """
        reset the value of the variables [to edit]
        :return: None
        """

        self.z *= 0
        self.activity *= 0


class ProteinExpBeta(Protein):

    """
    a Molecule that takes an input and returns an output by using a specific activity kernel
    """

    def __init__(self, dna: dict, verbose=True):

        # dna
        super().__init__(dna=dna)

        # param
        self.tau = self.DNA['attributes']['tau']
        self.Eq = self.DNA['attributes']['Eq']
        self.alpha = self.DNA['attributes']['alpha']
        self.beta = self.DNA['attributes']['beta']

        # var
        self.inputs = np.zeros((self.nb_input, 1))
        self.z = self.Eq
        self.a = self.z

        if verbose:
            print('\n@MoleculeExpBeta', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()

    def step(self):

        """
        receive an input and the state is updated
        :return: None
        """

        u = np.dot(self.w, self.inputs).item()
        if u:
            self.z = u

        else:
            self.z = self.z + (self.Eq - self.z) / self.tau

            # activation
            self.a = (self.z ** (self.alpha - 1)) * ((1 - self.z) ** (self.beta - 1)) * 11

    def backward(self, error: np.ndarray):

        """
        the trainable parameters are updated [to edit]
        :param error: np.ndarray
        :return: None
        """

        if not error[0].item():
            return

        if 'tau' in self.trainable_params:
            self.tau += self.lr * error * ((self.alpha - 1) * (self.z ** (self.alpha - 2) *
                                            self.beta - 1) * ((1 - self.z) ** (self.beta - 2))) * (self.Eq - self.z)

        if 'Eq' in self.trainable_params:
            self.Eq += self.lr * error / self.tau

        if 'alpha' in self.trainable_params:
            self.alpha -= self.lr * error * (self.alpha - 1) * self.a

        if 'beta' in self.trainable_params:
            self.beta -= self.lr * error * (self.beta - 1) * self.a

        if 'w' in self.trainable_params:
            self.w -= self.lr * error * ((self.alpha - 1) * (self.z ** (self.alpha - 2) *
                                            self.beta - 1) * ((1 - self.z) ** (self.beta - 2))) * self.inputs

    def check_dna(self):

        """
        check if the DNA contains the right keys
        :return: None
        """

        attributes_keys = tuple(self.DNA['attributes'].keys())
        more_keys = tuple(self.DNA['more'])
        if not attributes_keys.__contains__('tau') or not attributes_keys.__contains__('Eq') or \
                not attributes_keys.__contains__('alpha') or not attributes_keys.__contains__('beta') or \
                not attributes_keys.__contains__('w') or \
                not more_keys.__contains__('lr') or not more_keys.__contains__('trainable_params'):

            raise ValueError('DNA does not contain one or more required keys')

    def collect_input(self, inputs: np.ndarray):
        """
        receive and store the inputs
        :param inputs: np.ndarray
        :return: None
        """

        self.inputs = inputs

    def get_output(self):

        """
        return the output state
        :return: float
        """

        return self.a

    def get_trainable_params(self):

        """
        :return: ndarray shape=(nb_trainable,)
        """

        k = 0  # weight index
        for i, param in enumerate(self.trainable_params):
            if param == 'tau':
                self.track[i] = self.tau

            elif param == 'Eq':
                self.track[i] = self.Eq

            elif param == 'alpha':
                self.track[i] = self.alpha

            elif param == 'beta':
                self.track[i] = self.beta

            elif param == 'w':
                self.track[i] = self.w[k]
                k += 1

        return self.track

    def reset(self):

        """
        reset the run-time variables
        :return: None
        """

        self.z = self.Eq
        self.a *= 0


class ProteinExp(Protein):

    """
    a Molecule that takes an input and returns an output by exploiting exponential decay
    """

    def __init__(self, dna: dict, verbose=True):

        # dna
        super().__init__(dna=dna)
        self.check_dna()

        # param
        self.tau = self.DNA['attributes']['tau']
        self.Eq = self.DNA['attributes']['Eq']
        self.sign = 2 * (self.Eq < 0.5) - 1

        # var
        self.z = self.Eq
        
        # activation
        if 'activation' in dna['more']:
            activation_name = dna['more']['activation'] 
        else:
            activation_name = 'none'
            self.DNA['more']['activation'] = 'none'
        self.activation, self.activation_deriv = activation_functions[activation_name]

        
        if verbose:
            print('\n@ProteinExp', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()

    def step(self):

        """
        receive an input and the state is updated
        :return: None
        """
        
        self.z = abs(self.z + (self.Eq - self.z) / self.tau + self.sign \
                     * np.dot(self.weights, self.ext_inputs))
        
        self.activity = self.activation(self.z)
        if self.verbose:
            print(f'\n#\nx: {self.inputs}\nEq: {self.Eq}\nz: {self.z}\na: {self.activity}')

        self.ext_inputs *= 0

    def update(self):

        """
        the trainable parameters are updated [to edit]
        :return: None
        """
        
        # print('updating ', self.back_loss, self.z)

        if self.back_loss.item() == 0:
            return

        if 'w1' in self.trainable_names:
            self.weights += self.lr * self.back_loss * self.ext_inputs.T

        if 'tau' in self.trainable_names: 
            self.tau -= self.lr * self.back_loss * (self.Eq - self.z)
            
            if self.verbose:
                print(f'tau: {self.tau}\nbloss: {self.back_loss}\ngrad: {self.lr * self.back_loss * (self.Eq - self.z)}')
            
        if 'Eq' in self.trainable_names:
            self.Eq += self.lr * self.back_loss / self.tau
            
            
        self.back_loss *= 0
            
    def add_loss(self, backpropagated_loss: float):
        """
        record another loss value from back-propagated from downstream
        :param backpropagated_loss: float, external loss
        :return: None
        """

        self.back_loss += backpropagated_loss * self.activation_deriv(self.z)
        if self.verbose:
            print(f'backprop loss={backpropagated_loss}\nact der={self.activation_deriv(self.z)} ')


    def check_dna(self):

        """
        check if the DNA contains the right keys
        :return: None
        """

        attributes_keys = tuple(self.DNA['attributes'].keys())
        if not attributes_keys.__contains__('tau') or not attributes_keys.__contains__('Eq'):

            raise ValueError('DNA does not contain one or more required keys')
                
    
    def get_trainable_params(self):

        """
        :return: ndarray shape=(nb_trainable,)  | <to edit>
        """

        k = 0
        for i, param in enumerate(self.trainable_names):
            
            if param == 'tau':
                self.trainable_params[i] = self.tau

            elif param == 'Eq':
                self.trainable_params[i] = self.Eq

            elif param == 'w1':
                self.trainable_params[i:] = self.weights[k]
                k += 1

        return self.trainable_params
    
    
    def initialize(self, nb_inputs: int, idx: int):

        """
        initialize the weights value and check the consistency if they are already initialized
        :param nb_inputs: int, number of inputs
        :param idx: int, id of the component
        :return: None
        """
        
        self.nb_input = nb_inputs
        self.id = str(idx)
        self.ext_inputs = np.zeros((nb_inputs, 1))

        # weights not initialized yet
        if not isinstance(self.weights, np.ndarray):
            self.weights = np.abs(np.random.normal(1, 1 / np.sqrt(nb_inputs + 1), (1, nb_inputs)))

        # weights already initialized, check
        elif self.weights.shape[1] != nb_inputs:
            raise ValueError(f'weights of shape {self.weights.shape} do not match the number of inputs [{nb_inputs:d}]')

        # if the weights are trainable, then update the trainable record adjusting for the number of inputs
        if 'w' in self.DNA['more']['trainable_params']:
            
            self.nb_trainable = self.nb_input + 1
            self.trainable_params = np.zeros(self.nb_trainable)  # number of weights + one bias
            
            # weights
            self.trainable_names = [f'w{i+1}' for i in range(nb_inputs)]
            
            # tuple
            self.trainable_names = tuple(self.trainable_names)
            
        #
        self.update_dna()
        
        # 
        self.initialization_flag = True
        
        
    def update_dna(self):
        
        """
        update the dna with the new parameters
        :return: None
        """
        
        self.DNA['attributes']['w'] = self.weights.tolist()
        self.DNA['attributes']['tau'] = self.tau
        self.DNA['attributes']['Eq'] = self.Eq
            

    def reset(self):

        """
        reset the run-time variables
        :return: None
        """

        self.z *= self.Eq
        self.activity *= 0


class ProteinBase(Protein):

    """
    a Protein designed as an artificial neuron
    """

    def __init__(self, dna: dict, verbose=False):

        # dna
        super().__init__(dna=dna)
        self.check_dna()
        self.verbose = False

        # param
        self.bias = np.abs(np.random.normal(0, 0.1))

        # var
        self.z = 0.        
        
        # activation
        if 'activation' in dna['more']:
            activation_name = dna['more']['activation'] 
        else:
            activation_name = 'sigmoid'
            self.DNA['more']['activation'] = 'sigmoid'
   
        self.activation, self.activation_deriv = activation_functions[activation_name]

        if verbose:
            print('\n@ProteinBase', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()

    def step(self):

        """
        receive an input and the state is updated
        :return: None
        """

        self.z = np.dot(self.weights, self.ext_inputs) + self.bias
        self.activity = self.activation(self.z) * int(np.any(self.ext_inputs != 0))  # dont step if absent inputs
       
    def update(self):

        """
        the trainable parameters are updated [to edit]
        :return: None
        """

        if 'w1' in self.trainable_names:
            self.weights += self.lr * self.back_loss * self.ext_inputs.T
            self.bias += self.lr * self.back_loss

        # reset
        self.back_loss *= 0


    def add_loss(self, backpropagated_loss: float):
        
        """
        record another loss value from back-propagated from downstream
        :param backpropagated_loss: float, external loss
        :return: None
        """

        self.back_loss += backpropagated_loss * self.activation_deriv(self.z)

    
    def get_trainable_params(self):

        """
        :return list, if trainable is True then return a list with the current 
        values for the weights and bias
        """
        

        if self.trainable:
            
            self.trainable_params[:-1] = self.weights[0]
            self.trainable_params[-1] = self.bias
        
        
        return self.trainable_params


    def initialize(self, nb_inputs: int, idx: int):

        """
        initialize the weights value and check the consistency if they are already initialized
        :param nb_inputs: int, number of inputs
        :param idx: int, id of the component
        :return: None
        """
        
        self.nb_input = nb_inputs
        self.id = str(idx)
        self.ext_inputs = np.zeros((nb_inputs, 1))

        # weights not initialized yet -> create and record
        if not isinstance(self.weights, np.ndarray):
            self.weights = np.abs(np.random.normal(1, 1 / np.sqrt(nb_inputs + 1), (1, nb_inputs)))

        # weights already initialized, check
        elif self.weights.shape[1] != nb_inputs:
            raise ValueError(f'weights of shape {self.weights.shape} do not match the number of inputs [{nb_inputs:d}]')

        # if the weights are trainable, then update the trainable record adjusting for the number of inputs
        if 'w' in self.DNA['more']['trainable_params']:
                        
            self.nb_trainable = self.nb_input + 1
            self.trainable_params = np.zeros(self.nb_trainable)  # number of weights + one bias
            
            # weights
            self.trainable_names = [f'w{i+1}' for i in range(nb_inputs)]
            
            # bias
            self.trainable_names += ['b'] 
            
            # tuple
            self.trainable_names = tuple(self.trainable_names)
            
        #
        self.update_dna()
        # 
        self.initialization_flag = True
            
            
    def update_dna(self):
        
        """
        update the dna with the new parameters
        :return None
        """
        
        self.DNA['attributes']['w'] = self.weights.tolist()
        self.DNA['attributes']['b'] = self.bias




class ProteinCond(Protein):

    """
    a Molecule takes an input and returns an output by using a specific activity kernel
    """

    def __init__(self, dna: dict, verbose=False):

        super().__init__(dna=dna)
        self.check_dna()
        self.verbose = verbose

        # param
        self.tau = self.DNA['attributes']['tau']
        self.taug = self.DNA['attributes']['taug']
        self.Eq = self.DNA['attributes']['Eq']
        self.Epeak = self.DNA['attributes']['Epeak']

        # var
        self.inputs = np.zeros((self.nb_input, 1))
        self.z = self.Eq
        self.g = 0
        self.activity = self.Eq
        
        # activation
        if 'activation' in dna['more']:
            activation_name = dna['more']['activation'] 
        else:
            activation_name = 'none'
            self.DNA['more']['activation'] = 'none'
        self.activation, self.activation_deriv = activation_functions[activation_name]


        print('\n@MoleculeCond', end='')
        if self.trainable:
            print(' [trainable]')
        else:
            print()
            
            
    def step(self):
        
        """
        receive an input and the state is updated
        :return: None
        """
        
        self.z += (self.Eq - self.z) / self.tau + self.g * (self.Epeak - self.z)
        self.g += (np.dot(self.weights, self.ext_inputs) \
                               - self.g) / self.taug

        self.activity = self.activation(self.z)

        self.ext_inputs *= 0


    def update(self):

        """
        the trainable parameters are updated [to edit]
        :return: None
        """

        if 'tau' in self.trainable_params:
            self.tau -= self.lr * self.back_loss * (self.Eq - self.z)

        if 'taug' in self.trainable_params and len(self.g) > 1:
            self.taug += self.lr * self.back_loss * self.g * (self.Epeak - self.z)

        self.back_loss *= 0
        
        
    def add_loss(self, backpropagated_loss: float):
        
        """
        record another loss value from back-propagated from downstream
        :param backpropagated_loss: float, external loss
        :return: None
        """

        self.back_loss += backpropagated_loss * self.activation_deriv(self.z)


    def check_dna(self):

        """
        check if the DNA contains the right keys
        :return: None
        """

        attributes_keys = tuple(self.DNA['attributes'].keys())
        if not attributes_keys.__contains__('tau') or not attributes_keys.__contains__('Eq') or \
                not attributes_keys.__contains__('taug') or not attributes_keys.__contains__('Epeak'):

            raise ValueError('DNA does not contain one or more required keys')


    def get_trainable_params(self):

        """
        :return: ndarray shape=(nb_trainable,)
        """

        k = 0
        for i, param in enumerate(self.trainable_names):
            if param == 'tau':
                self.trainable_params[i] = self.tau

            elif param == 'taug':
                self.trainable_params[i] = self.taug

            elif param == 'Eq':
                self.trainable_params[i] = self.Eq

            elif param == 'Epeak':
                self.trainable_params[i] = self.Epeak

            elif param == 'w1':
                self.trainable_params[i] = self.weights[k]
                k += 1

        return self.trainable_params
    
    
    def initialize(self, nb_inputs: int, idx: int):

        """
        initialize the weights value and check the consistency if they are already initialized
        :param nb_inputs: int, number of inputs
        :param idx: int, id of the component
        :return: None
        """
        
        self.nb_input = nb_inputs
        self.id = str(idx)
        self.ext_inputs = np.zeros((nb_inputs, 1))

        # weights not initialized yet
        if not isinstance(self.weights, np.ndarray):
            self.weights = np.abs(np.random.normal(1, 1 / np.sqrt(nb_inputs + 1), (1, nb_inputs)))

        # weights already initialized, check
        elif self.weights.shape[1] != nb_inputs:
            raise ValueError(f'weights of shape {self.weights.shape} do not match the number of inputs [{nb_inputs:d}]')

        # if the weights are trainable, then update the trainable record adjusting for the number of inputs
        if 'w' in self.DNA['more']['trainable_params']:
            
            self.nb_trainable = self.nb_input + 1
            self.trainable_params = np.zeros(self.nb_trainable)  # number of weights + one bias
            
            # weights
            self.trainable_names = [f'w{i+1}' for i in range(nb_inputs)]
            
            # tuple
            self.trainable_names = tuple(self.trainable_names)
            
        #
        self.update_dna()
    
        # 
        self.initialization_flag = True
        
        
    def update_dna(self):
        
        """
        update the dna with the new parameters
        :return: None
        """
        
        self.DNA['attributes']['w'] = self.weights.tolist()
        self.DNA['attributes']['tau'] = self.tau
        self.DNA['attributes']['Eq'] = self.Eq
        self.DNA['attributes']['taug'] = self.taug
        self.DNA['attributes']['Epeak'] = self.Epeak
        

    def reset(self):

        """
        reset the run-time variables
        :return: None
        """

        self.z = [self.Eq]
        self.g = [0]

    @staticmethod
    def prod(x: list):

        if len(x) == 0:
            return 0

        x0 = x[0]
        for xi in x[1:]:
            x0 = x0 * xi

        return x0

    
class ProteinSpike(Protein):

    """
    a Molecule that takes a rate and returns a spike, if generated
    """

    def __init__(self, dna: dict, verbose=True):

        # dna
        super().__init__(dna=dna)
        self.verbose = verbose
        
        # param
        # if weights already defined in the DNA, use those
        if 'attributes' in tuple(self.DNA.keys()):
            self.scale = self.DNA['attributes']['scale'] if 'scale' in \
                tuple(self.DNA['attributes'].keys()) else 0
        else:
            self.scale = 0
            
        
        if verbose:
            print(f'\n@ProteinSpk [{1000*self.scale:.0f}ms]')
            
            

    def step(self):

        """
        receive an input and the state is updated
        :return: None
        """

        self.activity = np.random.binomial(n=1, p=self.ext_inputs)
        
        self.ext_inputs *= 0
        
        
    def collect_input(self, inputs: np.ndarray):

        """
        receive and store the inputs
        :param inputs: np.ndarray
        :return: None
        """

        # check input
        if inputs.item() < 0 or inputs.item() > 1:
            raise ValueError(f'{inputs} is not a valid rate')    
            
        # 
        self.ext_inputs[:] = self.scale * inputs
        
        
    def initialize(self, nb_inputs: int, idx: int):

        """
        initialize the weights value and check the consistency if they are already initialized
        :param nb_inputs: int, number of inputs
        :param idx: int, id of the component
        :return: None
        """
        
        self.nb_input = nb_inputs
        if nb_inputs > 1:
            raise ValueError(f'{nb_inputs} input rates number not valid')
        
        self.id = str(idx)
        self.ext_inputs = np.zeros((1, 1))
      
        # 
        self.initialization_flag = True
      

    def reset(self):

        """
        reset the run-time variables
        :return: None
        """

        self.activity *= 0
        

class ProteinPoly(Protein):

    def __init__(self, dna: dict, verbose=False):

        super().__init__(dna=dna)

        # parameters
        self.W = self.DNA['attributes']['W']
        self.Eq = self.DNA['attributes']['Eq']

        self.scale = 1 / 100
        self.sign = 1 if self.Eq == 0 else -1
        self.Eq *= self.scale

        # bool, if True the provided input is used [scaled] otherwise it is non-zero onset is used as delay,
        # default False
        self.proper = self.DNA['more']['proper']

        # var
        self.inputs = np.zeros((self.nb_input, 1))
        self.x = 1000  # time from the input onset
        self.z = self.Eq

        print('\n@MoleculePoly', end='')
        if self.trainable:
            print(' [trainable]')
        else:
            print()

    def step(self):

        """
        a non-zero input will trigger the start of a delay counter from the input onset,
        a zero input will just make the delay counter make one timestep
        :return: None
        """

        u = np.dot(self.w, self.inputs).item()

        # the actual input is the scaled provided input
        if self.proper:
            self.x = u * self.scale

        # the actual input is the delay from the input onset
        else:
            # non-zero input -> reset delay
            if self.inputs.item() != 0:
                self.x = 0

            # delay steps
            else:
                self.x += 1 * self.scale

        # state
        self.z = np.clip(self.Eq + self.sign * np.clip(sum([w * (self.x ** i) for i, w in enumerate(self.W)]), 0, 1), 0,
                         1)

    def backward(self, error: np.ndarray):

        """
        backward pass, the error is used to update the weights of the polynomial
        :param error: 1d np.ndarray
        :return: None
        """

        # scaling
        error = error.item() * self.scale

        # non-trainable weights
        if 'W' not in self.trainable_params:
            return

        # trainable weights
        # gradient w_i := lr * (dLoss / dz) * (dz / dw_i) -> lr * (y - z) * (sign * x**i)
        for i in range(len(self.W)):
            self.W[i] += self.lr * error * (self.sign * self.x ** i)

    def check_dna(self):

        """
        check if the DNA contains the right keys
        :return: None
        """

        attributes_keys = tuple(self.DNA['attributes'].keys())
        more_keys = tuple(self.DNA['more'])
        if not attributes_keys.__contains__('W') or not attributes_keys.__contains__('Eq') or \
                not attributes_keys.__contains__('beta') or \
                not more_keys.__contains__('lr') or not more_keys.__contains__('trainable_params') or\
                not attributes_keys.__contains__('proper'):

            raise ValueError('DNA does not contain one or more required keys')

    def collect_input(self, inputs: np.ndarray):

        """
        receive and store the inputs
        :param inputs: np.ndarray
        :return: None
        """

        self.inputs = inputs

    def get_trainable_params(self):

        """
        :return: ndarray shape=(nb_trainable,)
        """

        if 'w' in self.trainable_params:

            for i, w in enumerate(self.W):
                self.track[i] = w

        return self.track

    def get_output(self):

        """
        :return: float, scaled output
        """

        return self.z / self.scale

    def reset(self):

        """
        reset the run-time variables
        :return: None
        """

        self.z = 0
        self.x = 1000


class ProteinPlasticity(Protein):

    """
    base class for Protein implementing a plasticity rule
    """

    def __init__(self, dna: dict):

        """
        :param dna: dict,
        :param verbose: bool
        """

        # dna
        super().__init__(dna=dna)
        self.substrate_name = 'ProteinPlasticity'

        # plasticity variables
        self.internals = np.zeros(2)

    def step(self):

        """
        receive and input and compute the output as a weighted sum
        :return: None
        """

        self.activity = np.dot(self.weights, self.ext_inputs)

    def collect_internals(self, internals: np.ndarray):

        """
        collect internal variables relevant for the plasticity rule
        :param internals: np.ndarray, array of variables of interest
        :return: None
        """

        self.internals = internals

    @staticmethod
    def get_substrate_name():

        """
        :return: str, "ProteinPlasticity"
        """

        return 'ProteinPlasticity'


class ProteinPlasticitySTDP(ProteinPlasticity):

    """
    ProteinPlasticity implementing an STDP rule
    """

    def __init__(self, dna: dict, verbose=False):

        """
        :param dna: dict,
        :param verbose: bool
        """

        # dna
        super().__init__(dna=dna)

        # plasticity parameters
        self.A_plus = self.DNA['attributes']['A+']
        self.A_minus = self.DNA['attributes']['A-']
        self.magnitudes = np.array([self.DNA['attributes']['a+'], self.DNA['attributes']['a-']])
        self.tau_tr = self.DNA['attributes']['tau_tr']
        self.tau_stdp = self.DNA['attributes']['tau_stdp']

        # variable parameters
        self.traces = np.zeros(2)  # <-------------------- for now considers only 1 input and 1 cell spike channel

        self.stdp = 0.

        if verbose:
            print('\n@ProteinPlasticitySTDP', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()

    def update(self):

        """
        STDP step
        :return: None
        """

        self.traces = self.traces - self.traces / self.tau_tr + self.magnitudes * self.internals[:2]
        self.stdp = self.A_plus * self.traces[0] * self.internals[1] + self.A_minus * self.traces[1] * self.internals[0]

        if self.trainable:
            self.weights = self.weights + self.lr * self.stdp


class ProteinPlasticityReward(ProteinPlasticity):
    """
    ProteinPlasticity implementing an [dopamine] reward-based STDP rule
    """

    def __init__(self, dna: dict, verbose=False):

        """
        :param dna: dict,
        :param verbose: bool
        """

        # dna
        super().__init__(dna=dna)

        # plasticity parameters
        self.A_plus = self.DNA['attributes']['A+']
        self.A_minus = self.DNA['attributes']['A-']
        self.magnitudes = np.array([self.DNA['attributes']['a+'], self.DNA['attributes']['a-']])
        self.magnitude_rew = self.DNA['attributes']['a_rew']
        self.tau_tr = self.DNA['attributes']['tau_tr']
        self.tau_stdp = self.DNA['attributes']['tau_stdp']
        self.tau_rew = self.DNA['attributes']['tau_re']

        # variable parameters
        self.traces = np.zeros(2)   # <-------------------- for now considers only 1 input, 1 cell spike channel and
                                    # 1 reward channel
        self.trace_rew = 0.
        self.stdp = 0.

        if verbose:
            print('\n@ProteinPlasticitySTDP', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()

    def update(self):

        """
        STDP step
        :return: None
        """

        self.traces = self.traces - self.traces / self.tau_tr + self.magnitudes * self.internals[:2]
        self.trace_rew = self.trace_rew - self.trace_rew / self.tau_rew + self.magnitude_rew * self.internals[2]
        self.stdp = self.A_plus * self.traces[0] * self.internals[1] + self.A_minus * self.traces[1] * self.internals[0]

        if self.trainable:
            self.weights = self.weights + self.lr * self.stdp * self.trace_rew



protein_dict = {'exp': lambda dna, verbose: ProteinExp(dna=dna, verbose=verbose),
                'base': lambda dna, verbose: ProteinBase(dna=dna, verbose=verbose),
                'expbeta': lambda dna, verbose: ProteinExpBeta(dna=dna, verbose=verbose),
                'cond': lambda dna, verbose: ProteinCond(dna=dna, verbose=verbose),
                'poly': lambda dna, verbose: ProteinPoly(dna=dna, verbose=verbose),
                'spike': lambda dna, verbose: ProteinSpike(dna=dna, verbose=verbose) 
                }


#### CELLS ####


class Cell(SubstrateStructure):

    def __init__(self, dna: dict, verbose=True):

        # set up
        super().__init__(dna=dna, verbose=verbose)
        self.check_dna()
        self.substrate_name = 'Cell'

        if verbose:
            print('\n@Cell', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()
    
            print(f'\nnb_components: {self.nb_components}\nnb_inputs: {self.nb_input}\nnb_outputs: {self.nb_output}'
                  f'\nnb_trainable: {self.nb_trainable}')


    def check_dna(self):

        """
        check if the DNA contains the right keys
        :return: None
        """

        # general keys
        keys = tuple(self.DNA.keys())
        if not keys.__contains__('components') or not keys.__contains__('connections') or \
                not keys.__contains__('attributes') or not keys.__contains__('more'):

            raise ValueError('DNA does not contain one or more requires general keys')

        # <more> keys
        more_keys = tuple(self.DNA['more'])
        if not more_keys.__contains__('nb_in') or not more_keys.__contains__('nb_out'):

            raise ValueError('DNA does not contain one or more required <more> keys')

        # warning
        if self.DNA['components'].__len__() == 0:
            warnings.warn('Empty Cell: no components detected')

        if self.DNA['connections'].__len__() == 0:
            warnings.warn('Disconnected components: no connections detected')

        if self.DNA['more']['nb_in'] == 0:
            warnings.warn('Autonomous Cell: zero inputs specified')

        if self.DNA['more']['nb_out'] == 0:
            warnings.warn('Isolated Cell: zero outputs specified')
            

    @staticmethod
    def get_substrate_name():

        """
        :return: str
        """

        return 'Cell'



#### NETWORK ####


class Network(SubstrateStructure):

    def __init__(self, dna: dict):

        # set up
        super().__init__(dna=dna)



    def __init__(self, dna=5, actions=3):

        self.brain = Network(n=dna)
        self.activity = 0
        self.actions = np.zeros(actions)

    def step(self, inputs: np.ndarray):

        s = inputs

        self.brain.step()
        self.activity = self.brain.get_output()

    def get_action(self):

        # print(self.activity.T)

        return self.actions[np.argmax(self.activity)]
