import numpy as np
import json
import warnings


#### SUBSTRATE GENERATOR ####

# load libraries
with open(r"/Users/daniekru/Research/lab/samples/proteins.json", 'r') as f:
    proteins_library = json.loads(f.read())

with open(r"/Users/daniekru/Research/lab/samples/cells.json", 'r') as f:
    cells_library = json.loads(f.read())

def load_dna(dna: tuple):

    """
    load a DNA from the record and return it in its complete form 

    Parameters
    ----------
    dna : tuple 
        format ('Substrate_name', {'family': str,
                                   'id': int})

    Returns
    -------
    dna : dict 
    """
  
    # check dna 
    assert isinstance(dna, tuple) or isinstance(dna, list), f"invalid DNA format, it shall be a tuple or list [{dna}]"
    assert dna[0] in ('Protein', 'Cell', 'Network'), f"{dna[0]} is an invalid substrate.Class"
    assert 'family' in dna[1], "DNA family not specified"

    substrate_name = dna[0]
    family_name = dna[1]['family']

    # a complete DNA has been provided <-- no 'id'
    if 'id' not in dna[1].keys():
        
        return dna 
    
    # a saved substrate DNA has been queried 
    else:
        id_name = dna[1]['id']
        try:
            if substrate_name == 'Protein':
                recorded_specs = proteins_library[family_name][id_name]
            elif substrate_name == 'Cell':
                recorded_specs = cells_library[family_name][id_name]

                # merge cell specific keys 
                dna[1]['components'] = recorded_specs['components']
                dna[1]['connections'] = recorded_specs['connections']

            elif substrate_name == 'Network':
                raise NotImplementedError("Network substrate class not implemented yet, hopefully soon")
            else:
                raise NotImplementedError(f"{substrate_name} is not a valid substrate")

        except KeyError:
            raise KeyError(f"{substrate_name}.{family_name} class does not have a saved id {id_name}")
        
        # merge 
        dna[1]['params'] = recorded_specs['params']
        dna[1]['more'] = recorded_specs['more']

    #
    return dna 


def generate_substrate(dna: tuple, verbose=False):

    """
    generate a substrate given a DNA

    Parameters
    ----------
    dna: tuple
        with structure as (substrate_name, {})
    verbose: bool
        if True each substrate and its components will print their stamp
    
    Returns 
    -------
    substrate : class.Substrate object
        or one of its child classes
    """

    # load/complete dna 
    dna = load_dna(dna=dna)

    substrate_name = dna[0]
    family_name = dna[1]['family']

    if substrate_name == 'Protein':
        substrate = protein_dict[family_name](dna[1], verbose=verbose)

    elif substrate_name == 'Cell':
        substrate = cell_dict[family_name](dna=dna[1], verbose=verbose)

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

class Substrate:

    """
    base Substrate class:
        an object endowed with internal dynamics and input-output channels,
        whose definition is encoded in a DNA
    """

    def __init__(self, dna: dict):

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
        self.id = '0'
        self.substrate_class = 'Substrate'
        self.substrate_family = 'root'
        self.substrate_id = '0'

        # hyper-parameters
        self.lr = 0.

        # numbers
        self.nb_inputs = 0
        
        # training
        self.trainable_names = []
        self.nb_trainable = 0
        self.trainable_params = None
        self.trainable = False

        self.original_params = {}

        # variables
        self.ext_inputs = None
        self.activity = 0.
       
        # backward
        self.back_loss = np.zeros((1, 1))
        
        # initialization
        self.initialization_flag = False

        # check 
        self._substrate_initialization()

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
        assert 'more' in keys, "missing general key 'more' in DNA"

        # check attributes keys
        attributes_keys = tuple(self.DNA['more'].keys())
        assert 'nb_int' in attributes_keys, "missing attribute 'nb_int' in DNA"

        # set trainable params 
        for param in self.DNA['more']['trainable_params']:
            self.trainable_names += [param]
        self.nb_trainable = len(self.trainable_names)

        if self.nb_trainable > 0:
            self.trainable_params = np.zeros(self.nb_trainable)
            self.trainable = True

        # optionally available <more> keys
        self.lr = self.DNA['more']['lr'] if 'lr' in self.DNA['more'].keys() else 0.

        ### INITIALIZATION ###
        self.nb_inputs = self.DNA['more']['nb_int']
        self.ext_inputs = np.zeros((self.nb_inputs, 1))

    def _update_dna(self):
        
        """
        update the dna with the new parameters
        Returns
        -------
        None
        """
        
        pass
    
    def step(self):

        """
        receive an input and the state is updated [to edit]
        
        Returns
        -------
        None
        """

        pass

    def update(self):

        """
        the trainable parameters are updated [to edit]
        
        Returns
        -------
        None
        """

        pass

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

        self.ext_inputs = inputs

    def add_loss(self, backpropagated_loss: np.ndarray):
        
        """
        record another loss value from back-propagated from downstream
        
        Parameters
        ----------
        backpropagated_loss : np.ndarray
            external loss
        
        Returns 
        -------
        None
        """

        self.back_loss += backpropagated_loss
    
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

        self.id = str(idx)

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

    def get_loss(self) -> float:
        

        """
        compute and return the loss at this node
        
        Returns
        -------
        loss : np.ndarray
        """

        return self.back_loss

    def get_output(self):

        """
        Returns
        -------
        output : float
            default activity
        """

        return self.activity
    
    def get_activity(self):

        """
        return the output state
        
        Returns 
        -------
        output : float
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

    def get_substrate_name(self):

        """
        Returns
        -------
        substrate_name : str
        """

        return f"{self.substrate_class}.{self.substrate_family}.{self.substrate_id}"
   
    def get_nb_inout(self):

        """
        Returns
        -------
        nb_inputs : int
        """

        return self.nb_inputs

    def get_dna(self):
        
        """
        update the DNA with the current parameters and returns it
        
        Returns
        -------
        DNA : dict
        """
        
        self._update_dna()

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

        self.ext_inputs *= 0
        self.activity *= 0.


#### SUBSTRATE STRUCTURE ####

class SubstrateStructure(Substrate):
    
    """ base SubstrateStructure class
    
    a Substrate object having a structure made of more elementary substrates, 
    its components, which made up the internal dynamics (and input-output) of 
    the object through pre-defined connections 
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
        super().__init__(dna=dna)
        self.substrate_class = 'Structure'
        
        # hyperparams
        self.cycles = 1
        
        # structure
        self.components = None 
        self.connections = None
        self.connectivity_matrix = np.zeros(0)
        self.nb_components = 0
        self.idx_out = 0
        self.nb_outputs = 0
        
        self.trainable_components = []
        
        # grapher
        self.grapher = False
        self.livestream = False
        
        # var
        self.activity = np.zeros(self.nb_inputs + self.nb_components)
        self.output = np.zeros(self.nb_outputs)

        # check 
        self._substrate_structure_initialization()
        self._build_structure()
        self._update_structure_dna()

        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()

    
            print(f'\nnb_components: {self.nb_components}\nnb_inputs: '
                  '{self.nb_input}\nnb_outputs: {self.nb_output}'
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
        assert 'connections' in structure_keys, "missing key 'connections'"

        # attributes_keys check 
        more_key = tuple(self.DNA['more'].keys())
        assert 'idx_out' in more_key, "missing attribute 'idx_out'"
        assert 'nb_out' in more_key, "missing attribute 'nb_out'"
        assert 'cycles' in more_key, "missing attribute 'cycles'"
        
        self.idx_out = self.DNA['more']['idx_out']
        self.nb_outputs = self.DNA['more']['nb_out']
        self.cycles = self.DNA['more']['cycles']

        # 
        self.initialization_flag = True

    def _build_structure(self):

        """
        create each component from the DNA and store them in a list
        
        Returns
        -------
        None
        """
        
        # alphabet for the paramters name
        alphabet = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 'k')

        # list of components and connections
        dna_components = self.DNA['components']
        dna_connections = self.DNA['connections']

        self.components = []
        self.connections = []
        
        # create each component
        for idx, protein_gene in enumerate(dna_components):
            protein = generate_substrate(dna=protein_gene)
            self.components += [protein]
            self.trainable_components += [idx] * int(protein_gene[1]['more']['trainable_params'].__len__() > 0)

        # trainable
        self.trainable = self.trainable_components.__len__() > 0

        # number of inputs and components
        self.nb_components = len(self.components)  # components
        tot = len(self.components) + self.nb_inputs   # total

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

            # re-initialize each component and add its index
            self.components[i].re_initialize(nb_inputs=int(self.connectivity_matrix[i + self.nb_inputs].sum()))
            self.components[i].add_idx(idx=i)

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

        # warning
        if self.DNA['components'].__len__() == 0:
            warnings.warn('Empty Structure: no components detected')

        if self.DNA['connections'].__len__() == 0:
            warnings.warn('Disconnected components: no connections detected')

        if self.DNA['more']['nb_int'] == 0:
            warnings.warn('Autonomous Structure: zero inputs specified')

        if self.DNA['more']['nb_out'] == 0:
            warnings.warn('Isolated Structure: zero outputs specified')

    def step(self):

        """
        the internal components of the cell have their state updated

        Returns
        -------
        None
        """

        for cycle in range(self.cycles):

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
            for idx, component in enumerate(self.components[-self.nb_outputs:]):
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
        
        Returns
        -------
        None
        """

        if self.back_loss.sum() == 0:
            return

        # loss at the output nodes
        for k in range(self.nb_outputs):
            self.components[-k-1].add_loss(backpropagated_loss=self.back_loss[k])

        # BACKPROPAGATION #
        # starting from the output nodes, loop over all the components
        for i in range(self.nb_components - 1, 0, - 1):

            loss_i = self.components[i].get_loss()[0]

            # loop over the input nodes to i and add its loss
            for k, j in enumerate(np.where(self.connectivity_matrix[i + \
                                                    self.nb_inputs] == 1)[0]):
                self.components[j - self.nb_inputs].add_loss(backpropagated_loss=loss_i[k])  # indexed at j

        # parameter update
        for idx in range(self.nb_components):
            self.components[idx].update()
        
        # reset
        self.back_loss *= 0

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
    
    def add_grapher(self):
        
        """
        add a mix.tools.Grapher object to live stream the activiy
        
        Returns
        -------
        None
        """
        
        self.grapher = Grapher(connections=self.connections,
                               nb_input=self.nb_input,
                               nb_output=self.nb_output)
        
    def show_graph(self):
       
        """
        plot the graph of connections if a grapher object has been added
        
        Returns
        -------
        None
        """
        
        if not self.grapher:
            warnings.warn("no grapher object has been added, no plot generated")
            return
        
        self.grapher.draw_graph()
        
    def get_grapher(self):
       
        """
        Returns
        -------
        object : class.Grapher 
            object with its graph data relative to this substrate structure;
            return False if no grapher has been addedd
        """
        
        return self.grapher

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
        super().__init__(dna=dna)
        self.substrate_class = 'Protein'
        self.substrate_family = 'root'

        # if weights already defined in the DNA, use those
        #if 'params' in tuple(self.DNA.keys()):
        #    if 'w' in tuple(self.DNA['params'].keys()):
        #        self.weights = np.array(self.DNA['params']['w']).reshape(1, \
        #                                                    -1).astype(float)
        #        self.initial_weights = self.weights.copy()

        # variables
        self.z = 0.

        # parameters
        self.weights = None
        
        self.original_params['w'] = None

        # activation
        self.activation = lambda x: x
        self.activation_deriv = lambda x: 1
        
        #
        self._protein_initialization()

        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}', end='')
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
        if 'activation' in self.DNA['more']:
            activation_name = self.DNA['more']['activation'] 
            self.activation, self.activation_deriv = activation_functions[activation_name]

        ### WEIGHT INITIALIZATION ###

        # weights in the DNA and no new weights are needed 
        if 'w' in self.DNA['params'] and not new_weights:

            if len(self.DNA['params']['w']) != self.nb_inputs:
                warnings.warn(f"weights size {len(self.DNA['params']['w'])} does not match input size {self.nb_inputs}, resizing", RuntimeWarning)

                # re-launch function but now generating new weights
                self._protein_initialization(new_weights=True)
                return

            self.weights = np.array(self.DNA['params']['w']).reshape(1, -1)

        # new weights
        else:
            self.weights = np.around(np.abs(np.random.normal(1, 1 / np.sqrt(\
                                    self.nb_inputs + 1), (1, self.nb_inputs))), 4)
        
        
        # weight original copy
        self.original_params['w'] = self.weights.copy()

        ### TRAINABLE RECORD ###
        # if the weights are trainable, then update the trainable record 
        # adjusting for the number of inputs
        if 'w' in self.DNA['more']['trainable_params']:

            self.nb_trainable = self.nb_inputs
            
            # number of weights + one bias
            self.trainable_params = np.zeros(self.nb_trainable)  

            # weights
            self.trainable_names = [f'w{i + 1}' for i in range(self.nb_inputs)]

            # tuple
            self.trainable_names = tuple(self.trainable_names)

        #
        self.trainable = self.nb_trainable > 0
        
        #
        self.initialization_flag = True

    def _update_protein_dna(self):

        """ 
        update the dna with the current parameters [weights]

        Returns
        -------
        None
        """

        self.DNA['params']['w'] = self.weights.tolist()

    # def collect_input(self, inputs: np.ndarray):

     #   """
     #   receive and store the inputs
        
     #   Parameters
     #   ----------
     #   inputs: np.ndarray
        
     #   Returns
     #   -------
     #   None
     #   """

     #   # dont store absent inputs
     #   if np.any(inputs != 0):
     #       self.ext_inputs[:, 0] = inputs

    def re_initialize(self, nb_inputs: int):

        """
        re-initialization of the weights and parameters with a new number 
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

        self._update_dna()

    def get_trainable_params(self):

        """
        Returns
        -------
        trainable_params : list
            if trainable is True then return a list with the current
            values for the weights and bias
        """

        if self.trainable:
            self.trainable_params = self.weights[0].copy()
            
        return self.trainable_params

    def reset_protein(self):

        """
        reset the value of the variables
        
        Returns
        -------
        None
        """

        self.z *= 0
        self.activity *= 0
        self.weights = self.original_params['w'].copy()


class ProteinExp(Protein):

    """ a Protein subclass
    
    its activity relies on exponential decay
    """

    def __init__(self, dna: dict, verbose=True):
        
        # dna
        super().__init__(dna=dna)
        self.substrate_family = 'exp'

        # param
        self.tau = 0
        self.Eq = 0
        self.sign = 0

        #
        self._protein_exp_initialization()
        self._update_dna()

        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()

    def _protein_exp_initialization(self):

        """
        check if the DNA contains the right parameters keys and assign \
        their values
        
        Returns
        -------
        None
        """

        parameters_keys = tuple(self.DNA['params'].keys())

        assert 'tau' in parameters_keys, "missing parameter 'tau'"
        assert 'Eq' in parameters_keys, "missing parameter 'Eq'"

        # param
        self.tau = self.DNA['params']['tau']
        self.Eq = self.DNA['params']['Eq']
        self.sign = 2 * (self.Eq < 0.5) - 1
        
        self.original_params['tau'] = self.tau 
        self.original_params['Eq'] = self.Eq 
        self.original_params['sign'] = self.sign

        # var
        self.z = self.Eq

    def _update_dna(self):
        
        """
        update the dna with the new parameters
        
        Returns
        -------
        None
        """
        
        self._update_protein_dna()
        self.DNA['params']['tau'] = self.tau
        self.DNA['params']['Eq'] = self.Eq

    def step(self):

        """
        receive an input and the state is updated
        
        Returns
        -------
        None
        """
        
        self.z = abs(self.z + (self.Eq - self.z) / self.tau + self.sign \
                     * np.dot(self.weights, self.ext_inputs))
        
        self.activity = self.activation(self.z)
        if self.verbose:
            print(f'\n#\nx: {self.inputs}\nEq: {self.Eq}\nz: {self.z}\na: '
                  '{self.activity}')

        self.ext_inputs *= 0

    def update(self):

        """
        the trainable parameters are updated 
        
        Returns
        -------
        None
        """
        
        # print('updating ', self.back_loss, self.z)

        if self.back_loss.item() == 0:
            return

        if 'w1' in self.trainable_names:
            self.weights += self.lr * self.back_loss * self.ext_inputs.T

        if 'tau' in self.trainable_names: 
            self.tau -= self.lr * self.back_loss * (self.Eq - self.z)
            
            if self.verbose:
                print(f'tau: {self.tau}\nbloss: {self.back_loss}\ngrad: '
                      '{self.lr * self.back_loss * (self.Eq - self.z)}')
            
        if 'Eq' in self.trainable_names:
            self.Eq += self.lr * self.back_loss / self.tau
            
            
        self.back_loss *= 0

    def add_loss(self, backpropagated_loss: float):
        
        """
        record another loss value from back-propagated from downstream
        
        Parameters
        ----------
        backpropagated_loss: float
            external loss
        
        Returns
        -------
        None
        """

        self.back_loss += backpropagated_loss * self.activation_deriv(self.z)
        if self.verbose:
            print(f'backprop loss={backpropagated_loss}\nactivation '
                  'derivative={self.activation_deriv(self.z)} ')

    def get_trainable_params(self):

        """
        Returns
        -------
        trainable_params : np.ndarray
            shape=(nb_trainable,)
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

    def reset(self):

        """
        reset the run-time variables
        
        Returns
        -------
        None
        """

        self.reset_protein()
        self.tau = self.original_params['tau']
        self.Eq = self.original_params['Eq']
        self.sign = self.original_params['sign']


class ProteinLinear(Protein):

    """ a Protein subclass
    
    it is designed as an artificial [linear] neuron : W • x + b
    """

    def __init__(self, dna: dict, verbose=False):

        """
        Parameters
        ----------
        dna : dict 
        verbose : bool 

        Returns
        -------
        None
        """
        
        # dna
        super().__init__(dna=dna)
        self.substrate_family = 'linear'

        # param
        self.bias = 0. 
        
        #
        self._protein_linear_initialization()
        self._update_dna()

        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()

    def _protein_linear_initialization(self):

        """
        check if the DNA contains the right parameters [bias] andd initialize their values 

        Returns
        -------
        None
        """

        parameters_keys = tuple(self.DNA['params'].keys())

        assert 'b' in parameters_keys, "missing parameter 'b'"

        # initialiization
        self.bias = np.array([self.DNA['params']['b']]).reshape(1, -1)
        self.original_params['b'] = self.bias.copy()

    def step(self):

        """
        receive an input and the state is updated
        
        Returns
        -------
        None
        """

        self.z = np.dot(self.weights, self.ext_inputs) + self.bias
        
        # dont step if absent inputs
        self.activity = self.activation(self.z) * int(np.any(
                                                        self.ext_inputs != 0)) 
       
    def update(self):

        """
        the trainable parameters are updated
        
        Returns
        -------
        None
        """

        if 'w1' in self.trainable_names:
            self.weights += self.lr * self.back_loss * self.ext_inputs.T
            self.bias += self.lr * self.back_loss

        # reset
        self.back_loss *= 0

    def add_loss(self, backpropagated_loss: float):
        
        """
        record another loss value from back-propagated from downstream

        Parameters
        ----------
        backpropagated_loss: float
            external loss
        
        Returns
        -------
        None
        """

        self.back_loss += backpropagated_loss * self.activation_deriv(self.z)

    
    def get_trainable_params(self):

        """
        Returns
        -------
        trainable_params : list
            if trainable is True then return a list with the \
            current values for the weights and bias
        """
        

        if self.trainable:
            
            self.trainable_params[:-1] = self.weights[0]
            self.trainable_params[-1] = self.bias
        
        
        return self.trainable_params

    def _update_dna(self):
        
        """
        update the dna with the new parameters
        
        Returns
        -------
        None
        """
        
        self._update_protein_dna()
        self.DNA['params']['b'] = self.bias

    def reset(self):

        """
        reset the run-time variables, weights and bias

        Returns
        -------
        None
        """

        self.reset_protein()
        self.bias = self.original_params['b'].copy()


class ProteinCond(Protein):

    """ a Protein subclass
    
    its dynamics are specified by a conductance-based voltage,
    its activity shape is a delayed onset and exponential decay
    """

    def __init__(self, dna: dict, verbose=False):

        """
        Parameters
        ----------
        dna : dict 
        verbose : bool 
            default False 

        Returns
        -------
        None
        """
        
        super().__init__(dna=dna)
        self.substrate_family = 'Cond'

        # param
        self.tau = 0
        self.taug = 0
        self.Eq = 0
        self.Epeak = 0

        # var
        self.g = 0
        
        #
        self._protein_cond_initialization()
        self._update_dna()

        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()                

    def _protein_cond_initialization(self):

        """
        check if the DNA contains the right parameters keys and \
        initialize their values 

        Returns
        -------
        None 
        """

        parameters_keys = tuple(self.DNA['params'].keys())

        assert 'tau' in parameters_keys, "missing parameter 'tau'"
        assert 'taug' in parameters_keys, "missing parameter 'taug'"
        assert 'Eq' in parameters_keys, "missing parameter 'Eq'"
        assert 'Epeak' in parameters_keys, "missing parameter 'Epeak'"

        # initialization 
        self.tau = self.DNA['params']['tau']
        self.taug = self.DNA['params']['taug']
        self.Eq = self.DNA['params']['Eq']
        self.Epeak = self.DNA['params']['Epeak']
        
        self.original_params['tau'] = self.tau
        self.original_params['taug'] = self.taug 
        self.original_params['Eq'] = self.Eq 
        self.original_params['Epeak'] = self.Epeak 

        # var 
        self.z = self.Eq

    def _update_dna(self):
        
        """
        update the dna with the current parameters
        
        Returns
        -------
        None
        """
        
        self._update_protein_dna()
        self.DNA['params']['tau'] = self.tau
        self.DNA['params']['Eq'] = self.Eq
        self.DNA['params']['taug'] = self.taug
        self.DNA['params']['Epeak'] = self.Epeak        

    def step(self):
        
        """
        receive an input and the state is updated
        
        Returns
        -------
        None
        """
        
        if np.isnan(self.z + (self.Eq - self.z) / self.tau + (self.Eq - \
                                                        self.z) / self.tau):
            print('cond pre ', (self.Eq - self.z) / self.tau, (self.Eq - \
                                        self.z) / self.tau, self.z, self.tau)
            raise ValueError
        
        self.z += (self.Eq - self.z) / self.tau + (self.Epeak - self.z) * self.g
        if np.isnan(self.z):
            print('cond ', self.activity, self.z, self.g, self.ext_inputs, \
                  self.weights, self.tau, self.taug)
            print()
            raise ValueError
        self.g += (np.dot(self.weights, self.ext_inputs) \
                               - self.g) / self.taug

        self.activity = self.activation(self.z)
        
        self.ext_inputs *= 0 

    def update(self):

        """
        the trainable parameters are updated
        
        Returns
        -------
        None
        """

        if 'tau' in self.trainable_params:
            self.tau -= self.lr * self.back_loss * (self.Eq - self.z)

        if 'taug' in self.trainable_params and len(self.g) > 1:
            self.taug += self.lr * self.back_loss * self.g * (self.Epeak - self.z)

        self.back_loss *= 0        
        
    def add_loss(self, backpropagated_loss: float):
        
        """
        record another loss value from back-propagated from downstream

        Parameters
        ----------
        backpropagated_loss : float
            external loss
        
        Returns
        -------
        None
        """

        self.back_loss += backpropagated_loss * self.activation_deriv(self.z)

    def get_trainable_params(self):

        """
        Returns
        -------
        trainable_params : ndarray
            shape=(nb_trainable,)
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
        initialize the weights value and check the consistency if they are 
        already initialized
        :param nb_inputs: int, number of inputs
        :param idx: int, id of the component
        :return: None
        """
        
        self.nb_input = nb_inputs
        self.id = str(idx)
        self.ext_inputs = np.zeros((nb_inputs, 1))

        # weights not initialized yet
        if not isinstance(self.weights, np.ndarray):
            self.weights = np.abs(np.random.normal(1, 1 / np.sqrt(nb_inputs \
                                                         + 1), (1, nb_inputs)))

        # weights already initialized, check
        elif self.weights.shape[1] != nb_inputs:
            raise ValueError(f'{self.substrate_name + self.substrate_type} - '
                             'weights of shape {self.weights.shape} do not match the number of inputs [{nb_inputs:d}]')

        # if the weights are trainable, then update the trainable record 
        # adjusting for the number of inputs
        if 'w' in self.DNA['more']['trainable_params']:
            
            self.nb_trainable = self.nb_input + 1
            
            # number of weights
            self.trainable_params = np.zeros(self.nb_trainable)  
            
            # weights
            self.trainable_names = [f'w{i+1}' for i in range(nb_inputs)]
            
            # tuple
            self.trainable_names = tuple(self.trainable_names)
            
        #
        self.update_dna()
    
        # 
        self.initialization_flag = True        
        
    def reset(self):

        """
        reset the run-time variables
        
        Returns
        -------
        None
        """
        
        self.reset_protein()
        self.g = 0
        self.tau = self.original_params['tau']
        self.taug = self.original_params['taug']
        self.Eq = self.original_params['Eq']
        self.Epeak = self.original_params['Epeak']


class ProteinSpike(Protein):

    """ a Protein subclass
    
    aits activity consists of a spike generator using a input in [0, 1] as 
    probability parameter of a Binomial distribution
    """

    def __init__(self, dna: dict, verbose=True):

        """
        Parameters
        ----------
        dna : dict
        verbose : bool

        Returns
        -------
        None
        """
        
        # dna
        super().__init__(dna=dna)
        self.substrate_family = 'Spike'
        
        # param
        self.scale = 0.
        self.rest_rate = 0.
        self.ext_rate = 0.

        #
        self._protein_spike_initialization()
        self._update_dna()
                    
        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}', end='')
            print(f' [{self.scale}ms] [{1000*self.rest_rate:.0f}Hz]', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()
    
    def _protein_spike_initialization(self):

        """
        check if the DNA contains the right parameters keys and \
        initialize their values 

        Returns
        -------
        None
        """

        parameters_keys = tuple(self.DNA['params'].keys())

        assert 'scale' in parameters_keys, "missing parameter 'scale'"
        assert 'rest_rate' in parameters_keys, "missing parameter 'rest_rate'"

        # initialization 
        self.scale = self.DNA['params']['scale']
        self.rest_rate = self.DNA['params']['rest_rate'] / self.scale 
        
        self.original_params['scale'] = self.scale 
        self.original_params['rest_rate'] = self.rest_rate

        # the weight is meaningless for this protein
        self.weights *= 0

    def step(self):

        """
        receive an input and the state is updated
        
        Returns
        -------
        None
        """
        
        try:
            self.activity = np.random.binomial(n=1, p=np.clip(self.ext_inputs, 
                                                    self.rest_rate, 0.9))
        except ValueError:
            print('probability of ', np.clip(self.ext_inputs, self.rest_rate,
                                        0.9), self.ext_inputs, self.rest_rate)
            raise ValueError
        
        self.ext_inputs *= 0
        
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
        
        # check input
        if np.any(inputs < 0) and np.any(inputs > 1):
            raise ValueError(f'{inputs} is not a valid rate')   
        
        assert inputs.size == self.nb_inputs, f"external inputs with size {inputs.size} does not match expected input size {self.nb_inputs}"

        # 
        self.ext_inputs = inputs.copy() / self.scale
        
    def _update_dna(self):

        """ 
        update the dna with the current parameters

        Returns
        -------
        None
        """
        
        self._update_protein_dna()
        self.DNA['params']['scale'] = self.scale
        self.DNA['params']['rest_rate'] = self.rest_rate

    def reset(self):

        """
        reset the run-time variables
        
        Returns
        -------
        None
        """

        self.reset_protein()
        self.scale = self.original_params['scale']
        self.rest_rate = self.original_params['rest_rate']


class ProteinPlasticity(Protein):

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
        super().__init__(dna=dna)
        self.substrate_family = 'Plasticity'

        # plasticity variables
        self.externals = 0 
        
        #
        self._protein_plasticity_initialization()
        self._update_protein_dna()

        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()

    def _protein_plasticity_initialization(self):

        """
        minimal DNA check of the class.ProteinPlasticity, \
        presence of nb of externals: factors used for the plasticity \
        implemented

        Returns
        -------
        None
        """
        
        attributes_keys = self.DNA['more']
        assert 'nb_extf' in attributes_keys, "missing attribute 'nb_extf'"

        # initialization
        self.nb_extf = self.DNA['more']['nb_extf']
        self.externals = np.zeros(self.nb_extf)

    def step(self):

        """
        receive and input and compute the output as a weighted sum
        
        Returns
        -------
        None
        """

        self.activity = np.dot(self.weights, self.ext_inputs)

    def update(self):

        """
        Returns
        -------
        None
        """

        if self.trainable:
            self.weights = self.weights + self.lr * self.back_loss

            self.back_loss *= 0

    def collect_internals(self, internals: np.ndarray):

        """
        collect internal variables relevant for the plasticity rule
        Parameters
        ----------
        internals: np.ndarray
            array of variables of interest
        
        Returns
        -------
        None
        """

        self.internals = internals

    def reset_protein_plasticity(self):

        """
        reset the value of run-time variables ['externals']

        Returns
        -------
        None
        """
        self.reset_protein()
        self.externals *= 0 


class ProteinPlasticitySTDP(ProteinPlasticity):

    """ a ProteinPlasticity subclass
    
    it implements an STDP rule
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
        super().__init__(dna=dna)
        self.substrate_family = 'PlasticitySTDP'

        # plasticity parameters
        self.A_plus = 0
        self.A_minus = 0
        self.magnitudes = None
        self.tau_tr = 0
        self.tau_stdp = 0

        # variable parameters
        # for now considers only 1 input and 1 cell spike channel
        self.traces = np.zeros(2)  
        self.stdp = 0.

        #
        self._protein_stdp_initalization()
        self._update_dna()
       
        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()

    def _protein_stdp_initalization(self):

        """
        check if the DNA contains the right parameters_keys and \
        intialize them 

        Returns
        -------
        None
        """
        
        parameters_keys = self.DNA['params']

        assert 'A+' in parameters_keys, "missing parameter 'A+'"
        assert 'A-' in parameters_keys, "missing parameter 'A-'"
        assert 'a+' in parameters_keys, "missing parameter 'a+'"
        assert 'a-' in parameters_keys, "missing parameter 'a-'"
        assert 'tau_tr' in parameters_keys, "missing parameter 'tau_tr"
        assert 'tau_stdp' in parameters_keys, "missing parameter 'tau_stdp'"

        self.A_plus = self.DNA['params']['A+']
        self.A_minus = self.DNA['params']['A-']
        self.magnitudes = np.array([self.DNA['params']['a+'], 
                                    self.DNA['params']['a-']])
        self.tau_tr = self.DNA['params']['tau_tr']
        self.tau_stdp = self.DNA['params']['tau_stdp']
        
        self.original_params['A+'] = self.A_plus 
        self.original_params['A-'] = self.A_minus
        self.original_params['magnitudes'] = self.magnitudes.copy() 
        self.original_params['tau_tr'] = self.tau_tr
        self.original_params['tau_stpd'] = self.tau_stdp 

    def _update_dna(self):

        """
        update the dna with the current parameters 

        Returns
        -------
        None
        """

        self._update_protein_dna()
        self.DNA['params']['A+'] = self.A_plus 
        self.DNA['params']['A-'] = self.A_minus 
        self.DNA['params']['a+'] = self.magnitudes[0]
        self.DNA['params']['a-'] = self.magnitudes[1]
        self.DNA['params']['tau_tr'] = self.tau_tr 
        self.DNA['params']['tau_stdp'] = self.tau_stdp 

    def update(self):

        """
        STDP step
        
        Returns
        -------
        None
        """

        self.traces = self.traces - self.traces / self.tau_tr + \
            self.magnitudes * self.internals[:2]
        self.stdp = self.A_plus * self.traces[0] * self.internals[1] \
            + self.A_minus * self.traces[1] * self.internals[0]

        if self.trainable:
            self.weights = self.weights + self.lr * self.stdp
    
    def reset(self):

        """
        reset the run-time variables and original parameters 

        Returns
        -------
        None
        """
        
        self.reset_protein_plasticity()

        self.A_plus = self.original_params['A+']
        self.A_minus = self.original_params['A-']
        self.magnitudes = self.original_params['magnitudes']
        self.tau_tr = self.original_params['tau_tr']
        self.tau_stdp = self.original_params['tau_stdp']
       

class ProteinPlasticityReward(ProteinPlasticity):
    
    """ a ProteinPlasticity subclass
    
    it implements a [dopamine] reward-based STDP rule
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
        super().__init__(dna=dna)
        self.substrate_family = 'PlasticityReward'

        # plasticity parameters
        self.A_plus = 0
        self.A_minus = 0
        self.magnitudes = None
        self.magnitude_rew = 0
        self.tau_tr = 0
        self.tau_stdp = 0
        self.tau_rew = 0

        # variable parameters
        # for now considers only 1 input, 1 cell spike channel and 1 reward 
        self.traces = np.zeros(2)
        self.trace_rew = 0.
        self.stdp = 0.

        self._protein__reward__initalization()
        
        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()

    def _protein__reward__initalization(self):

        """
        check if the DNA contains the right parameters_keys and \
        intialize them 

        Returns
        -------
        None
        """
        
        parameters_keys = self.DNA['params']

        assert 'A+' in parameters_keys, "missing parameter 'A+'"
        assert 'A-' in parameters_keys, "missing parameter 'A-'"
        assert 'a+' in parameters_keys, "missing parameter 'a+'"
        assert 'a-' in parameters_keys, "missing parameter 'a-'"
        assert 'u+' in parameters_keys, "missing parameter 'u+'"
        assert 'tau_tr' in parameters_keys, "missing parameter 'tau_tr"
        assert 'tau_stdp' in parameters_keys, "missing parameter 'tau_stdp'"
        assert 'tau_rew' in parameters_keys, "missing parameter 'tau_rw'"

        self.A_plus = self.DNA['params']['A+']
        self.A_minus = self.DNA['params']['A-']
        self.magnitudes = np.array([self.DNA['params']['a+'], 
                                    self.DNA['params']['a-']])
        self.magnitude_rew = self.DNA['params']['u+']
        self.tau_tr = self.DNA['params']['tau_tr']
        self.tau_stdp = self.DNA['params']['tau_stdp']
        self.tau_rew = self.DNA['params']['tau_rew']

        self.original_params['A+'] = self.A_plus 
        self.original_params['A-'] = self.A_minus
        self.original_params['magnitudes'] = self.magnitudes.copy() 
        self.original_params['u+'] = self.magnitude_rew
        self.original_params['tau_tr'] = self.tau_tr
        self.original_params['tau_stpd'] = self.tau_stdp
        self.original_params['tau_rew'] = self.tau_rew

    def _update_dna(self):

        """
        update the dna with the current parameters 

        Returns
        -------
        None
        """

        self._update_protein_dna()

        self.DNA['params']['A+'] = self.A_plus 
        self.DNA['params']['A-'] = self.A_minus 
        self.DNA['params']['a+'] = self.magnitudes[0]
        self.DNA['params']['a-'] = self.magnitudes[1]
        self.DNA['params']['u+'] = self.magnitude_rew
        self.DNA['params']['tau_tr'] = self.tau_tr 
        self.DNA['params']['tau_stdp'] = self.tau_stdp
        self.DNA['params']['tau_rew'] = self.tau_rew 

    def update(self):

        """
        reward STDP step
        
        Returns
        -------
        None
        """

        self.traces = self.traces - self.traces / self.tau_tr + self.magnitudes * self.internals[:2]
        self.trace_rew = self.trace_rew - self.trace_rew / self.tau_rew + self.magnitude_rew * self.internals[2]
        self.stdp = self.A_plus * self.traces[0] * self.internals[1] + self.A_minus * self.traces[1] * self.internals[0]

        if self.trainable:
            self.weights = self.weights + self.lr * self.stdp * self.trace_rew

    def reset(self):

        """
        reset the run-time variables and original parameters 

        Returns
        -------
        None
        """
        
        self.reset_protein_plasticity()

        self.A_plus = self.original_params['A+']
        self.A_minus = self.original_params['A-']
        self.magnitudes = self.original_params['magnitudes']
        self.magnitude_rew = self.original_params['u+']
        self.tau_tr = self.original_params['tau_tr']
        self.tau_stdp = self.original_params['tau_stdp']
        self.tau_rew = self.original_params['tau_rew']


class ProteinPlasticityHebb(ProteinPlasticity):

    """ a ProteinPlasticity subclass

    it implements a simple Hebbian rule 
    """

    def __init__(self, dna: dict, verbose=False):

        """
        Parameters
        ----------
        dna : dict 
        verbose : bool 
            default False 

        Returns
        -------
        None 
        """

        # dna
        super().__init__(dna=dna)
        self.substrate_family = 'PlasticityHebb'

        # param
        self.tau = 0
        self.taug = 0
        self.Eq = 0
        self.Epeak = 0

        # var
        self.g = 0

        #
        self._protein_hebb_initialization()
        self._update_dna()

        if verbose:
            print(f"\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}", end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()

    def _protein_hebb_initialization(self):

        """
        check the DNA parameters and initialize their values

        Returns
        -------
        None
        """

        parameters_keys = tuple(self.DNA['params'].keys())

        assert 'tau' in parameters_keys, "missing parameter 'tau'"
        assert 'taug' in parameters_keys, "missing parameter 'taug'"
        assert 'Eq' in parameters_keys, "missing parameter 'Eq'"
        assert 'Epeak' in parameters_keys, "missing parameter 'Epeak'"

        self.tau = self.DNA['params']['tau']
        self.taug = self.DNA['params']['taug']
        self.Eq = self.DNA['params']['Eq']
        self.Epeak = self.DNA['params']['Epeak']

        self.z = self.Eq
        self.activity = self.Eq 
        
        # input activity | the basal input doesn't count       
        self.externals = np.zeros((1, self.nb_inputs - 1))  
        
        self.original_params['tau'] = self.tau 
        self.original_params['taug'] = self.taug 
        self.original_params['Eq'] = self.Eq 
        self.original_params['Epeak'] = self.Epeak

    def _update_dna(self):
        
        """
        update the dna with the new parameters
        
        Returns
        -------
        None
        """
        
        self._update_protein_dna()

        self.DNA['params']['tau'] = self.tau
        self.DNA['params']['taug'] = self.taug
        self.DNA['params']['Eq'] = self.Eq
        self.DNA['params']['Epeak'] = self.Epeak

    def step(self):
        
        """
        receive an input and the state is updated
        
        Returns
        -------
        None
        """
        z = self.z + (self.Eq - self.z) / self.tau + self.g * (self.Epeak - self.z) 
        if np.isnan(z):
            print('cond pre ', z)
            print('metrics ', self.activity, self.z, self.g, self.ext_inputs, \
                  self.weights, self.tau, self.taug)
            print()
            raise ValueError
 
        self.z = z
        self.g += (np.dot(self.weights, self.ext_inputs) - self.g) / self.taug
        self.activity = self.activation(self.z)
        
        self.ext_inputs *= 0
        
    def update(self):

        """
        Hebbian step 
        
        Returns
        -------
        None 
        """
        
        # actually now it implements a simple Oja rule 
        if self.trainable:

            # exp(w_i) / sum(exp(w_i))_i
            hierarchy = np.exp(self.weights[0, 1:]) / np.exp(self.weights[0, 1:]).sum()

            # the first weight is for the basal noise 
            # the remaining weight entries are actual inputs
            delta = self.activity * self.externals - self.weights[0, 1:] * self.externals ** 2

            # w * h / max(h) - w 
            drift = self.weights[0, 1:] * (hierarchy / hierarchy.max()) - self.weights[0, 1:]
            
            # update 
            self.weights[0, 1:] = self.weights[0, 1:] + self.lr * delta + 0.4 * drift * (0.003 + delta)   

            # normalization 
            #self.weights[0, 1:] = self.weights[0, 1:] + 0.35 * drift * (delta) 
        
        # reset input activity
        self.externals *= 0

    def collect_input(self, inputs: np.ndarray):

        """
        receive and store the inputs.
        
        Parameters
        ----------
        inputs: np.ndarray
            shape=(nb_inputs,)
        
        Returns
        -------
        None
        """

        # dont store absent inputs
        if np.any(inputs != 0):
            self.ext_inputs[:, 0] = inputs.copy()
            self.externals[0, :] = inputs[1:].copy() 

    def reset(self):

        """
        reset the run-time variables
        
        Returns
        -------
        None
        """
        
        self.reset_protein_plasticity()
        self.g *= 0
        self.externals *= 0



protein_dict = {'exp': lambda dna, verbose: ProteinExp(dna=dna, verbose=verbose),
                'linear': lambda dna, verbose: ProteinLinear(dna=dna, verbose=verbose),
                'expbeta': lambda dna, verbose: ProteinExpBeta(dna=dna, verbose=verbose),
                'cond': lambda dna, verbose: ProteinCond(dna=dna, verbose=verbose),
                'poly': lambda dna, verbose: ProteinPoly(dna=dna, verbose=verbose),
                'spike': lambda dna, verbose: ProteinSpike(dna=dna, verbose=verbose),
                'plasticity_root': lambda dna, verbose: ProteinPlasticity(dna=dna, verbose=verbose),
                'plasticity_stdp': lambda dna, verbose: ProteinPlasticitySTDP(dna=dna, verbose=verbose),
                'plasticity_reward': lambda dna, verbose: ProteinPlasticityReward(dna=dna, verbose=verbose),
                'plasticity_hebb': lambda dna, verbose: ProteinPlasticityHebb(dna=dna, verbose=verbose),
                'oja': lambda dna, verbose: ProteinPlasticityHebb(dna=dna, verbose=verbose),
                }


#### CELLS ####


class Cell(SubstrateStructure):
    
    """ a Structure subclass
    
    it implements a network of elementary Proteins component
    
    """

    def __init__(self, dna: dict, verbose=False):
        
        """
        Parameters
        ----------
        dna: dict
        verbose, bool
        
        Returns
        -------
        None
        """

        # set up
        super().__init__(dna=dna)
        self.substrate_family = 'Cell'
        
        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()
    
            print(f'\nnb_components: {self.nb_components}\nnb_inputs: '
                  f'{self.nb_inputs}\nnb_outputs: {self.nb_outputs}'
                  f'\nnb_trainable: {self.nb_trainable}')


class CellPlasticity(SubstrateStructure):
    
    """ a Structure subclass,
    
    it is like a Cell.class but implementing one or more ProteinPlasticity
    """
    
    def __init__(self, dna: dict, verbose=False):

        """
        Parameters
        ----------
        dna : dict 
        verbose : bool
            default False 

        Returns
        -------
        None
        """

        # set up
        super().__init__(dna=dna)
        self.substrate_family = 'CellPlasticity'

        # output indexes 
        if 'idx_out' in dna['more'].keys():
            self.idx_out = dna['more']['idx_out']
            if isinstance(self.idx_out, int):
                self.idx_out = [self.idx_out]
                print('tupled!')
        else:
            warnings.warn("indexes of the output nodes not provided")
            self.idx_out = ()

      
        # plasticity variables [output, inputs..., modulators...]
        self.idx_intf = 0
        self.idx_extf = 0
        self.internals = None
        self.externals = None 
        
        # initialize
        self._cell_plasticity_initialization()
        self._build_structure()
        
        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()
    
            print(f'\nnb_components: {self.nb_components}\nnb_inputs: '
                  '{self.nb_input}\nnb_outputs: {self.nb_output}'
                  f'\nnb_trainable: {self.nb_trainable}')

    def _cell_plasticity_initialization(self):

        """
        check DNA requirements and initialize class terms 

        Returns
        -------
        None
        """

        # attributes keys check 
        attributes_keys = self.DNA['more']
        assert 'idx_intf' in attributes_keys, "missing attribute key 'idx_intf'"
        assert 'idx_extf' in attributes_keys, "missing attribute key 'idx_extf'"

        # initialize
        self.idx_intf = self.DNA['more']['idx_intf']
        self.idx_extf = self.DNA['more']['idx_extf']
        self.internals = np.zeros(self.idx_intf)
        self.externals = np.zeros(self.idx_extf)

    def _build_structure(self):
        
        """
        create each component from the DNA and store them in a list
        
        Returns
        -------
        None
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

            if 'more' in tuple(protein_gene[1].keys()):
                self.trainable_components += [idx] * int(protein_gene[1][
                    'more']['trainable_params'].__len__() > 0)

        # trainable
        self.trainable = self.trainable_components.__len__() > 0

        # number of inputs and components
        self.nb_input = self.DNA['more']['nb_in']  # inputs
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

        # initialize the weights for each component
        for i in range(self.nb_components):
            self.components[i].initialize(nb_inputs=int(
                self.connectivity_matrix[i + self.nb_input].sum()), idx=i)

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
                        
        self.trainable_params = np.zeros(self.nb_trainable)  
        
        # internals
        self.internals = np.zeros(self.nb_output + self.nb_input)
        
        # var
        self.activity = np.zeros(self.nb_input + self.nb_components)
        self.output = np.zeros(self.nb_output)
        
        # 
        self.initialization_flag = True
        
    def step(self):

        """
        the internal components of the cell have their state updated
        
        Returns
        -------
        None
        """

        for cycle in range(self.cycles):

            # get the activity of each component, input to its downstream neighbours
            for idx, component in enumerate(self.components):
                self.activity[self.nb_input + idx] = component.get_output()

            # each component loads the inputs from its inputs sources +
            # each component steps
            for i in range(self.nb_components):

                # indexes of the inputs to component idx
                inputs_j = np.where(self.connectivity_matrix[i + 
                                                             self.nb_input] != 0)[0]

                if len(inputs_j) == 0:
                    continue

                # load
                self.components[i].collect_input(inputs=
                                                 self.activity.take(inputs_j).T)

                # step
                self.components[i].step()
                
                
                # live graph
                if self.grapher and self.livestream:
                        
                    self.grapher.live_graph(activations=self.activity)
                

            # define the output as the activity of the output components
            for l, idx in enumerate(self.idx_out):
                self.output[l] = self.components[idx].get_output()

            # set plasticity variables
            self.set_internals()

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

        # if self.back_loss.sum() == 0:
        #    return

        # loss at the output nodes
        for k in range(self.nb_output):
            self.components[-k-1].add_loss(backpropagated_loss=self.back_loss[k])

        # BACKPROPAGATION #
        # starting from the output nodes, loop over all the components except
        # the input nodes
        for i in range(self.nb_components - 1 - self.nb_input, 0, - 1):

            loss_i = self.components[i].get_loss()[0]

            # loop over the input nodes to i and add its loss
            for k, j in enumerate(np.where(self.connectivity_matrix[i + 
                                                    self.nb_input] == 1)[0]):
                self.components[j - self.nb_input].add_loss(
                    backpropagated_loss=loss_i[k])  # indexed at j

        # the input nodes collect their plasticity variables
        for k in range(self.nb_input):
            self.components[k].collect_internals(internals=self.internals)

        # parameter update
        for idx in range(self.nb_components):
            self.components[idx].update()
        
        # reset
        self.back_loss *= 0        
        
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
        self.activity[:self.nb_input] = inputs
        
    def set_internals(self):
        
        """
        collect the values of the internal variables of interests
        
        Returns
        -------
        None
        """
    
        for i, idx in enumerate(self.idx_internals):
            self.internals[i] = self.activity[idx]
    
    def reset(self):

        """
        reset current variables 

        Returns
        -------
        None
        """
        self.reset_structure()

        self.internals *= 0 
        self.externals *= 0



cell_dict = {'root': lambda dna, verbose: Cell(dna=dna, verbose=verbose),
             'plasticity': lambda dna, verbose: CellPlasticity(dna=dna, 
                                                               verbose=verbose)}
    

