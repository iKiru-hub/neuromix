import numpy as np 
import warnings
from neuromix.brain import templates as T

# available protein classes
PROTEIN_LIST = ['Protein', 'ProteinPlasticity']


class ProteinExp(T.Protein):

    """ a Protein subclass
    
    its activity relies on exponential decay
    """

    def __init__(self, dna: dict, verbose=True):
        
        # dna
        super().__init__(dna=dna)
        self.substrate_family = 'exp'

        # param
        self._tau = 0
        self._Eq = 0
        self._sign = 0

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
        self._tau = self.DNA['params']['tau']
        self._Eq = self.DNA['params']['Eq']
        self._sign = 2 * (self._Eq < 0.5) - 1
        
        self._original_params['tau'] = self._tau 
        self._original_params['Eq'] = self._Eq 
        self._original_params['sign'] = self._sign

        # var
        self._z = self._Eq

    def _update_dna(self):
        
        """
        update the dna with the new parameters
        
        Returns
        -------
        None
        """
        
        self._update_protein_dna()
        self.DNA['params']['tau'] = self._tau
        self.DNA['params']['Eq'] = self._Eq

    def step(self):

        """
        receive an input and the state is updated
        
        Returns
        -------
        None
        """
        
        self._z = abs(self._z + (self._Eq - self._z) / self._tau + self._sign \
                     * np.dot(self._weights, self._ext_inputs))
        
        self._activity = self._activation(self._z)
        if self.verbose:
            print(f'\n#\nx: {self._inputs}\n_Eq: {self._Eq}\nz: {self._z}\na: '
                  f'{self._activity}')

        self._ext_inputs *= 0

    def update(self):

        """
        the trainable parameters are updated 
        
        Returns
        -------
        None
        """
        
        # print('updating ', self._feedback, self._z)

        if self._feedback.item() == 0:
            return

        if 'w1' in self.trainable_names:
            self._weights += self._lr * self._feedback * self._ext_inputs.T

        if '_tau' in self.trainable_names: 
            self._tau -= self._lr * self._feedback * (self._Eq - self._z)
            
            if self.verbose:
                print(f'_tau: {self._tau}\nbloss: {self._feedback}\ngrad: '
                      f'{self._lr * self._feedback * (self._Eq - self._z)}')
            
        if 'Eq' in self.trainable_names:
            self._Eq += self._lr * self._feedback / self._tau
            
            
        self._feedback *= 0

    def add_feedback(self, ext_feedback: float):
        
        """
        record another loss value from back-propagated from downstream
        
        Parameters
        ----------
        ext_feedback: float
            external loss
        
        Returns
        -------
        None
        """

        self._feedback += ext_feedback * self._activation_deriv(self._z)
        if self.verbose:
            print(f'backprop loss={ext_feedback}\nactivation '
                  f'derivative={self._activation_deriv(self._z)} ')

    def get_trainable_params(self):

        """
        Returns
        -------
        trainable_params : np.ndarray
            shape=(nb_trainable,)
        """

        k = 0
        for i, param in enumerate(self.trainable_names):
            
            if param == '_tau':
                self.trainable_params[i] = self._tau

            elif param == '_Eq':
                self.trainable_params[i] = self._Eq

            elif param == 'w1':
                self.trainable_params[i:] = self._weights[k]
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
        self._tau = self._original_params['tau']
        self._Eq = self._original_params['Eq']
        self._sign = self._original_params['sign']


class ProteinLinear(T.Protein):

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
        self._bias = 0. 
        
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
        self._bias = np.array([self.DNA['params']['b']]).reshape(1, -1)
        self._original_params['b'] = self._bias.copy()

    def step(self):

        """
        receive an input and the state is updated
        
        Returns
        -------
        None
        """

        self._z = np.dot(self._weights, self._ext_inputs) + self._bias
        
        # dont step if absent inputs
        self._activity = self._activation(self._z) * int(np.any(
                                                        self._ext_inputs != 0)) 
       
    def update(self):

        """
        the trainable parameters are updated
        
        Returns
        -------
        None
        """

        if 'w1' in self.trainable_names:
            self._weights += self._lr * self._feedback * self._ext_inputs.T
            self._bias += self._lr * self._feedback

        # reset
        self._feedback *= 0

    def add_feedback(self, ext_feedback: float):
        
        """
        record another loss value from back-propagated from downstream

        Parameters
        ----------
        ext_feedback: float
            external loss
        
        Returns
        -------
        None
        """

        self._feedback += ext_feedback * self._activation_deriv(self._z)

    def get_trainable_params(self):

        """
        Returns
        -------
        trainable_params : list
            if trainable is True then return a list with the \
            current values for the _weights and bias
        """
        

        if self.trainable:
            
            self.trainable_params[:-1] = self._weights[0]
            self.trainable_params[-1] = self._bias
        
        
        return self.trainable_params

    def _update_dna(self):
        
        """
        update the dna with the new parameters
        
        Returns
        -------
        None
        """
        
        self._update_protein_dna()
        self.DNA['params']['b'] = self._bias

    def reset(self):

        """
        reset the run-time variables, _weights and bias

        Returns
        -------
        None
        """

        self.reset_protein()
        self._bias = self._original_params['b'].copy()


class ProteinCond(T.Protein):

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
        self._tau = 0
        self._taug = 0
        self._Eq = 0
        self._Epeak = 0

        # var
        self._g = 0
        
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
        self._tau = self.DNA['params']['tau']
        self._taug = self.DNA['params']['taug']
        self._Eq = self.DNA['params']['Eq']
        self._Epeak = self.DNA['params']['Epeak']
        
        self._original_params['tau'] = self._tau
        self._original_params['taug'] = self._taug 
        self._original_params['Eq'] = self._Eq 
        self._original_params['Epeak'] = self._Epeak 

        # var 
        self._z = self._Eq

    def _update_dna(self):
        
        """
        update the dna with the current parameters
        
        Returns
        -------
        None
        """
        
        self._update_protein_dna()
        self.DNA['params']['tau'] = self._tau
        self.DNA['params']['Eq'] = self._Eq
        self.DNA['params']['taug'] = self._taug
        self.DNA['params']['Epeak'] = self._Epeak        

    def step(self):
        
        """
        receive an input and the state is updated
        
        Returns
        -------
        None
        """
        
        if np.isnan(self._z + (self._Eq - self._z) / self._tau + (self._Eq - \
                                                        self._z) / self._tau):
            print('cond pre ', (self._Eq - self._z) / self._tau, (self._Eq - \
                                        self._z) / self._tau, self._z, self._tau)
            raise ValueError
        
        self._z += (self._Eq - self._z) / self._tau + (self._Epeak - self._z) * self._g
        if np.isnan(self._z):
            print('cond ', self._activity, self._z, self._g, self._ext_inputs, \
                  self._weights, self._tau, self._taug)
            print()
            raise ValueError
        self._g += (np.dot(self._weights, self._ext_inputs) \
                               - self._g) / self._taug

        self._activity = self._activation(self._z)
        
        self._ext_inputs *= 0 

    def update(self):

        """
        the trainable parameters are updated
        
        Returns
        -------
        None
        """

        if 'tau' in self.trainable_params:
            self._tau -= self._lr * self._feedback * (self._Eq - self._z)

        if 'taug' in self.trainable_params:
            self._taug += self._lr * self._feedback * self._g * (self._Epeak - self._z)

        self._feedback *= 0        
        
    def add_feedback(self, ext_feedback: float):
        
        """
        record another loss value from back-propagated from downstream

        Parameters
        ----------
        ext_feedback : float
            external loss
        
        Returns
        -------
        None
        """

        self._feedback += ext_feedback * self._activation_deriv(self._z)

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
                self.trainable_params[i] = self._tau

            elif param == 'taug':
                self.trainable_params[i] = self._taug

            elif param == 'Eq':
                self.trainable_params[i] = self._Eq

            elif param == 'Epeak':
                self.trainable_params[i] = self._Epeak

            elif param == 'w1':
                self.trainable_params[i] = self._weights[k]
                k += 1

        return self.trainable_params    
    
    def initialize(self, nb_inputs: int, idx: int):

        """
        initialize the _weights value and check the consistency if they are 
        already initialized

        Parameters
        ----------
        nb_inputs: int
            number of inputs
        idx: int
            id of the component
        
        Returns
        -------
        None
        """
        
        self.nb_input = nb_inputs
        self.id = str(idx)
        self._ext_inputs = np.zeros((nb_inputs, 1))

        # _weights not initialized yet
        if not isinstance(self._weights, np.ndarray):
            self._weights = np.abs(np.random.normal(1, 1 / np.sqrt(nb_inputs \
                                                         + 1), (1, nb_inputs)))

        # _weights already initialized, check
        elif self._weights.shape[1] != nb_inputs:
            raise ValueError(f'{self.substrate_name + self.substrate_type} - '
                             '_weights of shape {self.weights.shape} do not match the number of inputs [{nb_inputs:d}]')

        # if the _weights are trainable, then update the trainable record 
        # adjusting for the number of inputs
        if 'w' in self.DNA['attrb']['trainable_params']:
            
            self.nb_trainable = self.nb_input + 1
            
            # number of _weights
            self.trainable_params = np.zeros(self.nb_trainable)  
            
            # _weights
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
        self._g = 0
        self._tau = self._original_params['tau']
        self._taug = self._original_params['taug']
        self._Eq = self._original_params['Eq']
        self._Epeak = self._original_params['Epeak']


class ProteinSpike(T.Protein):

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
        self._scale = 0.
        self._rest_rate = 0.
        self._ext_rate = 0.

        #
        self._protein_spike_initialization()
        self._update_dna()
                    
        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}', end='')
            print(f' [{self._scale}ms] [{1000*self._rest_rate:.0f}Hz]', end='')
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
        self._scale = self.DNA['params']['scale']
        self._rest_rate = self.DNA['params']['rest_rate'] / self._scale 
        
        self._original_params['scale'] = self._scale 
        self._original_params['rest_rate'] = self._rest_rate

        # the weight is meaningless for this protein
        self._weights *= 0

    def step(self):

        """
        receive an input and the state is updated
        
        Returns
        -------
        None
        """
        
        try:
            self._activity = np.random.binomial(n=1, p=np.clip(self._ext_inputs, 
                                                    self._rest_rate, 0.9))
        except ValueError:
            print('probability of ', np.clip(self._ext_inputs, self._rest_rate,
                                        0.9), self._ext_inputs, self._rest_rate)
            raise ValueError
        
        self._ext_inputs *= 0
        
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
        self._ext_inputs = inputs.copy() / self._scale
        
    def _update_dna(self):

        """ 
        update the dna with the current parameters

        Returns
        -------
        None
        """
        
        self._update_protein_dna()
        self.DNA['params']['scale'] = self._scale
        self.DNA['params']['rest_rate'] = self._rest_rate

    def reset(self):

        """
        reset the run-time variables
        
        Returns
        -------
        None
        """

        self.reset_protein()
        self._scale = self._original_params['scale']
        self._rest_rate = self._original_params['rest_rate']


class ProteinPlasticity_stdp(T.ProteinPlasticity):

    """ a ProteinPlasticity subclass
    
    it implements an _stdp rule
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
        self.substrate_family = 'Plasticity_stdp'

        # plasticity parameters
        self._A_plus = 0
        self._A_minus = 0
        self._magnitudes = None
        self._tau_tr = 0
        self._tau__stdp = 0

        # variable parameters
        # for now considers only 1 input and 1 cell spike channel
        self._traces = np.zeros(2)  
        self._stdp = 0.

        #
        self._protein__stdp_initalization()
        self._update_dna()
       
        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()

    def _protein__stdp_initalization(self):

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

        self._A_plus = self.DNA['params']['A+']
        self._A_minus = self.DNA['params']['A-']
        self._magnitudes = np.array([self.DNA['params']['a+'], 
                                    self.DNA['params']['a-']])
        self._tau_tr = self.DNA['params']['tau_tr']
        self._tau_stdp = self.DNA['params']['tau_stdp']
        
        self._original_params['A+'] = self._A_plus 
        self._original_params['A-'] = self._A_minus
        self._original_params['magnitudes'] = self._magnitudes.copy() 
        self._original_params['tau_tr'] = self._tau_tr
        self._original_params['tau_stpd'] = self._tau__stdp 

    def _update_dna(self):

        """
        update the dna with the current parameters 

        Returns
        -------
        None
        """

        self._update_protein_dna()
        self.DNA['params']['A+'] = self._A_plus 
        self.DNA['params']['A-'] = self._A_minus 
        self.DNA['params']['a+'] = self._magnitudes[0]
        self.DNA['params']['a-'] = self._magnitudes[1]
        self.DNA['params']['tau_tr'] = self._tau_tr 
        self.DNA['params']['tau_stdp'] = self._tau__stdp 

    def update(self):

        """
        _stdp step
        
        Returns
        -------
        None
        """

        self._traces = self._traces - self._traces / self._tau_tr + \
            self._magnitudes * self._internals[:2]
        self._stdp = self._A_plus * self._traces[0] * self._internals[1] \
            + self._A_minus * self._traces[1] * self._internals[0]

        if self.trainable:
            self._weights = self.weights + self._lr * self._stdp
    
    def reset(self):

        """
        reset the run-time variables and original parameters 

        Returns
        -------
        None
        """
        
        self.reset_protein_plasticity()

        self._A_plus = self._original_params['A+']
        self._A_minus = self._original_params['A-']
        self._magnitudes = self._original_params['_magnitudes']
        self._tau_tr = self._original_params['tau_tr']
        self._tau__stdp = self._original_params['tau_stdp']


class ProteinPlasticityReward(T.ProteinPlasticity):
    
    """ a ProteinPlasticity subclass
    
    it implements a [dopamine] reward-based _stdp rule
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
        self._A_plus = 0
        self._A_minus = 0
        self._magnitudes = None
        self._magnitude_rew = 0
        self._tau_tr = 0
        self._tau__stdp = 0
        self._tau_rew = 0

        # variable parameters
        # for now considers only 1 input, 1 cell spike channel and 1 reward 
        self._traces = np.zeros(2)
        self._trace_rew = 0.
        self._stdp = 0.

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
        assert 'tau__stdp' in parameters_keys, "missing parameter 'tau_stdp'"
        assert 'tau_rew' in parameters_keys, "missing parameter 'tau_rw'"

        self._A_plus = self.DNA['params']['A+']
        self._A_minus = self.DNA['params']['A-']
        self._magnitudes = np.array([self.DNA['params']['a+'], 
                                    self.DNA['params']['a-']])
        self._magnitude_rew = self.DNA['params']['u+']
        self._tau_tr = self.DNA['params']['tau_tr']
        self._tau__stdp = self.DNA['params']['tau_stdp']
        self._tau_rew = self.DNA['params']['tau_rew']

        self._original_params['A+'] = self._A_plus 
        self._original_params['A-'] = self._A_minus
        self._original_params['_magnitudes'] = self._magnitudes.copy() 
        self._original_params['u+'] = self._magnitude_rew
        self._original_params['tau_tr'] = self._tau_tr
        self._original_params['tau_stpd'] = self._tau__stdp
        self._original_params['tau_rew'] = self._tau_rew

    def _update_dna(self):

        """
        update the dna with the current parameters 

        Returns
        -------
        None
        """

        self._update_protein_dna()

        self.DNA['params']['A+'] = self._A_plus 
        self.DNA['params']['A-'] = self._A_minus 
        self.DNA['params']['a+'] = self._magnitudes[0]
        self.DNA['params']['a-'] = self._magnitudes[1]
        self.DNA['params']['u+'] = self._magnitude_rew
        self.DNA['params']['_tau_tr'] = self._tau_tr 
        self.DNA['params']['_tau__stdp'] = self._tau__stdp
        self.DNA['params']['_tau_rew'] = self._tau_rew 

    def update(self):

        """
        reward _stdp step
        
        Returns
        -------
        None
        """

        self._traces = self._traces - self._traces / self._tau_tr + self._magnitudes * self._internals[:2]
        self._trace_rew = self._trace_rew - self._trace_rew / self._tau_rew + self._magnitude_rew * self._internals[2]
        self._stdp = self._A_plus * self._traces[0] * self._internals[1] + self._A_minus * self._traces[1] * self._internals[0]

        if self.trainable:
            self._weights = self.weights + self._lr * self._stdp * self._trace_rew

    def reset(self):

        """
        reset the run-time variables and original parameters 

        Returns
        -------
        None
        """
        
        self.reset_protein_plasticity()

        self._A_plus = self._original_params['A+']
        self._A_minus = self._original_params['A-']
        self._magnitudes = self._original_params['magnitudes']
        self._magnitude_rew = self._original_params['u+']
        self._tau_tr = self._original_params['_tau_tr']
        self._tau__stdp = self._original_params['_tau__stdp']
        self._tau_rew = self._original_params['_tau_rew']


class ProteinPlasticityHebb(T.ProteinPlasticity):

    """ a ProteinPlasticity subclass

    it i:mplements a simple Hebbian rule 
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
        self._tau = 0
        self._taug = 0
        self._Eq = 0
        self._Epeak = 0

        # var
        self._g = 0

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

        self._tau = self.DNA['params']['tau']
        self._taug = self.DNA['params']['taug']
        self._Eq = self.DNA['params']['Eq']
        self._Epeak = self.DNA['params']['Epeak']

        self._z = self._Eq
        self._activity = self._Eq 
        
        # input activity | the basal input doesn't count       
        self._externals = np.zeros((1, self.nb_inputs - 1))  
        
        self._original_params['tau'] = self._tau 
        self._original_params['taug'] = self._taug 
        self._original_params['Eq'] = self._Eq 
        self._original_params['Epeak'] = self._Epeak

    def _update_dna(self):
        
        """
        update the dna with the new parameters
        
        Returns
        -------
        None
        """
        
        self._update_protein_dna()

        self.DNA['params']['_tau'] = self._tau
        self.DNA['params']['_taug'] = self._taug
        self.DNA['params']['_Eq'] = self._Eq
        self.DNA['params']['_Epeak'] = self._Epeak

    def step(self):
        
        """
        receive an input and the state is updated
        
        Returns
        -------
        None
        """
        z = abs(self._z + (self._Eq - self._z) / self._tau + self._g * (self._Epeak - self._z)) 

        if np.isnan(z):
            print('cond pre ', z)
            print('metrics ', self._activity, self._z, self._g, self._ext_inputs, \
                  self._weights, self._tau, self._taug)
            print()
            raise ValueError
 
        self._z = z
        self._g += (np.dot(self._weights, self._ext_inputs) - self.g) / self._taug
        self._activity = self._activation(self._z)
        
        self._ext_inputs *= 0
        
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
            #hierarchy = np.exp(self._weights[0, 1:]) / np.exp(self.weights[0, 1:]).sum()

            # the first weight is for the basal noise 
            # the remaining weight entries are actual inputs
            #delta = self._activity * self._externals - self._weights[0, :] * self._externals ** 2
            delta = self._internals[2] * self._internals[:2] #- self._weights[0, :] * self._internals[:2] ** 2 

            # w * h / max(h) - w 
            #drift = self._weights[0, 1:] * (hierarchy / hierarchy.max()) - self.weights[0, 1:]
            
            # update 
            self._weights[0, :] = self._weights[0, :] + self._lr * delta #+ 0. * drift * (0.003 + delta)   

            # normalization 
            #self._weights[0, 1:] = self.weights[0, 1:] + 0.35 * drift * (delta) 
        
        # reset input activity
        self._externals *= 0
        self._internals *= 0

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
            self._ext_inputs[:, 0] = inputs.copy()
            self._externals[0, :] = inputs[1:].copy() 

    def reset(self):

        """
        reset the run-time variables
        
        Returns
        -------
        None
        """
        
        self.reset_protein_plasticity()
        self._g *= 0
        self._externals *= 0


class ProteinJump(T.Protein):

    """ a Protein subclass 

    its dynamics are specified by a jump function 
    
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
        self.substrate_family = 'Jump'

        # param
        self._jump_time = 0
        self._tau = 0
        self._var_jump = 0 

        # variables
        self._drift = 0
        self._feedback = 0
        self._pre_weights = np.zeros(1)
        self._post_weights = np.zeros(1)
        self._counter = 0 
        self._error = 0

        #
        self._protein_jump_initialization()
        self._update_dna()

        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()

    def _protein_jump_initialization(self):

        """
        check if the DNA contains the right parameters keys and \
        initialize their values 

        Returns
        -------
        None 
        """

        parameters_keys = tuple(self.DNA['params'].keys())

        assert 'tau' in parameters_keys, "missing parameter 'tau'"
        assert 'jump_time' in parameters_keys, "missing parameter 'jump_time'"
        assert 'var_jump' in parameters_keys, "missing parameter 'var_jump'"

        self._tau = self.DNA['params']['tau']
        self._jump_time = self.DNA['params']['jump_time']
        self._var_jump = self.DNA['params']['var_jump']

        # record the original parameters
        self._original_params['tau'] = self._tau 
        self._original_params['jump_time'] = self._jump_time 
        self._original_params['var_jump'] = self._var_jump

        # initialize the variables
        self._pre_weights = self._weights.copy()
        self._post_weights = self._weights.copy()
    
    def _update_dna(self):
        
        """
        update the dna with the current parameters
        
        Returns
        -------
        None
        """
        
        self._update_protein_dna()
        self.DNA['params']['tau'] = self._tau
        self.DNA['params']['jump_time'] = self._jump_time
        self.DNA['params']['var_jump'] = self._var_jump

    def _random_jump(self):

        """
        generate a random jump in the parameter space, by adding a noise vector \
        to the current parameters
        """

        # copy weight matrix before the jump 
        self._pre_weights = self._weights.copy()

        # generate a random jump 
        self._weights *= np.random.normal(1, self._var_jump, self._weights.shape)

        # copu weight matrix after the jump
        self._post_weights = self._weights.copy()

    def _step_drift(self):

        """
        compute and apply a drift vector to the weights 
        """

        # compute the drift vector 
        if self._feedback <= 0:
            self._weights += (self._pre_weights - self._weights) / self._tau

        else:
            self._weights = self._post_weights.copy()

    def step(self):
        
        """
        receive an input and the state is updated
        
        Returns
        -------
        None
        """
        
        # forward
        self.activity = self._weights @ self._ext_inputs
    
    def update(self):

        """
        the trainable parameters are updated
        
        Returns
        -------
        None
        """

        if 'w1' in self.trainable_names:

            self._counter += 1
            
            # jump
            if self._counter == self._jump_time:

                self._random_jump()
                self._counter = 0

            # drift
            self._step_drift()

            # reset
            self._feedback *= 0

            #self._error *= 0
    
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

        #self._error = ext_feedback - self._feedback 
        self._feedback = ext_feedback

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
                self.trainable_params[i] = self._tau

            elif param == 'jump_time':
                self.trainable_params[i] = self._jump_time

            elif param == 'var_jump':
                self.trainable_params[i] = self._var_jump

            elif param == f'w{k+1}':
                self.trainable_params[i] = self._weights[0, k]
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
        self._drift *= 0
        self._feedback *= 0
        self._pre_weights = self._weights.copy()
        self._post_weights = self._weights.copy()

        self._counter = 0
        self._jump_time = self._original_params['jump_time']
        self._tau = self._original_params['tau']
        self._var_jump = self._original_params['var_jump']


class ProteinJumpStep(T.ProteinPlasticity):

    """ a Protein subclass 

    its dynamics are specified by a jump function with step-wise drift
    
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
        self.substrate_family = 'JumpStep'

        # param
        self._jump_time = 0
        self._tau = 0
        self._var_jump = 0 

        # variables
        self._drift = 0
        self._feedback = 0
        self._pre_weights = np.zeros(1)
        self._post_weights = np.zeros(1)
        self._counter = 0 

        #
        self._lr_mean = 0.01
        self._lr_var = 0.01
        self._est_mean = 0.
        self._est_var = 0.

        # activation functions
        self._compute_jump_var = lambda x: max((1 - 3 * x, 0))
        self._compute_direction = lambda x: np.sign(x - 0.5)
        self._compute_direction = np.vectorize(self._compute_direction)

        #
        self._protein_jump_initialization()
        self._update_dna()

        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()

    def _protein_jump_initialization(self):

        """
        check if the DNA contains the right parameters keys and \
        initialize their values 

        Returns
        -------
        None 
        """

        parameters_keys = tuple(self.DNA['params'].keys())

        assert 'tau' in parameters_keys, "missing parameter 'tau'"
        assert 'jump_time' in parameters_keys, "missing parameter 'jump_time'"
        assert 'var_jump' in parameters_keys, "missing parameter 'var_jump'"

        self._tau = self.DNA['params']['tau']
        self._jump_time = self.DNA['params']['jump_time']
        self._var_jump = self.DNA['params']['var_jump']

        # optional parameters
        if "lr_mean" in parameters_keys:
            self._lr_mean = self.DNA['params']['lr_mean']
        else:
            self._lr_mean = 0.01
            self.DNA['params']['lr_mean'] = self._lr_mean

        if "lr_var" in parameters_keys:
            self._lr_var = self.DNA['params']['lr_var']
        else:
            self._lr_var = 0.01
            self.DNA['params']['lr_var'] = self._lr_var

        # record the original parameters
        self._original_params['tau'] = self._tau 
        self._original_params['jump_time'] = self._jump_time 
        self._original_params['var_jump'] = self._var_jump
        self._original_params['lr_mean'] = self._lr_mean
        self._original_params['lr_var'] = self._lr_var

        # initialize the variables
        self._pre_weights = self._weights.copy()
        self._post_weights = self._weights.copy()
    
    def _update_dna(self):
        
        """
        update the dna with the current parameters
        
        Returns
        -------
        None
        """
        
        self._update_protein_dna()
        self.DNA['params']['tau'] = self._tau
        self.DNA['params']['jump_time'] = self._jump_time
        self.DNA['params']['var_jump'] = self._var_jump

    def _random_jump(self):

        """
        generate a random jump in the parameter space, by adding a noise vector \
        to the current parameters
        """

        # copy weight matrix before the jump 
        self._pre_weights = self._weights.copy()

        # generate a random jump 
        self._var_jump = self._compute_jump_var(self._est_var)
        self._weights += self._compute_direction(np.random.binomial(1, self._est_mean, size=self._weights.shape)) * \
            np.random.normal(0, self._var_jump, size=self._weights.shape)

        # copy weight matrix after the jump
        self._post_weights = self._weights.copy()

    def _step_drift(self):

        """
        compute and apply a drift vector to the weights 
        """

        # compute the drift vector 
        if self._feedback <= 0:
            self._weights += (self._pre_weights - self._weights) / self._tau

        else:
            self._weights = self._post_weights.copy() # (self._post_weights - self._weights) / 3

    def step(self):
        
        """
        receive an input and the state is updated
        
        Returns
        -------
        None
        """
        
        # forward
        self.activity = self._weights @ self._ext_inputs
    
    def update(self):

        """
        the trainable parameters are updated
        
        Returns
        -------
        None
        """

        if 'w1' in self.trainable_names:

            self._counter += 1
            
            # jump
            if self._counter == self._jump_time:

                self._random_jump()
                self._counter = 0

            # drift
            self._step_drift()

            # reset
            self._feedback *= 0
    
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


        self._feedback = ext_feedback.copy() if isinstance(ext_feedback, np.ndarray) else ext_feedback

        # squash feedback from [-1, 1] to [0, 1]
        squashed_feedback = (ext_feedback + 1) / 2

        # estimate the mean and variance of the feedback
        self._est_mean += self._lr_mean * (squashed_feedback - self._est_mean)
        self._est_var += self._lr_var * (abs(squashed_feedback - self._est_mean) - self._est_var)

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
                self.trainable_params[i] = self._tau

            elif param == 'jump_time':
                self.trainable_params[i] = self._jump_time

            elif param == 'var_jump':
                self.trainable_params[i] = self._var_jump

            elif param == f'w{k+1}':
                self.trainable_params[i] = self._weights[0, k]
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
        self._drift *= 0
        self._feedback *= 0
        self._est_mean = 0
        self._est_var = 0
        self._pre_weights = self._weights.copy()
        self._post_weights = self._weights.copy()

        self._counter = 0
        self._jump_time = self._original_params['jump_time']
        self._tau = self._original_params['tau']
        self._var_jump = self._original_params['var_jump']
        self._lr_mean = self._original_params['lr_mean']
        self._lr_var = self._original_params['lr_var']


class ProteinJumpTrace(T.ProteinPlasticity):

    """ a Protein subclass 

    its dynamics are specified by a jump function 
    
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
        self.substrate_family = 'JumpTrace'

        # param
        self._jump_time = 0
        self._tau = 0
        self._lr_mean = 0
        self._tau_trace = 0

        # variables
        self._est_mean = 0
        self._drift = 0
        self._feedback = 0
        self._pre_weights = np.zeros(1)
        self._post_weights = np.zeros(1)
        self._counter = 0 
        self._trace_jump = 0.

        #
        self._compute_direction = lambda x: np.sign(x - 0.5)
        self._compute_direction = np.vectorize(self._compute_direction)


        # the externals is partioned as follow:
        # [0] : the flag for the jump
        # [1:] : the customized variance of the jump

        #
        self._protein_jump_initialization()
        self._update_dna()

        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}.{self.substrate_role}', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()

    def _protein_jump_initialization(self):

        """
        check if the DNA contains the right parameters keys and \
        initialize their values 

        Returns
        -------
        None 
        """

        parameters_keys = tuple(self.DNA['params'].keys())

        assert 'tau' in parameters_keys, "missing parameter 'tau'"
        assert 'tau_trace' in parameters_keys, "missing parameter 'tau_trace'"
        assert 'jump_time' in parameters_keys, "missing parameter 'jump_time'"
        assert 'var_jump' in parameters_keys, "missing parameter 'var_jump'"

        self._tau = self.DNA['params']['tau']
        self._tau_trace = self.DNA['params']['tau_trace']
        self._jump_time = self.DNA['params']['jump_time']
        self._var_jump = self.DNA['params']['var_jump']
        
        # optional parameters
        if "lr_mean" in parameters_keys:
            self._lr_mean = self.DNA['params']['lr_mean']
        else:
            self._lr_mean = 0.01
            self.DNA['params']['lr_mean'] = self._lr_mean

        # record the original parameters
        self._original_params['tau'] = self._tau 
        self._original_params['jump_time'] = self._jump_time 
        self._original_params['var_jump'] = self._var_jump
        self._original_params['lr_mean'] = self._lr_mean

        # initialize the variables
        self._pre_weights = self._weights.copy()
        self._post_weights = self._weights.copy()

    def _update_dna(self):
        
        """
        update the dna with the current parameters
        
        Returns
        -------
        None
        """
        
        self._update_protein_dna()
        self.DNA['params']['tau'] = self._tau
        self.DNA['params']['jump_time'] = self._jump_time
        self.DNA['params']['var_jump'] = self._var_jump

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

        # compute the reward credit trace and record it as an internal variable
        self._internals[0] = self._trace_jump * self._est_mean 

    def _random_jump(self):

        """
        generate a random jump in the parameter space, by adding a noise vector \
        to the current parameters

        Returns
        -------
        None
        """

        # copy weight matrix before the jump 
        self._pre_weights = self._weights.copy()

        # generate a random jump with:
        # direction : binomial with parameter est_mean
        # magnitude : normal with parameter var_jump from externals[1]
        self._weights += self._compute_direction(np.random.binomial(1, self._est_mean, size=self._weights.shape)) * \
            np.random.normal(0, self._externals[1], size=self._weights.shape)

        # copu weight matrix after the jump
        self._post_weights = self._weights.copy()

    def _step_drift(self):

        """
        compute and apply a drift vector to the weights 
        """

        # compute the drift vector 
        if self._feedback <= 0:
            self._weights += (self._pre_weights - self._weights) / self._tau
        else:
            self._weights = self._post_weights.copy()

    def step(self):
        
        """
        receive an input and the state is updated
        
        Returns
        -------
        None
        """
        
        # forward
        self.activity = self._weights @ self._ext_inputs
    
    def update(self):

        """
        the trainable parameters are updated
        
        Returns
        -------
        None
        """

        if 'w1' in self.trainable_names:

            # compute jump trace 
            try:
                self._trace_jump -= self._trace_jump / self._tau_trace
            except ZeroDivisionError:
                self._trace_jump = 0

            # udpate counter 
            self._counter += 1
            
            # jump if the counter is greater than the jump time and the flag is on
            if self._counter > self._jump_time and self._externals[0] > 0:

                self._random_jump()

                # record jump time
                self._counter = 0
                self._trace_jump += 1

            # drift
            self._step_drift()

            # reset feedback
            self._feedback *= 0

            # collect internals
            self._collect_internals()
    
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

        # record feedback
        self._feedback = ext_feedback

        # squash feedback from [-1, 1] to [0, 1]
        # estimate the mean of the feedback
        self._est_mean += self._lr_mean * ((ext_feedback + 1) / 2 - self._est_mean)

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
                self.trainable_params[i] = self._tau

            elif param == 'jump_time':
                self.trainable_params[i] = self._jump_time

            elif param == 'var_jump':
                self.trainable_params[i] = self._var_jump

            elif param == f'w{k+1}':
                self.trainable_params[i] = self._weights[0, k]
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
        self._reset_plasticity()

        self._drift *= 0
        self._feedback *= 0
        self._pre_weights = self._weights.copy()
        self._post_weights = self._weights.copy()

        self._counter = 0
        self._jump_time = self._original_params['jump_time']
        self._tau = self._original_params['tau']
        self._var_jump = self._original_params['var_jump']


### PROTEINS DICTIONARY ###

protein_dict = {'exp': lambda dna, verbose: ProteinExp(dna=dna, verbose=verbose),
                'linear': lambda dna, verbose: ProteinLinear(dna=dna, verbose=verbose),
                'expbeta': lambda dna, verbose: ProteinExpBeta(dna=dna, verbose=verbose),
                'cond': lambda dna, verbose: ProteinCond(dna=dna, verbose=verbose),
                'poly': lambda dna, verbose: ProteinPoly(dna=dna, verbose=verbose),
                'spike': lambda dna, verbose: ProteinSpike(dna=dna, verbose=verbose),
                'plasticity_root': lambda dna, verbose: ProteinPlasticity(dna=dna, verbose=verbose),
                'plasticity__stdp': lambda dna, verbose: ProteinPlasticity_stdp(dna=dna, verbose=verbose),
                'plasticity_reward': lambda dna, verbose: ProteinPlasticityReward(dna=dna, verbose=verbose),
                'plasticity_hebb': lambda dna, verbose: ProteinPlasticityHebb(dna=dna, verbose=verbose),
                'oja': lambda dna, verbose: ProteinPlasticityHebb(dna=dna, verbose=verbose),
                'jump': lambda dna, verbose: ProteinJump(dna=dna, verbose=verbose),
                'jump_step': lambda dna, verbose: ProteinJumpStep(dna=dna, verbose=verbose),
                'jump_trace': lambda dna, verbose: ProteinJumpTrace(dna=dna, verbose=verbose),
                }


if __name__ == "__main__":

    print("%proteins")
