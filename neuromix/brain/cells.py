import numpy as np
from neuromix.brain import templates as T


CELL_LIST = ['Cell', 'CellPlasticity']


### CELLS ###

class CellPlasticityJump(T.CellPlasticity):
    
    """ a Cell Plasticity subclass
    
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
        super().__init__(dna=dna, built_components=built_components)
        self.substrate_family = 'CellPlasticityJump'
        
        if verbose:
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}.{self.substrate_role}', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()
    
            print(f'\nnb_components: {self.nb_components}\nnb_inputs: '
                  f'{self.nb_inputs}\nnb_outputs: {self.nb_outputs}'
                  f'\nnb_trainable: {self.nb_trainable}')

    def update(self):

        """
        the trainable parameters are updated [to edit]
        
        Returns
        -------
        None
        """

        # the plasticity proteins collect internals 
        for k in self.idx_plastic:
            self.components[k].add_feedback(ext_feedback=self._feedback)
            self.components[k].update()

        # reset
        self._feedback *= 0


class CellJumpTrace(T.CellPlasticity):
    
    """ a Cell Plasticity subclass
    
    it implements a network of Protein Jump Trace components
    
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
        super().__init__(dna=dna, built_components=built_components)
        self.substrate_family = 'CellJumpTrace'

        # parameters
        self._lr_est = 0.02
        self._nb_jumps = 0
        self._compute_jump_var = lambda x: max((1 - 2.*x, 0.04))
        self._softmax = T.activation_functions['softmax'][0]

        # variables
        self._jump_var = 1.
        self._est_mean = 0.5
        self._est_var = 1.

        self._short_mean = 0.
        self._short_var = .0

        # saturation
        # idx_intf -> [-1]: saturation protein
        self._saturation_lv = 0.

        # initialize
        self._cell_jump_initialization()
        self.reset()
        
        if verbose:
            print(f"\n@{self.substrate_class}.{self.substrate_family}.", \
                  f"{self.substrate_id}.{self.substrate_role}", end="")
            if self.trainable:
                print(' [trainable]')
            else:
                print()
    
            print(f'\nnb_components: {self.nb_components}\nnb_inputs: '
                  f'{self.nb_inputs}\nnb_outputs: {self.nb_outputs}'
                  f'\nnb_trainable: {self.nb_trainable}')

    def _cell_jump_initialization(self):

        """
        initialize cell jump
        
        Returns
        -------
        None
        """

        # check that each plastic component belongs to the family JumpTrace
        for idx in self.idx_plastic:
            assert self.components[idx].substrate_family == 'JumpTrace', \
                f"component {idx} must be of family JumpTrace, given {self.components[idx].substrate_family}"
            
        if 'nb_jumps' in self.DNA['attrb']:
            self._nb_jumps = self.DNA['attrb']['nb_jumps']
            assert 0 <= self._nb_jumps <= len(self.idx_plastic), \
                    f"nb_jumps must be > 0 and <= {len(self.idx_plastic)}"
        else:
            self._nb_jumps = 1
            self.DNA['attrb']['nb_jumps'] = self._nb_jumps

        # optional parameters
        if "lr_est" in self.DNA['params'].keys():
            self._lr_est = self.DNA['params']['lr_est']
        else:
            self._lr_est = 0.01
            self.DNA['params']['lr_est'] = self._lr_est

        # define comp_internals | 2 columns: 
        # 1st: the jump flag
        # 2nd: the jump variance
        self._comp_internals = np.zeros((len(self.idx_plastic), 2))

    def _collect_internals(self):
        
        """
        collect the values of the internal variables of interests
        
        Returns
        -------
        None
        """
   
        # compute the softmax of the credit traces
        credit_traces = np.concatenate([self.components[idx].get_internals() \
                                           for idx in self.idx_plastic])
        credit_traces_soft = self._softmax(credit_traces)

        # define the jump variance from the estimated variance
        self._jump_var = (1 - self._saturation_lv) * self._compute_jump_var(self._est_var)

        # define the selection of the jump probability distirbution among the
        # plasticity proteins #
        # get weights sum from each component | although a more complex selection
        # could be implemented
        weights = [self.components[idx].get_trainable_params().sum() for idx in self.idx_plastic]

        # complement of the softmax of the weights, to be used as a probability distribution
        jump_probabilities = self._softmax(1 - self._softmax(weights))

        # partition of the jump variance among the plasticity proteins
        # 2nd column of comp_internals
        self._comp_internals[:, 1] = self._jump_var * credit_traces_soft

        # draw the plastic protein to jump | 1st column of comp_internals
        jump_idx = np.random.choice(self.idx_plastic, size=self._nb_jumps)
        self._comp_internals[jump_idx, 0] = 1

    def _compute_saturation(self):

        """
        sample the temperature of the cell through the CondSat component

        Returns
        -------
        None
        """

        # run the CondSat component
        self.components[self.idx_intf[-1]].collect_input(inputs=self._short_var)
        self.components[self.idx_intf[-1]].step()

        # get the output
        self._saturation_lv = self.components[self.idx_intf[-1]].get_output()

    def update(self):

        """
        the trainable parameters are updated 
        
        Returns
        -------
        None
        """

        # 
        self._compute_saturation()

        # the plasticity proteins their collect externals and update
        for k in self.idx_plastic:
            self.components[k].collect_externals(externals=self._comp_internals[k])
            self.components[k].update()

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
        self._est_mean += self._lr_est * (squashed_feedback - self._est_mean)
        self._est_var += self._lr_est * (abs(squashed_feedback - self._est_mean) - self._est_var)
        self._short_mean += 0.5 * (self.output.item() - self._short_mean)
        self._short_var += 0.2 * (abs(self.output.item() - self._short_mean) - self._short_var)

        # deliver the global feedback to the plasticity proteins
        for k in self.idx_plastic:
            self.components[k].add_feedback(ext_feedback=ext_feedback)

    def reset(self):

        """
        reset the cell
        
        Returns
        -------
        None
        """

        self.reset_structure()
        self._reset_plasticity()

        # reset the internal variables
                # variables
        self._jump_var = 1
        self._est_mean = 0.5
        self._est_var = 0
        self._short_mean = 0.5
        self._short_var = 0
        self._saturation_lv = 0


### dict 

cell_dict = {'root': lambda dna, built_components, verbose: \
        T.Cell(dna=dna, built_components=built_components, verbose=verbose),

             'plasticity': lambda dna, built_components, verbose: \
        T.CellPlasticity(dna=dna, built_components=built_components, verbose=verbose),

             'jump': lambda dna, built_components, verbose: \
        CellPlasticityJump(dna=dna, built_components=built_components, verbose=verbose),

             'jump_trace': lambda dna, built_components, verbose: \
        CellJumpTrace(dna=dna, built_components=built_components, verbose=verbose),
}



if __name__ == '__main__':
    
    print('%cells')
