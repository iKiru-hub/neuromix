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
            print(f'\n@{self.substrate_class}.{self.substrate_family}.{self.substrate_id}', end='')
            if self.trainable:
                print(' [trainable]')
            else:
                print()
    
            print(f'\nnb_components: {self.nb_components}\nnb_inputs: '
                  f'{self.nb_inputs}\nnb_outputs: {self.nb_outputs}'
                  f'\nnb_trainable: {self.nb_trainable}')


    def collect_input(self, inputs: np.ndarray):

   
        # external inputs
        self.activity[:self.nb_inputs] = inputs.reshape(-1)

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




### dict 

cell_dict = {'root': lambda dna, built_components, verbose: \
        T.Cell(dna=dna, built_components=built_components, verbose=verbose),

             'plasticity': lambda dna, built_components, verbose: \
        T.CellPlasticity(dna=dna, built_components=built_components, verbose=verbose),

             'jump': lambda dna, built_components, verbose: \
        CellPlasticityJump(dna=dna, built_components=built_components, verbose=verbose)
}


if __name__ == '__main__':
    
    print('%cells')
