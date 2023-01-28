import json
import warnings
#from .cells import *
#from .networks import *
from neuromix.brain import proteins as P
from neuromix.brain import cells as C
from neuromix.brain import networks as N
from neuromix.brain import templates as T


#### SUBSTRATE GENERATOR ####

# list of available substrates
SUBSTRATE_LIST = P.PROTEIN_LIST + C.CELL_LIST + N.NETWORK_LIST

# load libraries
with open(r"/Users/daniekru/Research/codebase/src/neuromix/neuromix/samples/proteins.json", 'r') as f:
    proteins_library = json.loads(f.read())
    #print("+Proteins library loaded")

with open(r"/Users/daniekru/Research/codebase/src/neuromix/neuromix/samples/cells.json", 'r') as f:
    cells_library = json.loads(f.read())
    #print("+Cells library loaded")


def print_protein_library(show_names=False, show_dna=False, family=None, id_=None):

    """
    print the protein library

    Parameters
    ----------
    show_names : bool
        if True, print the names of the proteins
    show_dna : bool
        if True, print the dna of the proteins
    family : str
        if provided, print the dna of the proteins of the given family
    id_ : int
        if provided, print the dna of the protein with the given id

    Returns
    -------
    None
    """

    # check inputs
    if not show_names and not show_dna and family is None:
        warnings.warn("No input provided, nothing to print")
        return

    # print the names of the proteins
    if show_names:
        print("Recorded protein families:\n")
        for family, ids in proteins_library.items():
            print(f"\t{family}: {list(ids.keys())}")
        print()

        # exit
        if not show_dna:
            return

    # print the dna of the proteins
    if show_dna:
        for family in proteins_library:
            print(f"{family=}")
            for id_ in proteins_library[family]:
                print(f"\n\t### id: {id_} ###")
                T.print_dict(gene=proteins_library[family][id_])
            print('\n', '- ' * 15, '\n')
        return
            
    # print the dna of the proteins of the given family
    if family is not None:
        assert family in proteins_library, f"{family} is not a valid protein family"
        if id_ is not None:
            assert id_ in proteins_library[family], f"{id_} is not a valid protein id"
            print(f"{family=}")
            print(f"\n\t### id: {id_} ###")
            T.print_dict(gene=proteins_library[family][id_])
        else:
            for id_ in proteins_library[family]:
                print(f"\n\t### id: {id_} ###")
                T.print_dict(gene=proteins_library[family][id_])


def print_cell_library(show_names=False, show_dna=False, family=None, id_=None):

    """
    print the cell library

    Parameters
    ----------
    show_names : bool
        if True, print the names of the cells
    show_dna : bool
        if True, print the dna of the cells
    family : str
        if provided, print the dna of the cells of the given family
    id_ : int
        if provided, print the dna of the cell with the given id

    Returns
    -------
    None
    """

    # check inputs
    if not show_names and not show_dna:
        warnings.warn("No input provided, nothing to print")
        return

    # print the names of the cells
    if show_names:
        print("Recorded cell families:\n")
        for family, ids in cells_library.items():
            print(f"\t{family}: {list(ids.keys())}")
        print()

        # exit
        if not show_dna:
            return
   
    # print the dna of the cells
    if show_dna:
        for family in cells_library:
            print(f"{family=}")
            for id_ in cells_library[family]:
                print(f"\n\t### id: {id_} ###")
                T.print_dict(gene=cells_library[family][id_])
            print('\n', '- ' * 15, '\n')
        return

    # print the dna of the cells of the given family
    if family is not None:
        assert family in cells_library, f"{family} is not a valid cell family"
        if id_ is not None:
            assert id_ in cells_library[family], f"{id_} is not a valid cell id"
            print(f"{family=}")
            print(f"\n\t### id: {id_} ###")
            T.print_dict(gene=cells_library[family][id_])
        else:
            for id_ in cells_library[family]:
                print(f"\n\t### id: {id_} ###")
                T.print_dict(gene=cells_library[family][id_])


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
    assert dna[0] in SUBSTRATE_LIST, f"{dna[0]} is an invalid substrate.Class"
    assert 'family' in dna[1], "DNA family not specified"

    substrate_name = dna[0]
    family_name = dna[1]['family']
    id_ = dna[1]['id'] if 'id' in dna[1].keys() else None

    # if the substrate is a root Substrate class, then build
    if family_name == 'root' and id_ == '-1':
        recorded_specs = T.root_library[substrate_name]

        # add each key of the recorded specs to the dna
        for key in recorded_specs:
            if key not in dna[1]:
                dna[1][key] = recorded_specs[key]
        
    # a complete DNA has been provided <-- no 'id'
    elif 'id' not in dna[1].keys():
        return dna 

    # a saved substrate DNA has been queried 
    else:
        id_name = dna[1]['id']
        try:
            if substrate_name in P.PROTEIN_LIST:
                recorded_specs = proteins_library[family_name][id_name]

            elif substrate_name in C.CELL_LIST:
                recorded_specs = cells_library[family_name][id_name]

                # merge cell specific keys 
                dna[1]['components'] = recorded_specs['components']
                dna[1]['connections'] = recorded_specs['connections']

            elif substrate_name in N.NETWORK_LIST:
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
    id_ = dna[1]['id'] if 'id' in dna[1].keys() else None

    # if the substrate is a root Substrate class, then build from root_dict 
    if family_name == 'root' and id_ == '-1':

        # build the root substrate
        assert substrate_name in T.root_dict.keys(), f"<{substrate_name}> is not a valid root substrate [{T.root_dict.keys()}]"
        substrate = T.root_dict[substrate_name](dna=dna[1], verbose=verbose)
        
    # if the substrate is a substrate subclass, then build from the specific dict
    else:
        if substrate_name in P.PROTEIN_LIST:

            # build the protein
            assert family_name in P.protein_dict.keys(), f"<{family_name}> is not a valid protein family [{P.protein_dict.keys()}"
            substrate = P.protein_dict[family_name](dna[1], verbose=verbose)

        elif substrate_name in C.CELL_LIST:
            
            # collect all the proteins first 
            proteins = []
            for protein_dna in dna[1]['components']:
                proteins.append(generate_substrate(dna=protein_dna))
            
            # build the cell
            assert family_name in C.cell_dict.keys(), f"<{family_name}> is not a valid cell family [{C.cell_dict.keys()}]"
            substrate = C.cell_dict[family_name](dna=dna[1], built_components=proteins, verbose=verbose)

        else:
            raise NotImplementedError(f'substrate <{substrate_name}> not supported for now')

    return substrate


if __name__ == '__main__':
    
    print('%generator')
