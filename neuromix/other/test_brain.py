import pytest
from brain import *

def test_spike():

    dna = ('Protein', {'family': 'spike',
                       'id': '0'})
    sub = generate_substrate(dna=dnar, verbose=True)

    assert isinstance(sub.get_dna(), dict)
