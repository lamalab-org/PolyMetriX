from polymetrix.featurizer import  NumAtoms, SideChainFeaturizer, NumSideChainFeaturizer
from polymetrix.polymer import Polymer

def test_integration_1():
    polymer = Polymer.from_psmiles('*c1ccc(Oc2ccc(-c3nc4cc(Oc5ccc(NC(=O)c6cc(C(=O)Nc7ccc(Oc8ccc9nc(-c%10ccccc%10)c(*)nc9c8)cc7)cc(N7C(=O)c8ccccc8C7=O)c6)cc5)ccc4nc3-c3ccccc3)cc2)cc1')

    featurizer = NumSideChainFeaturizer()
    assert featurizer.featurize(polymer)[0] == 3

    featurizer = SideChainFeaturizer(NumAtoms(agg=['sum']))
    assert featurizer.featurize(polymer)[0] == 23
