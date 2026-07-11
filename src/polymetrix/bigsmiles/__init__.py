"""BigSMILES support for PolyMetriX.

Public API:
    BigSmilesPolymer  -- BigSMILES-backed polymer with backbone/sidechain
                         classification and JSON storage, compatible with the
                         existing PolyMetriX featurizer and splitter stack.
    RepeatUnit        -- one classified stochastic fragment (repeat unit).
"""

from polymetrix.bigsmiles.bigsmiles_polymer import BigSmilesPolymer, RepeatUnit

__all__ = ["BigSmilesPolymer", "RepeatUnit"]
