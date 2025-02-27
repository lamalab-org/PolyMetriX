from typing import List
import numpy as np
import networkx as nx
import logging
from rdkit import Chem
from polymetrix.featurizers.base_featurizer import (
    BaseFeatureCalculator,
    PolymerPartFeaturizer,
)


class SideChainFeaturizer(PolymerPartFeaturizer):
    def featurize(self, polymer) -> np.ndarray:
        sidechain_mols = polymer.get_backbone_and_sidechain_molecules()[1]
        if not sidechain_mols:
            logging.info("No sidechains found in the molecule")
            return np.zeros(len(self.calculator.feature_labels()))
        features = [self.calculator.calculate(mol) for mol in sidechain_mols]
        return self.calculator.aggregate(features)

    def feature_labels(self) -> List[str]:
        if self.calculator:
            return [
                f"{label}_{self.__class__.__name__.lower()}_{agg}"
                for label in self.calculator.feature_base_labels()
                for agg in self.calculator.agg
            ]
        else:
            return [self.__class__.__name__.lower()]


class NumSideChainFeaturizer(PolymerPartFeaturizer):
    def featurize(self, polymer) -> np.ndarray:
        sidechain_mols = polymer.get_backbone_and_sidechain_molecules()[1]
        return np.array([len(sidechain_mols)])


class BackBoneFeaturizer(PolymerPartFeaturizer):
    def featurize(self, polymer) -> np.ndarray:
        backbone_mol = polymer.get_backbone_and_sidechain_molecules()[0][0]
        return self.calculator.calculate(backbone_mol)


class NumBackBoneFeaturizer(PolymerPartFeaturizer):
    def featurize(self, polymer) -> np.ndarray:
        backbone_mols = polymer.get_backbone_and_sidechain_molecules()[0]
        return np.array([len(backbone_mols)])


class FullPolymerFeaturizer(PolymerPartFeaturizer):
    def featurize(self, polymer) -> np.ndarray:
        mol = Chem.MolFromSmiles(polymer.psmiles)
        return self.calculator.calculate(mol)


class SidechainLengthToStarAttachmentDistanceRatioFeaturizer(BaseFeatureCalculator):
    """Computes aggregated ratios of sidechain lengths to the shortest backbone distance from the polymer's star node (*) to each sidechain's attachment point."""

    def _compute_min_backbone_length(self, sidechain, star_nodes, star_paths, graph):
        """Calculate the minimum backbone distance from any star node to the sidechain's attachment point."""
        min_backbone_length = float("inf")
        side_nodes = set(sidechain.nodes())
        for node in side_nodes:
            neighbors = set(graph.neighbors(node))
            backbone_neighbors = neighbors - side_nodes
            if backbone_neighbors:
                attachment_point = next(iter(backbone_neighbors))
                for star in star_nodes:
                    if attachment_point in star_paths[star]:
                        path_length = star_paths[star][attachment_point] + 1
                        min_backbone_length = min(min_backbone_length, path_length)
        return min_backbone_length

    def featurize(self, polymer) -> np.ndarray:
        graph = polymer.graph
        star_nodes = [
            node for node, data in graph.nodes(data=True) if data["element"] == "*"
        ]
        backbone_graphs, sidechain_graphs = polymer.get_backbone_and_sidechain_graphs()

        if not sidechain_graphs or not backbone_graphs:
            return np.zeros(len(self.agg))

        sidechain_lengths = [len(sc.nodes()) for sc in sidechain_graphs]
        star_paths = {
            star: nx.single_source_shortest_path_length(graph, star)
            for star in star_nodes
        }

        backbone_lengths = [
            self._compute_min_backbone_length(sidechain, star_nodes, star_paths, graph)
            for sidechain in sidechain_graphs
        ]

        ratios = [
            s_length / b_length
            for s_length, b_length in zip(sidechain_lengths, backbone_lengths)
            if b_length > 0
        ]
        if not ratios:
            return np.zeros(len(self.agg))

        agg_ratios = self.aggregate(ratios)
        return np.array(agg_ratios)

    def feature_base_labels(self) -> List[str]:
        return ["sidechainlength_to_star_attachment_distance_ratio"]


class StarToSidechainMinDistanceFeaturizer(BaseFeatureCalculator):
    """Computes aggregated minimum backbone distances from star nodes (*) to sidechains in a polymer."""

    def featurize(self, polymer) -> np.ndarray:
        graph = polymer.graph
        star_nodes = [
            node for node, data in graph.nodes(data=True) if data["element"] == "*"
        ]
        sidechain_graphs = polymer.get_backbone_and_sidechain_graphs()[1]

        distances = []
        for sidechain in sidechain_graphs:
            valid_dists = [
                nx.shortest_path_length(graph, star, node) - 1
                for star in star_nodes
                for node in sidechain.nodes()
                if nx.has_path(graph, star, node)
            ]
            if valid_dists:
                distances.append(min(valid_dists))

        if not distances:
            return np.zeros(len(self.agg))

        return self.aggregate(distances)

    def feature_base_labels(self) -> List[str]:
        return ["star_to_sidechain_min_distance"]


class SidechainDiversityFeaturizer(BaseFeatureCalculator):
    """Computes the number of structurally diverse sidechains in a polymer based on graph isomorphism."""

    def featurize(self, polymer) -> np.ndarray:
        sidechain_graphs = polymer.get_backbone_and_sidechain_graphs()[1]
        unique_hashes = set()
        for scg in sidechain_graphs:
            graph_hash = nx.weisfeiler_lehman_graph_hash(scg)
            unique_hashes.add(graph_hash)
        return np.array([len(unique_hashes)])

    def feature_labels(self) -> List[str]:
        return ["num_diverse_sidechains"]
