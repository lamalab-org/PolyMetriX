# PolyMetriX

PolyMetriX is a comprehensive Python library that powers the entire machine learning workflow for polymer informatics. From data preparation to feature engineering, it provides a unified framework for developing structure-property relationships in polymer science.
![PolyMetriX Overview](figures/overview_revised.png)



## Why PolyMetriX?
The polymer informatics community often works with fragmented tools and custom implementations, making it challenging to develop reproducible and standardized workflows. PolyMetriX addresses this by providing an integrated ecosystem:

- Data:
  - Standardized dataset objects for polymer data management
  - Curated datasets including glass transition temperature (Tg) database
  - Custom data splitting strategies optimized for polymer structures
- Advanced Featurization
  - Hierarchical feature extraction at full polymer, backbone, and side-chain levels
  - RDKit integration for robust molecular descriptor computation
  - Specialized polymer-specific descriptors
- Built-in data splitting strategies for robust model validation
- Consistent API inspired by established cheminformatics tools

Whether you're developing ML models for polymers, PolyMetriX streamlines your entire workflow. The package is open-source and designed to support reproducible research in polymer science and engineering, aiming to become the foundation for digital polymer chemistry.