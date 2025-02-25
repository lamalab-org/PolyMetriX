site_name: PolyMetriX
site_url: https://lamalab-org.github.io/polymetrix/
site_description: >-
  PolyMetriX is a comprehensive Python library that powers the entire machine learning workflow for polymer informatics. From data preparation to feature engineering, it provides a unified framework for developing structure-property relationships in polymer science.

# Repository
repo_name: lamalab-org/PolyMetriX
repo_url: https://github.com/lamalab-org/PolyMetriX.git

copyright: LAMAlab

docs_dir: docs
site_dir: site

theme:
  name: material
  custom_dir: docs/figures
  logo: static/overview.pdf
  favicon: static/overview.pdf
  features:
    - content.tabs.link
    - content.code.annotation
    - content.code.annotate
    - content.code.copy
    - content.code.copy_button
    - navigation.tabs
    - navigation.sections
    - navigation.top
    - toc.integrate
    - search.suggest
    - search.highlight
    - search.share
    - navigation.tabs.sticky

  palette:
    - scheme: default
      toggle:
        icon: material/toggle-switch-off-outline
        name: Switch to dark mode
      primary: custom
      accent: purple

    - scheme: slate
      toggle:
        icon: material/toggle-switch
        name: Switch to light mode
      primary: custom
      accent: lime

  hide:
    - navigation
    - toc

  font:
    text: Roboto #Helvetica
    code: Monaco #Roboto Mono

plugins:
  - search
  - autorefs
  # - social
  - mkdocstrings:
      default_handler: python
      handlers:
        python:
          paths: [src]
          options:
            show_symbol_type_toc: true
            allow_inspection: true
            show_submodules: true

markdown_extensions:
  - meta
  - tables
  - pymdownx.highlight
  - pymdownx.superfences
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - admonition
  - pymdownx.arithmatex:
      generic: true
  - footnotes
  - pymdownx.details
  - pymdownx.superfences
  - pymdownx.mark
  - attr_list
  - toc:
      permalink: '🔗'
  - pymdownx.emoji:
      emoji_index: !!python/name:material.extensions.emoji.twemoji
      emoji_generator: !!python/name:material.extensions.emoji.to_svg

extra_javascript:
  - https://unpkg.com/tablesort@5.3.0/dist/tablesort.min.js
  - javascripts/tablesort.js
  - https://polyfill.io/v3/polyfill.min.js?features=es6
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js

extra:
  social:
    - icon: fontawesome/brands/github-alt
      link: https://github.com/lamalab-org/
    - icon: fontawesome/brands/twitter
      link: https://twitter.com/jablonkagroup
#     - icon: fontawesome/brands/linkedin
#       link: https://www.linkedin.com/in/willettjames/

extra_css:
  - stylesheets/extra.css

nav:
  - Home: index.md
  - How-To Guides: getting_started.md
  - Installation: installation.md
  - Developer Notes: development.md
  - Contribute Questions: contribute_questions.md
  - API reference: api.md
  - Leaderboard: https://huggingface.co/spaces/jablonkagroup/ChemBench-Leaderboard





















# PolyMetriX

PolyMetriX is a Python-based tool designed to calculate various molecular descriptors for polymers generated from PSMILES strings.
## Descriptors Calculated

The following descriptors are calculated:

- **nconf20_2**: Number of conformers within 20 kcal/mol of the lowest energy conformer
- **rotatable_bonds**: Number of rotatable bonds in the molecule
- **num_rings**: Total number of rings in the molecule
- **num_aromatic_rings**: Number of aromatic rings in the molecule
- **num_non_aromatic_rings**: Number of non-aromatic rings in the molecule
- **hbond_acceptors**: Number of hydrogen bond acceptors
- **hbond_donors**: Number of hydrogen bond donors
- **n_sc**: Number of side chains
- **len_sc**: Lengths of the side chains
- **n_bb**: Number of backbones
- **len_bb**: Lengths of the backbones


## Local Installation

We recommend that you create a virtual conda environment on your computer in which you install the dependencies for this package. To do so head over to [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and follow the installation instructions there.


<!-- ### Install latest release

```bash
pip install mattext
``` -->

### Install development version

Clone this repository (you need `git` for this, if you get a `missing command` error for `git` you can install it with `sudo apt-get install git`)

```bash
git clone https://github.com/lamalab-org/PolyMetriX.git
cd polymetrix
```

```bash
pip install -e .
```
