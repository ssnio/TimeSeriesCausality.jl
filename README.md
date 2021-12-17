# PhaseSlopeIndex.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ssnio.github.io/PhaseSlopeIndex.jl/stable)
[![Master](https://img.shields.io/badge/docs-master-blue.svg)](https://ssnio.github.io/PhaseSlopeIndex.jl/dev)
[![Build Status](https://github.com/ssnio/PhaseSlopeIndex.jl/workflows/CI/badge.svg)](https://ssnio.github.io/PhaseSlopeIndex.jl/actions)
[![Coverage](https://codecov.io/gh/ssnio/PhaseSlopeIndex.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ssnio/PhaseSlopeIndex.jl)

This is a Julia implementation of the [*Phase Slope Index* method](http://link.aps.org/abstract/PRL/v100/e234101). Please refer to http://doc.ml.tu-berlin.de/causality/ for more information.

### Examples
The only exported function is [`data2psi`](https://ssnio.github.io/PhaseSlopeIndex.jl/dev/#Functions). For detailed examples, refer to the [docs](https://ssnio.github.io/PhaseSlopeIndex.jl/dev/generated/examples/) or the following notebook on the platform of your choice:


[![github](https://img.shields.io/badge/render-GitHub%20notebook-blue)](https://github.com/ssnio/PhaseSlopeIndex.jl/blob/gh-pages/dev/generated/examples.ipynb)
<!-- [![binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ssnio/PhaseSlopeIndex.jl/gh-pages?filepath=dev/generated/examples.ipynb) -->
<!-- [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/ssnio/PhaseSlopeIndex.jl/blob/gh-pages/dev/generated/examples.ipynb) -->


### Citation
Please [cite the following paper](https://github.com/ssnio/PhaseSlopeIndex.jl/blob/master/citation.bib) if you use this code in published work:
> Nolte, G., Ziehe, A., Nikulin, V., Schlögl, A., Krämer, N., Brismar, T., & Müller, K.R. (2008), *[Robustly Estimating the Flow Direction of Information in Complex Physical Systems](http://link.aps.org/abstract/PRL/v100/e234101)*, Phys. Rev. Lett., 100, 234101. 

### Acknowledgement
This work was funded by the German Federal Ministry of Education and Research ([BMBF](https://www.bmbf.de/)) in the project ALICE III under grant ref. 01IS18049B.
