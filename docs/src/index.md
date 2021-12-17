# PhaseSlopeIndex.jl
This is a Julia implementation of the [*Phase Slope Index* method]((http://link.aps.org/abstract/PRL/v100/e234101)). Please refer to http://doc.ml.tu-berlin.de/causality/ for more information.

## Outline
```@contents
Pages = ["index.md", "generated/examples.md"]
```

## Functions
The only exported function is `data2psi`:

```@docs
data2psi
```

## Citation
Please cite the following paper if you use this code in published work:
> Nolte, G., Ziehe, A., Nikulin, V., Schlögl, A., Krämer, N., Brismar, T., & Müller, K.R. (2008), *[Robustly Estimating the Flow Direction of Information in Complex Physical Systems](http://link.aps.org/abstract/PRL/v100/e234101)*, Phys. Rev. Lett., 100, 234101. 

## Acknowledgement
This work was funded by the German Federal Ministry of Education and Research ([BMBF](https://www.bmbf.de/)) in the project ALICE III under grant ref. 01IS18049B.
