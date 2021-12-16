# TimeSeriesCausality.jl
This is a Julia implementation of the [*Phase Slope Index*]((https://link.aps.org/doi/10.1103/PhysRevLett.100.234101)) and [*Granger Causality*](https://doi.org/10.2307/1912791) methods. Please refer to [http://doc.ml.tu-berlin.de/causality](http://doc.ml.tu-berlin.de/causality/) for more information.

<!-- ## Outline
```@contents
Pages = ["index.md", "generated/examples.md"]
``` -->

## Functions
Phase Slope Index (PSI) estimation `psi_est`:

```@docs
psi_est
```

Granger Causality estimation `granger_est`:

```@docs
granger_est
```

## Citation
Please cite the following paper if you use the PSI code in published work:
> Nolte, G., Ziehe, A., Nikulin, V., Schlögl, A., Krämer, N., Brismar, T., & Müller, K.R. (2008), *[Robustly Estimating the Flow Direction of Information in Complex Physical Systems](https://link.aps.org/doi/10.1103/PhysRevLett.100.234101)*, Phys. Rev. Lett., 100, 234101. 

## Acknowledgement
This work was funded by the German Federal Ministry of Education and Research ([BMBF](https://www.bmbf.de/)) in the project ALICE III under grant ref. 01IS18049B.
