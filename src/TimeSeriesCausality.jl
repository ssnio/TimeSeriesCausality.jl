module TimeSeriesCausality

# Imports
using Einsum
using FFTW: fft
using Statistics: mean, std, cov as covariance
using LinearAlgebra: det, I as identity

# Includes
include("utils.jl")
include("phase_slope_index.jl")
include("granger_estimation.jl")

# Exports
export psi_est
export granger_est, granger_aic, granger_bic

end # module
