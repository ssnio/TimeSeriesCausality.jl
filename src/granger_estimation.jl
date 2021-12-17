"""
	prep_data_granger(data, seglen, verbose)

Checks and prepares data shape.

### Arguments

  - `data::AbstractArray`: Nx2 array for N data points in 2 channels.
  - `seglen::Integer`: segment length.
  - `verbose::Bool`: if `true`, warnings and info logs would be echoed.

### Returns

  - `data::AbstractArray`: NxM array for N data points in M channels.
"""
function prep_data_granger(data::AbstractArray, seglen::Integer, verbose::Bool)
	
    if ndims(data) != 2  # data dimension
        data = squeeze(data)
        ndims(data) != 2 && throw(DimensionMismatch("Data must be a 2D-array!"))
        verbose && @info "Data is squeezed to a 2D-array)"
    end
    if size(data, 1) < size(data, 2)  # should be NxM array for N data points in M channels
        verbose && @info "Data is transposed to (#samples, #channels)"
        data = reshape(data, size(data, 2), size(data, 1))
    end
	if size(data, 2) != 2  # should be exactly 2 channels
		throw("Only 2 channels are supported!")
	end
    if size(data, 1) < seglen  # seglen must be smaller than number of samples
        throw(DimensionMismatch("seglen must be smaller than number of samples!"))
    end
    return data
end

"""
	granger_est(data, seglen; order, method, verbose)

Granger's method of causal relation approximation

### Arguments

  - `data::AbstractArray`: Nx2 array for N data points in 2 channels.
  - `seglen::Integer`: segment length.

*optional arguments*

  - `method::String`: standard deviation estimation method (default is `"jackknife"`)
  - `order::Int`: Model order. Assumed time delay order of interest (default is `n_samples / 2`)
  - `verbose::Bool`: if `true`, warnings and info logs would be echoed.

### Returns

  - `Grind::Float64`: Granger causality index
  - `Grind_std::Float64`: estimated standard deviation of error

### Internal variables:

  - Covs: Concatenated covariance mats of different orders
  - Acoef: Design (A) matrix coefficients
  - Perr: Prediction Error
  - Σ: Measures the accuracy of the autoregressive prediction

### References:
  - C. W. J. Granger (1969). Investigating causal relations by econometric models
    and cross-spectral methods, Econometrica, 37, 424-438.
  - M. Ding et al. (2006). Granger Causality: Basic Theory and Application to Neuroscience
    available online: https://arxiv.org/abs/q-bio/0608035

"""
function granger_est(data::Array{Float64, 2},
					 seglen::Int;
					 order::Int=0,
					 method::String="none",
					 verbose::Bool=true)

	data = prep_data_granger(data, seglen, verbose)
	
	if order == 0
		order = int(seglen / 2) - 1
	end

	Covs12 = est_sig_covs(data, order)
	Covs1 = est_sig_covs(review(data, 1), order)
	Covs2 = est_sig_covs(review(data, 2), order)
	
	Acoef12, Perr12 = mvar_est(Covs12)
	Acoef1, _ = mvar_est(Covs1)
	Acoef2, _ = mvar_est(Covs2)
	
	Σ12 = Perr12[:, end-1:end]
	Σ12 = est_noise_covs(data, Acoef12, seglen)
	Σ1 = est_noise_covs(review(data, 1), Acoef1, seglen)
	Γ1 = est_noise_covs(review(data, 2), Acoef2, seglen)
	Σ2 = Σ12[1, 1]
	Γ2 = Σ12[2, 2]
	
	X_to_Y = log(Γ1 / Γ2)
	Y_to_X = log(Σ1 / Σ2)
	
	Grind = (X_to_Y - Y_to_X)[1]

	if lowercase(method) == "jackknife"
		Grind_std = granger_jackknife(data, seglen, order, Covs12, Covs1, Covs2)
		return Grind, Grind_std
	else
		return Grind
	end
	
end

"""
	est_sig_covs(data, order)

Concatenated covariances of signal to the given order

### Arguments

  - `data::AbstractArray`: Nx2 array for N data points in 2 channels.
  - `order::Int`: Model order (default is nsamples/2 )

### Returns

  - `Covs::Array{Float64, 2}`: Concatenated covariances of 0 to the given order

"""
function est_sig_covs(data::AbstractArray{Float64, 2}, order::Int)::Array{Float64, 2}

	nsamples, nchan = size(data) # number of rows (samples) and columns (channels)
	
	Covs = Array{Float64}(undef, nchan, nchan*(order+1))
	
	Covs[:, 1:nchan] = covariance(data)
	for k = 1:order
		@views Covs[:, (1:nchan).+k*nchan] = covariance(data[k+1:nsamples, :], data[1:nsamples-k, :])
	end
	return Covs
end

"""
	mvar_est(Covs)

MultiVariate AutoRegressive (MVAR) model parameter estimation using
"Levinson-Wiggens-Robinson (LWR) algorithm using unbiased correlation function"

### Arguments

  - `Covs::Array{Float64, 2}`: Concatenated covariances of 0 to the given order

### Returns

  - `ARF::Array{Float64, 2}`: Design (A) matrix coefficients
  - `PE::Array{Float64}` : model Prediction Error

### References:
  - Alois Schlögl et al. <a.schloegl@ieee.org>, part of TSA-toolbox and FieldTrip-toolbox:
    https://github.com/fieldtrip/fieldtrip/blob/master/external/biosig/mvar.m

"""
function mvar_est(Covs::Array{Float64, 2})

	n_i, m_j = size(Covs)  # number of rows and columns
	order = int(m_j / n_i) - 1

	ARF = Array{Float64}(undef, n_i, n_i*order)
	ARB = Array{Float64}(undef, n_i, n_i*order)
	RCF = Array{Float64}(undef, n_i, n_i*order)
	RCB = Array{Float64}(undef, n_i, n_i*order)
	PE = Array{Float64}(undef, n_i, n_i*(order+1))
	
    PE[:, 1:n_i] = Covs[:, 1:n_i]
	PEF = Covs[:, 1:n_i]  # it's a different normalization in BioSig
	PEB = Covs[:, 1:n_i]  # it's a different normalization in BioSig
	
	for k in 1:order
		S1 = k*n_i .+ (1:n_i)  # slice
		S2 = k*n_i .+ (1-n_i:0)  # slice
		
		D = Covs[:, S1]
		for L in 1:k-1
			S3 = L*n_i .+ (1-n_i:0)  # slice
			D -= ARF[:, S3] * Covs[:, (k-L)*n_i .+ (1:n_i)]
		end
		
		ARF[:, S2] = D / PEB
        ARB[:, S2] = D' / PEF
		for L in 1:k-1
			S3 = L*n_i .+ (1-n_i:0)  # slice
			S4 = (k-L)*n_i .+ (1-n_i:0)  # slice
			ARB[:, S4], ARF[:, S3] = (ARB[:, S4] - ARB[:, S2] * ARF[:, S3],
									  ARF[:, S3] - ARF[:, S2] * ARB[:, S4])
		end
		
		RCF[:, S2] = ARF[:, S2]
        RCB[:, S2] = ARB[:, S2]
        PEF = (identity(n_i) - ARF[:, S2] * ARB[:, S2]) * PEF
        PEB = (identity(n_i) - ARB[:, S2] * ARF[:, S2]) * PEB
        PE[:, S1] = PEF
	end
	
	return ARF, PE
end

"""
	est_noise_covs(data, A, seglen)

Estimation of covariance of noise

### Arguments

  - `data::AbstractArray`: NxM array for N data points in M channels.
  - `A::Array{Float64, 2}`: Design (A) matrix coefficients
  - `seglen::Integer`: segment length.

### Returns
  
  - `C::Array{Float64}` : covariance matrix of noise

"""
function est_noise_covs(data::AbstractArray{Float64, 2},
						A::AbstractArray{Float64, 2},
						seglen::Int)

	nsamples, nchan = size(data)  # number of rows and columns
	order = int(size(A, 2) / nchan)  # number of columns = order * number of channels
	nsegments = int(nsamples / seglen)  # number of segments
	
	A = reshape(A, nchan, nchan, :)
    C = zeros(nchan, nchan)

	@inbounds for e in 1:nsegments
		S1 = (e-1)*seglen+1:e*seglen
		S2 = order+1:seglen
		@views X = data[S1, :][S2, :]'
		for i in 1:order
			@views X -= A[:, :, i] * data[S1, :][S2 .- i, :]'
		end
		C += X * X'
	end

    C /= nsegments * (seglen - order)
	
	return C
end

"""
	granger_jackknife(data, seglen, order, Covs12, Covs1, Covs2)

Jackknife sampling method for estimating the error standard deviation

### Arguments

  - `data::AbstractArray`: Nx2 array for N data points in 2 channels.
  - `order::Int`: Model order. Assumed time delay order of interest.
  - `seglen::Integer`: segment length.
  - `Covs12::Array{Float64, 2}`: Concatenated COvariances of 0 to the given order
  - `Covs1::Array{Float64, 2}`: Channel 1 Concatenated variances of 0 to the given order
  - `Covs2::Array{Float64, 2}`: Channel 2 Concatenated variances of 0 to the given order

### Returns

  - `::Float64`: estimated standard deviation of error
"""
function granger_jackknife(data::Array{Float64, 2},
						  seglen::Int,
						  order::Int,
						  Covs12::AbstractArray{Float64, 2},
						  Covs1::AbstractArray{Float64, 2},
						  Covs2::AbstractArray{Float64, 2})

	nsamples = size(data, 1)  # number of samples
	nsegments = int(nsamples / seglen)  # number of segments
	trials = zeros(nsegments, 1)  
	
	Threads.@threads for i in 1:nsegments
		Segment = view(data, (i-1)*seglen+1:i*seglen, :)
		
		Covs12_ = est_sig_covs(Segment, order)
		Covs1_ = est_sig_covs(review(Segment, 1), order)
		Covs2_ = est_sig_covs(review(Segment, 2), order)
		
		Covs12_s = (nsegments * Covs12 - Covs12_) / (nsegments - 1)
		Covs1_s = (nsegments * Covs1 - Covs1_) / (nsegments - 1)
		Covs2_s = (nsegments * Covs2 - Covs2_) / (nsegments - 1)
		
		Acoef12_s, _ = mvar_est(Covs12_s)
		Acoef1_s, _ = mvar_est(Covs1_s)
		Acoef2_s, _ = mvar_est(Covs2_s)

		SegmentC = vcat(view(data, 1:(i-1)*seglen, :),
			view(data, (i+1)*seglen .+ 1:nsamples, :))

		Σ12_s = est_noise_covs(SegmentC, Acoef12_s, seglen)
		Σ1_s = est_noise_covs(review(SegmentC, 1), Acoef1_s, seglen)
		Γ1_s = est_noise_covs(review(SegmentC, 2), Acoef2_s, seglen)
		Σ2_s = Σ12_s[1, 1]
		Γ2_s = Σ12_s[2, 2]
		
		X_to_Y_s = log(Γ1_s / Γ2_s)
		Y_to_X_s = log(Σ1_s / Σ2_s)
		
		trials[i, :] = X_to_Y_s - Y_to_X_s
		
	end
	return sqrt(nsegments) * std(trials)
end

"""
	granger_aic(data, seglen, order_range)

	Akaike Information Criterion
"""
function granger_aic(data::Array{Float64, 2},
					 seglen::Int,
					 order_range::UnitRange{Int64},)
	
	nchan = size(data, 2)
	aic_range = Array{Float64}(undef, size(order_range))
	
	for (i, order) in enumerate(order_range)
		Covs = est_sig_covs(data, order)
		Acoef, _ = mvar_est(Covs)
		Σ = est_noise_covs(data, Acoef, seglen)
		aic_range[i] = 2 * log(det(Σ)) + (2 * order * nchan^2 / seglen)
	end
	return aic_range
end

"""
	granger_bic(data, seglen, order_range)

Bayesian Information Criterion
"""
function granger_bic(data::Array{Float64, 2},
					 seglen::Int,
				     order_range::UnitRange{Int64})
	
	nchan = size(data, 2)
	bic_range = Array{Float64}(undef, size(order_range))
	
	for (i, order) in enumerate(order_range)
		Covs = est_sig_covs(data, order)
		Acoef, _ = mvar_est(Covs)
		Σ = est_noise_covs(data, Acoef, seglen)
		bic_range[i] = 2 * log(det(Σ)) + (2 * order * nchan^2 * log(seglen) / seglen)
	end
	return bic_range
end
