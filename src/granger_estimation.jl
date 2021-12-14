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

internal variables:
  - Covs: Concatenated covariance mats of different orders
  - Acoef: A matrix coefficients
  - Perr: Prediction Error
  - Grind: Granger index
"""
function granger_est(data::Array{Float64, 2}, order::Int, seglen::Int, misc="jackknife", verbose::Bool=true)

	data = prep_data_granger(data, seglen, verbose)
	
	Covs12 = est_covs(data, order)
	Covs1 = est_covs(review(data, 1), order)
	Covs2 = est_covs(review(data, 2), order)
	
	Acoef12, Perr12 = mvar_est(Covs12)
	Acoef1, _ = mvar_est(Covs1)
	Acoef2, _ = mvar_est(Covs2)
	
	Σ12 = Perr12[:, end-1:end]
	Σ12, _ = ΣEstimate(data, Acoef12, seglen)
	Σ1, _ = ΣEstimate(review(data, 1), Acoef1, seglen)
	Γ1, _ = ΣEstimate(review(data, 2), Acoef2, seglen)
	Σ2 = Σ12[1, 1]
	Γ2 = Σ12[2, 2]
	
	X_to_Y = log(Γ1 / Γ2)
	Y_to_X = log(Σ1 / Σ2)
	
	Grind = (X_to_Y - Y_to_X)[1]

	if lowercase(misc) == "jackknife"
		Grind_std = granger_jackknife(data, order, seglen, Covs12, Covs1, Covs2)
		return Grind, Grind_std
	else
		return Grind
	end
	
end

"""
	est_cov(X, order)

Estimates covariances of orders up to the given order

"""
function est_covs(data::AbstractArray{Float64, 2}, order::Int=NaN)

	nsamples, nchan = size(data) # number of rows (samples) and columns (channels)
	
	if isnan(order)
		order = nsamples - 1
	end
	
	Covs = Array{Float64}(undef, nchan, nchan*(order+1))
	
	Covs[:, 1:nchan] = covariance(data)
	for k = 1:order
		@views Covs[:, (1:nchan).+k*nchan] = covariance(data[k+1:nsamples, :], data[1:nsamples-k, :])
	end
	return Covs
end

"""

multivariate autoregressive model parameter estimation
Levinson-Wiggens-Robinson (LWR) algorithm using unbiased correlation function
"""
function mvar_est(Covs::Array{Float64, 2})

	nC, nR = size(Covs)  # number of rows and columns
	order = int(nR / nC) - 1

	ARF = Array{Float64}(undef, nC, order*nC)
	ARB = Array{Float64}(undef, nC, order*nC)
	RCF = Array{Float64}(undef, nC, order*nC)
	RCB = Array{Float64}(undef, nC, order*nC)
	PE = Array{Float64}(undef, nC, (order+1)*nC)
	
    PE[:, 1:nC] = Covs[:, 1:nC]
	PEF = Covs[:, 1:nC]  # it's a different normalization in BioSig
	PEB = Covs[:, 1:nC]  # it's a different normalization in BioSig
	
	for k in 1:order
		S1 = k*nC .+ (1:nC)  # slice
		S2 = k*nC .+ (1-nC:0)  # slice
		
		D = Covs[:, S1]
		for L in 1:k-1
			S3 = L*nC .+ (1-nC:0)  # slice
			D -= ARF[:, S3] * Covs[:, (k-L)*nC .+ (1:nC)]
		end
		
		ARF[:, S2] = D / PEB
        ARB[:, S2] = D' / PEF
		for L in 1:k-1
			S3 = L*nC .+ (1-nC:0)  # slice
			S4 = (k-L)*nC .+ (1-nC:0)  # slice
			ARB[:, S4], ARF[:, S3] = (ARB[:, S4] - ARB[:, S2] * ARF[:, S3],
									  ARF[:, S3] - ARF[:, S2] * ARB[:, S4])
		end
		
		RCF[:, S2] = ARF[:, S2]
        RCB[:, S2] = ARB[:, S2]
        PEF = (identity(nC) - ARF[:, S2] * ARB[:, S2]) * PEF
        PEB = (identity(nC) - ARB[:, S2] * ARF[:, S2]) * PEB
        PE[:, S1] = PEF
	end
	
	return ARF, PE
end

function ΣEstimate(X::AbstractArray{Float64, 2}, A::AbstractArray{Float64, 2}, eplen::Int)
	nR, nC = size(X)  # number of rows and columns
	nA = size(A, 2)
	P = int(nA / nC)
	nep = int(nR / eplen)
	Ax = reshape(A, nC, nC, :)
    C = zeros(nC, nC)
    S = zeros(nC, nC)

	@inbounds for e in 1:nep
		S1 = (e-1)*eplen+1:e*eplen
		S2 = P+1:eplen
		@views Xx = X[S1, :][S2, :]'
		S += Xx * Xx'
		for i in 1:P
			@views Xx -= Ax[:, :, i] * X[S1, :][S2 .- i, :]'
		end
		C += Xx * Xx'
	end

    S /= nep * (eplen-P)
    C /= nep * (eplen-P)
	
	return C , S
end

function granger_jackknife(data, order, seglen, R12, R1, R2)
	nsamples = size(data, 1)  # number of samples
	nSeg = int(nsamples / seglen)  # number of segments
	g_Seg = zeros(nSeg, 1)  
	
	Threads.@threads for i in 1:nSeg
		Segment = view(data, (i-1)*seglen+1:i*seglen, :)
		
		R12_ = est_covs(Segment, order)
		R1_ = est_covs(review(Segment, 1), order)
		R2_ = est_covs(review(Segment, 2), order)
		
		R12_s = (nSeg * R12 - R12_) / (nSeg - 1)
		R1_s = (nSeg * R1 - R1_) / (nSeg - 1)
		R2_s = (nSeg * R2 - R2_) / (nSeg - 1)
		
		Acoef12_s, PE12_s = mvar_est(R12_s)
		Acoef1_s, PE1_s = mvar_est(R1_s)
		Acoef2_s, PE2_s = mvar_est(R2_s)

		SegmentC = vcat(view(data, 1:(i-1)*seglen, :),
			view(data, (i+1)*seglen .+ 1:nsamples, :))

		Σ12_s, _ = ΣEstimate(SegmentC, Acoef12_s, seglen)
		Σ1_s, _ = ΣEstimate(review(SegmentC, 1), Acoef1_s, seglen)
		Γ1_s, _ = ΣEstimate(review(SegmentC, 2), Acoef2_s, seglen)
		Σ2_s = Σ12_s[1, 1]
		Γ2_s = Σ12_s[2, 2]
		
		Fx2y_s = log(Γ1_s / Γ2_s)
		Fy2x_s = log(Σ1_s / Σ2_s)
		
		g_Seg[i, :] = Fx2y_s - Fy2x_s
		
	end
	return sqrt(nSeg) * std(g_Seg)
end

"""
	granger_aic(data, order_range, seglen)

	Akaike Information Criterion
"""
function granger_aic(data::Array{Float64, 2}, order_range, seglen::Int)
	
	nchan = size(data, 2)
	aic_range = Array{Float64}(undef, size(order_range))
	
	for (i, order) in enumerate(order_range)
		Covs = est_covs(data, order)
		Acoef, _ = mvar_est(Covs)
		Σ, _ = ΣEstimate(data, Acoef, seglen)
		aic_range[i] = 2 * log(det(Σ)) + (2 * order * nchan^2 / seglen)
	end
	return aic_range
end

"""
	granger_bic(data, order_range, seglen)

Bayesian Information Criterion
"""
function granger_bic(data::Array{Float64, 2}, order_range, seglen::Int)
	
	nchan = size(data, 2)
	bic_range = Array{Float64}(undef, size(order_range))
	
	for (i, order) in enumerate(order_range)
		Covs = est_covs(data, order)
		Acoef, _ = mvar_est(Covs)
		Σ, _ = ΣEstimate(data, Acoef, seglen)
		bic_range[i] = 2 * log(det(Σ)) + (2 * order * nchan^2 * log(seglen) / seglen)
	end
	return bic_range
end
