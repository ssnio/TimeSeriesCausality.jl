function granger_est(data::Array{Float64, 2}, order::Int, seglen::Int, misc="jackknife")
	
	if size(data, 1) < size(data, 2)
		data = reshape(data, size(data, 2), size(data, 1))
	end
	if size(data, 2) != 2
		throw("Only 2 channels are supported!")
	end
	
	R12 = est_covs(data, order)
	R1 = est_covs(review(data, 1), order)
	R2 = est_covs(review(data, 2), order)
	
	Acoef12, PE12 = MVAR_M6(R12)
	Acoef1, _ = MVAR_M6(R1)
	Acoef2, _ = MVAR_M6(R2)
	
	Σ12 = PE12[:, end-1:end]
	Σ12 = ΣEstimate(data, Acoef12, seglen)
	Σ1 = ΣEstimate(review(data, 1), Acoef1, seglen)
	Γ1 = ΣEstimate(review(data, 2), Acoef2, seglen)
	Σ2 = Σ12[1, 1]
	Γ2 = Σ12[2, 2]
	
	Fx2y = log(Γ1 / Γ2)
	Fy2x = log(Σ1 / Σ2)
	# Fxoy = log(Σ2 * Γ2 / det(Σ12))
	
	g = (Fx2y - Fy2x)[1]

	if lowercase(misc) == "jackknife"
		g_std = granger_jackknife(data, order, seglen, R12, R1, R2)
		return g, g_std
	else
		return g
	end
	
end

"""
	est_cov(X, order)

Estimates covariances of orders up to the given order

"""
function est_covs(X::AbstractArray{Float64, 2}, order::Int=NaN)

	nR, nC = size(X) # number of rows (samples) and columns (channels)
	
	if isnan(order)
		order = nR - 1
	end
	
	Covs = Array{Float64}(undef, nC, nC*(order+1))
	
	Covs[:, 1:nC] = covariance(X)
	for k = 1:order
		@views Covs[:, (1:nC).+k*nC] = covariance(X[k+1:nR, :], X[1:nR-k, :])
	end
	return Covs
end

function MVAR_M6(R::Array{Float64, 2})

	nC, nR = size(R)
	order = int(nR / nC) - 1

	ARF = Array{Float64}(undef, nC, order*nC)
	ARB = Array{Float64}(undef, nC, order*nC)
	RCF = Array{Float64}(undef, nC, order*nC)
	RCB = Array{Float64}(undef, nC, order*nC)
	PE = Array{Float64}(undef, nC, (order+1)*nC)
	
    PE[:, 1:nC] = R[:, 1:nC]
	PEF = R[:, 1:nC]  # it's a different normalization in BioSig
	PEB = R[:, 1:nC]  # it's a different normalization in BioSig
	
	for k in 1:order
		S1 = k*nC .+ (1:nC)  # slice
		S2 = k*nC .+ (1-nC:0)  # slice
		
		D = R[:, S1]
		for L in 1:k-1
			S3 = L*nC .+ (1-nC:0)  # slice
			D -= ARF[:, S3] * R[:, (k-L)*nC .+ (1:nC)]
		end
		
		ARF[:, S2] = D / PEB
        ARB[:, S2] = D' / PEF
		for L in 1:k-1
			S3 = L*nC .+ (1-nC:0)  # slice
			S4 = (k-L)*nC .+ (1-nC:0)  # slice
			# tmp = ARB[:, S4]
			# ARB[:, S4] -= ARB[:, S2] * ARF[:, S3]
			# ARF[:, S3] -= ARF[:, S2] * tmp
			ARB[:, S4], ARF[:, S3] = (ARB[:, S4] - ARB[:, S2] * ARF[:, S3],
				ARF[:, S3] - ARF[:, S2] * ARB[:, S4])
			
		end
		
		RCF[:, S2] = ARF[:, S2]
        RCB[:, S2] = ARB[:, S2]
        PEF = (identity(nC) - ARF[:, S2] * ARB[:, S2]) * PEF
        PEB = (identity(nC) - ARB[:, S2] * ARF[:, S2]) * PEB
        PE[:, S1] = PEF
	end
	
	# DC = zeros(nC, nC)
	# for k in 1:order
	# 	S2 = k*nC .+ (1-nC:0)
	# 	DC += + ARF[:, S2]' .^ 2
	# end

	return ARF, PE #, RCF, DC
end

function ΣEstimate(X::AbstractArray{Float64, 2}, A::AbstractArray{Float64, 2}, eplen::Int)
	nR, nC = size(X)
	nA = size(A, 2)
	P = int(nA / nC)
	nEpoch = int(nR / eplen)
	Ax = reshape(A, nC, nC, :)
    C = zeros(nC, nC)
    S = zeros(nC, nC)
	@inbounds for e in 1:nEpoch
		S1 = (e-1)*eplen+1:e*eplen
		S2 = P+1:eplen
		@views Xx = X[S1, :][S2, :]'
		S += Xx * Xx'
		for i in 1:P
			@views Xx -= Ax[:, :, i] * X[S1, :][S2 .- i, :]'
		end
		C += Xx * Xx'
	end
    S /= nEpoch * (eplen-P)
    C /= nEpoch * (eplen-P)
	
	return C #, S
end

function granger_jackknife(data, order, seglen, R12, R1, R2)
	nSamples = size(data, 1)
	nSeg = int(nSamples / seglen)
	g_Seg = zeros(nSeg, 1)
	
	Threads.@threads for i in 1:nSeg
		Segment = view(data, (i-1)*seglen+1:i*seglen, :)
		
		R12_ = est_covs(Segment, order)
		R1_ = est_covs(review(Segment, 1), order)
		R2_ = est_covs(review(Segment, 2), order)
		
		R12_s = (nSeg * R12 - R12_) / (nSeg - 1)
		R1_s = (nSeg * R1 - R1_) / (nSeg - 1)
		R2_s = (nSeg * R2 - R2_) / (nSeg - 1)
		
		Acoef12_s, PE12_s = MVAR_M6(R12_s)
		Acoef1_s, PE1_s = MVAR_M6(R1_s)
		Acoef2_s, PE2_s = MVAR_M6(R2_s)
		
		# Σ1_s = PE1_s[1, end]
		# Γ1_s = PE2_s[1, end]
		# Σ2_s = PE12_s[1, end-1]
		# Γ2_s = PE12_s[2, end]
		
		SegmentC = vcat(view(data, 1:(i-1)*seglen, :),
			view(data, (i+1)*seglen .+ 1:nSamples, :))

		Σ12_s = ΣEstimate(SegmentC, Acoef12_s, seglen)
		Σ1_s = ΣEstimate(review(SegmentC, 1), Acoef1_s, seglen)
		Γ1_s = ΣEstimate(review(SegmentC, 2), Acoef2_s, seglen)
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
	
	nsamples, nchannels = size(data)
	aic_range = Array{Float64}(undef, size(order_range))
	
	for (i, order) in enumerate(order_range)
		R = est_covs(data, order)
		Acoef, _ = MVAR_M6(R)
		Σ = ΣEstimate(data, Acoef, seglen)
		aic_range[i] = 2 * log(det(Σ)) + (2 * order * nchannels^2 / seglen)
	end
	return aic_range
end

"""
	granger_bic(data, order_range, seglen)

Bayesian Information Criterion
"""
function granger_bic(data::Array{Float64, 2}, order_range, seglen::Int)
	
	nsamples, nchannels = size(data)
	bic_range = Array{Float64}(undef, size(order_range))
	
	for (i, order) in enumerate(order_range)
		R = est_covs(data, order)
		Acoef, _ = MVAR_M6(R)
		Σ = ΣEstimate(data, Acoef, seglen)
		bic_range[i] = 2 * log(det(Σ)) + (2 * order * nchannels^2 * log(seglen) / seglen)
	end
	return bic_range
end
