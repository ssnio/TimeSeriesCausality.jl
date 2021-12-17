"""
    int(x) = trunc(Int, x)
"""
int(x) = trunc(Int, x)

"""
    dropmean(X, d) = dropdims(mean(X, dims=d), dims=d)
"""
function dropmean(X, d)
    if ndims(X) == 1
        mean(X; dims=d)
    else
        dropdims(mean(X; dims=d); dims=d)
    end
end

"""
    squeeze(X::AbstractArray)

removing singleton dimensions
"""
function squeeze(X::AbstractArray)
    keepd = Tuple(d for d in size(X) if d != 1)
    return reshape(X, keepd)
end

"""
    detrend!(data, n)

(in place) Linear detrend of signals along first axis removing the n-th order polynomial.
This detrend function is limited to linear orders (0th and 1st order).

### Arguments

  - `data::AbstractArray`: N-dim array where signal is in column-major order
  - `n::Integer`: `n = 0` subtracts mean from data, `n = 1` removes linear trend

**Note**: shape of data must be (signal length, ...)
"""
function detrend!(data::AbstractArray, n::Integer)
    original_shape = size(data)
    nsamp = size(data, 1)  # number of samples

    A = Array{Float64}([ones(nsamp) Array(1:nsamp)])

    data = reshape(data, (nsamp, :))  # reshaping data
    if n == 0
        data .-= mean(data; dims=1)
    elseif n == 1
        data .-= A * (A \ data)
    end
    return reshape(data, original_shape)
end

"""
    window = hanning_fun(N)

Hanning window similar to MATLAB `hanning` implementation
"""
function hanning_fun(N::Integer)
    x = [range(0.0, 1.0; length=N + 2);]
    window = 0.5 .* (1 .- cospi.(2 .* x))
    window = (window + window[end:-1:1]) ./ 2  # forcing symmetry
    return window[2:(end - 1)]  # excluding the zero values
end

"""
    review(X, dim)

Reshape + View
"""
review(X, dim) = reshape(view(X, :, dim), :, 1)

"""
    pink_noise(n, c)

Pink noise with `n` samples and pinkness of:
```math
P_s \\propto \\frac{1}{f^c} .
```

### Arguments

  - `n::Integer`: Number of samples
  - `c::Float64`: pinkness order (frequency power)

"""
function pink_noise(n::Integer, c::Float64)
    nf = int(n / 2)  # Nyquist frequency
    x = reshape(range(start=0.0, stop=1.0, length=n), 1, :)
    f = reshape(range(start=1, stop=nf), :, 1)
    phi = rand(nf)
    y = f * x .+ phi
    z = sin.(2 * pi * y) ./ (f .^ c)
    return sum(z; dims=1)[1, :]
end
