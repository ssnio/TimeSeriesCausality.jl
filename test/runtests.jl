using TimeSeriesCausality
using Test
using Statistics: mean
using Distributions: MvNormal


@time @testset "granger_estimation.jl" begin
    time_steps = 1000000
    segment_length = 1000
    noise_cov = MvNormal([0.25 0.0; 0.0 0.64])  # uncorrelated cov matrix of noise
    noise = rand(noise_cov, time_steps)'
    
    # initial values
    signal = zeros(time_steps, 2)
    signal[1:3, :] = rand(3, 2) + noise[1:3, :]
    
    # simulation
    designer = [0.4 -0.6  0.8  0.0  0.0  0.0;
                0.5  0.9  0.0  0.0  0.0  0.7]

    # designer = [0.0  0.6  0.0  0.4  0.0  0.8;
    #             0.0  0.0  0.0  0.5 -0.9  0.7]
    
    for t in 4:time_steps
        signal[t, :] = designer * reshape(signal[t-3:t-1, :], :, 1) + noise[t, :]
    end
    
    covariances = TimeSeriesCausality.est_sig_covs(signal, 3)
    
    ar_factors, p_error = TimeSeriesCausality.mvar_est(covariances)
    begin
        @test all(isapprox.(ar_factors[:, 1], designer[:, 3], atol=0.01))
        @test all(isapprox.(ar_factors[:, 2], designer[:, 6], atol=0.01))
        @test all(isapprox.(ar_factors[:, 3], designer[:, 2], atol=0.01))
        @test all(isapprox.(ar_factors[:, 4], designer[:, 5], atol=0.01))
        @test all(isapprox.(ar_factors[:, 5], designer[:, 1], atol=0.01))
        @test all(isapprox.(ar_factors[:, 6], designer[:, 4], atol=0.01))
    end
    
    grager_idx = granger_est(signal, 3, segment_length, "none")
    @test grager_idx > 0.5

    order_range = 1:7
    best_aic = argmin(granger_aic(signal, order_range, segment_length))
    @test best_aic == 3
    best_bic = argmin(granger_bic(signal, order_range, segment_length))
    @test best_bic == 3
end


@time @testset "phase_slope_index.jl" begin
    # tests of psi_est ###############################################
    # two random signals
    signal = [[randn(1000000);] [randn(1000000);]]
    psi, _ = psi_est(signal, 100; subave=true, segave=true)
    @test all(isapprox(psi, zeros(2, 2); atol=0.01))

    # induced causality should always be inferred independent of optional arguments
    for eplen_ in [0, 200], detrend_ in [true, false]
        for segave_ in [true, false], subave_ in [true, false]
            for method_ in ["jackknife", "bootstrap"]
                # channel 2 being exactly channel 1
                ch1_ = randn(1000)
                signal = [[ch1_;] [ch1_;]]
                psi, _ = psi_est(
                    signal,
                    100;
                    method=method_,
                    eplen=eplen_,
                    detrend=detrend_,
                    segave=segave_,
                    subave=subave_,
                )
                @test psi == zeros(2, 2)

                # channel 1 leading channel 2
                ch1_ = randn(100000)
                signal = [[ch1_[2:end];] [ch1_[1:(end - 1)];]]
                psi, _ = psi_est(
                    signal,
                    100;
                    method=method_,
                    eplen=eplen_,
                    detrend=detrend_,
                    segave=segave_,
                    subave=subave_,
                )
                @test psi[1, 1] == 0.0 && psi[2, 2] == 0.0
                @test psi[1, 2] > 1.0 && psi[2, 1] + psi[1, 2] == 0.0

                # channel 2 leading channel 1
                ch2_ = randn(100000)
                signal = [[ch2_[1:(end - 1)];] [ch2_[2:end];]]
                psi, _ = psi_est(
                    signal,
                    100;
                    method=method_,
                    eplen=eplen_,
                    detrend=detrend_,
                    segave=segave_,
                    subave=subave_,
                )
                @test psi[1, 1] == 0.0 && psi[2, 2] == 0.0
                @test psi[1, 2] < -1.0 && psi[2, 1] + psi[1, 2] == 0.0
            end
        end
    end

    # test freqlist ###################################################
    ch1_ = randn(100000)
    signal = [[ch1_[2:end];] [ch1_[1:(end - 1)];]]
    psi_range, _ = psi_est(signal, 100; freqlist=1:49)
    psi_default, _ = psi_est(signal, 100)  # default is based on seglen
    @test all(psi_range == psi_default)

    # auxiliary function for testing
    """
        sin_sum = v_sin(X, A, Ω, Φ)

    returns an array summed over series of Sin waves with A-amplitudes,
        Ω-frequencies and Φ-phase delay
    """
    v_sin(X, A, Ω, Φ) = sum([a .* sin.(ω .* X .+ ϕ) for (a, ω, ϕ) in zip(A, Ω, Φ)])

    nsamples_ = 200000
    fs_ = 100  # sampling frequency (Hz)
    f_range_ = 12:30  # beta frequency band [12-30] Hz
    Ω_ = f_range_ * 2pi
    X_ = range(0.0, nsamples_ / fs_; length=nsamples_)
    A_ = rand(0.1:0.01:1.0, size(f_range_, 1))
    Φ_ = rand(size(f_range_, 1))
    Y_ = v_sin(X_, A_, Ω_, Φ_)
    Y_ch1, Y_ch2 = Y_[5:end], Y_[1:(end - 4)]
    noise_ = randn(size(Y_ch1))
    signal = [Y_ch1 Y_ch2] .+ noise_
    seglen_ = fs_
    psi_full_range, _ = psi_est(signal, seglen_)
    psi_low, _ = psi_est(signal, seglen_; freqlist=1:11)
    psi_freq, _ = psi_est(signal, seglen_; freqlist=f_range_)
    psi_high, _ = psi_est(signal, seglen_; freqlist=31:49)
    @test psi_full_range[1, 2] > 1.0  # ch1 leading ch2
    @test psi_freq[1, 2] > psi_full_range[1, 2]  # PSI higher in target frequency range
    @test psi_freq[1, 2] > psi_low[1, 2]  # PSI higher in target frequency range
    @test psi_freq[1, 2] > psi_high[1, 2]  # PSI higher in target frequency range

    # test data2para ##################################################
    # ndims(data) should be 2
    signal = rand(100, 3, 2)
    @test_throws DimensionMismatch psi_est(signal, 100)

    # squeeze
    signal = rand(100, 1, 4)
    @test psi_est(signal, 100)[1] == psi_est(signal[:, 1, :], 100)[1]

    # size(data, 1) >!> seglen
    signal = rand(50, 3)
    @test_throws DimensionMismatch psi_est(signal, 100)

    # if eplen == 0 then no std estimation
    signal = [[randn(1000000);] [randn(1000000);]]
    _, psi_std = psi_est(signal, 100)
    @test all(isnan.(psi_std))

    # if continuous data, subave shall not be true otherwise NaN is returned
    signal = [[randn(1000000);] [randn(1000000);]]
    psi, _ = psi_est(signal, 100; subave=true)
    @test !all(isnan.(psi))

    # tests of int ####################################################
    @test TimeSeriesCausality.int(3.14) == 3
    @test TimeSeriesCausality.int(-2.72) == -2
    @test TimeSeriesCausality.int(0.0) == 0

    # tests of dropmean ###############################################
    test_a = rand(13)
    @test TimeSeriesCausality.dropmean(test_a, 1) == mean(test_a; dims=1)
    test_a = rand(3, 5)
    @test TimeSeriesCausality.dropmean(test_a, 1) == mean(test_a; dims=1)[1, :]
    test_a = rand(7, 11)
    @test TimeSeriesCausality.dropmean(test_a, 2) == mean(test_a; dims=2)[:, 1]

    # tests of squeeze ################################################
    test_a = rand(17)
    @test TimeSeriesCausality.squeeze(test_a) == test_a
    test_a = rand(3, 5, 7)
    @test TimeSeriesCausality.squeeze(test_a) == test_a
    test_a = rand(3, 1, 7)
    @test TimeSeriesCausality.squeeze(test_a) == test_a[:, 1, :]

    # tests of detrend ################################################
    # testing the inplace
    test_x = [0:0.01:2.0;]
    zeros_x = zeros(size(test_x))
    @test all(isapprox.(TimeSeriesCausality.detrend!(test_x, 1), zeros_x, atol=0.001))
    @test all(isapprox.(test_x, zeros_x, atol=0.001))

    test_x = [0:0.01:2.0;]
    @test all(isapprox.(TimeSeriesCausality.detrend!(test_x, 0), [-1.0:0.01:1.0;], atol=0.001))
    test_x = [0:0.01:2.0;]
    @test all(isapprox.(TimeSeriesCausality.detrend!(test_x, 1), zeros_x, atol=0.001))

    test_x = [0:0.01:(2pi);]
    test_y = sin.(test_x)
    @test all(isapprox.(TimeSeriesCausality.detrend!(test_y, 0), test_y, atol=0.001))
    test_x = [0:0.01:(2pi);]
    test_y = sin.(test_x)
    @test all(isapprox.(TimeSeriesCausality.detrend!(test_y, 1), test_y, atol=0.001))

    test_x = [0:0.01:(2pi);]
    test_y = cos.(test_x) .+ test_x
    aux_x = [0:0.01:(2pi);] .- pi
    aux_y = cos.(test_x) .+ aux_x
    @test all(isapprox.(TimeSeriesCausality.detrend!(test_y, 0), aux_y, atol=0.001))
    test_x = [0:0.01:(2pi);]
    test_y = cos.(test_x) .+ test_x
    @test all(isapprox.(TimeSeriesCausality.detrend!(test_y, 1), cos.(test_x), atol=0.01))

    # tests of hanning_fun ############################################
    # the arrays are from MATLAB `hanning` function.
    hann_12 = [
        0.057271987173395
        0.215967626634422
        0.439731659872338
        0.677302443521268
        0.874255374085551
        0.985470908713026
        0.985470908713026
        0.874255374085551
        0.677302443521268
        0.439731659872338
        0.215967626634422
        0.057271987173395
    ]
    hann_13 = [
        0.049515566048790
        0.188255099070633
        0.388739533021843
        0.611260466978157
        0.811744900929367
        0.950484433951210
        1.000000000000000
        0.950484433951210
        0.811744900929367
        0.611260466978157
        0.388739533021843
        0.188255099070633
        0.049515566048790
    ]
    hanning_window_12 = TimeSeriesCausality.hanning_fun(12)
    hanning_window_13 = TimeSeriesCausality.hanning_fun(13)
    @test all(hanning_window_12 .≈ hann_12)
    @test all(hanning_window_13 .≈ hann_13)
    @test hanning_window_12[1] == hanning_window_12[end]
    @test hanning_window_13[1] == hanning_window_13[end]
    @test hanning_window_12[1] > 0.0
    @test hanning_window_13[1] > 0.0
end
