using PhaseSlopeIndex
using Test
using Statistics: mean

@time @testset "PhaseSlopeIndex.jl" begin
    # tests of data2psi ###############################################
    # two random signals
    signal = [[randn(1000000);] [randn(1000000);]]
    psi, _ = data2psi(signal, 100; subave=true, segave=true)
    @test all(isapprox(psi, zeros(2, 2); atol=0.01))

    # induced causality should always be inferred independent of optional arguments
    for eplen_ in [0, 200], detrend_ in [true, false]
        for segave_ in [true, false], subave_ in [true, false]
            for method_ in ["jackknife", "bootstrap"]
                # channel 2 being exactly channel 1
                ch1_ = randn(1000)
                signal = [[ch1_;] [ch1_;]]
                psi, _ = data2psi(
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
                psi, _ = data2psi(
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
                psi, _ = data2psi(
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
    psi_range, _ = data2psi(signal, 100; freqlist=1:49)
    psi_default, _ = data2psi(signal, 100)  # default is based on seglen
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
    psi_full_range, _ = data2psi(signal, seglen_)
    psi_low, _ = data2psi(signal, seglen_; freqlist=1:11)
    psi_freq, _ = data2psi(signal, seglen_; freqlist=f_range_)
    psi_high, _ = data2psi(signal, seglen_; freqlist=31:49)
    @test psi_full_range[1, 2] > 1.0  # ch1 leading ch2
    @test psi_freq[1, 2] > psi_full_range[1, 2]  # PSI higher in target frequency range
    @test psi_freq[1, 2] > psi_low[1, 2]  # PSI higher in target frequency range
    @test psi_freq[1, 2] > psi_high[1, 2]  # PSI higher in target frequency range

    # test data2para ##################################################
    # ndims(data) should be 2
    signal = rand(100, 3, 2)
    @test_throws DimensionMismatch data2psi(signal, 100)

    # squeeze
    signal = rand(100, 1, 4)
    @test data2psi(signal, 100)[1] == data2psi(signal[:, 1, :], 100)[1]

    # size(data, 1) >!> seglen
    signal = rand(50, 3)
    @test_throws DimensionMismatch data2psi(signal, 100)

    # if eplen == 0 then no std estimation
    signal = [[randn(1000000);] [randn(1000000);]]
    _, psi_std = data2psi(signal, 100)
    @test all(isnan.(psi_std))

    # if continuous data, subave shall not be true otherwise NaN is returned
    signal = [[randn(1000000);] [randn(1000000);]]
    psi, _ = data2psi(signal, 100; subave=true)
    @test !all(isnan.(psi))

    # tests of int ####################################################
    @test PhaseSlopeIndex.int(3.14) == 3
    @test PhaseSlopeIndex.int(-2.72) == -2
    @test PhaseSlopeIndex.int(0.0) == 0

    # tests of dropmean ###############################################
    test_a = rand(13)
    @test PhaseSlopeIndex.dropmean(test_a, 1) == mean(test_a; dims=1)
    test_a = rand(3, 5)
    @test PhaseSlopeIndex.dropmean(test_a, 1) == mean(test_a; dims=1)[1, :]
    test_a = rand(7, 11)
    @test PhaseSlopeIndex.dropmean(test_a, 2) == mean(test_a; dims=2)[:, 1]

    # tests of squeeze ################################################
    test_a = rand(17)
    @test PhaseSlopeIndex.squeeze(test_a) == test_a
    test_a = rand(3, 5, 7)
    @test PhaseSlopeIndex.squeeze(test_a) == test_a
    test_a = rand(3, 1, 7)
    @test PhaseSlopeIndex.squeeze(test_a) == test_a[:, 1, :]

    # tests of detrend ################################################
    # testing the inplace
    test_x = [0:0.01:2.0;]
    zeros_x = zeros(size(test_x))
    @test all(isapprox.(PhaseSlopeIndex.detrend!(test_x, 1), zeros_x, atol=0.001))
    @test all(isapprox.(test_x, zeros_x, atol=0.001))

    test_x = [0:0.01:2.0;]
    @test all(isapprox.(PhaseSlopeIndex.detrend!(test_x, 0), [-1.0:0.01:1.0;], atol=0.001))
    test_x = [0:0.01:2.0;]
    @test all(isapprox.(PhaseSlopeIndex.detrend!(test_x, 1), zeros_x, atol=0.001))

    test_x = [0:0.01:(2pi);]
    test_y = sin.(test_x)
    @test all(isapprox.(PhaseSlopeIndex.detrend!(test_y, 0), test_y, atol=0.001))
    test_x = [0:0.01:(2pi);]
    test_y = sin.(test_x)
    @test all(isapprox.(PhaseSlopeIndex.detrend!(test_y, 1), test_y, atol=0.001))

    test_x = [0:0.01:(2pi);]
    test_y = cos.(test_x) .+ test_x
    aux_x = [0:0.01:(2pi);] .- pi
    aux_y = cos.(test_x) .+ aux_x
    @test all(isapprox.(PhaseSlopeIndex.detrend!(test_y, 0), aux_y, atol=0.001))
    test_x = [0:0.01:(2pi);]
    test_y = cos.(test_x) .+ test_x
    @test all(isapprox.(PhaseSlopeIndex.detrend!(test_y, 1), cos.(test_x), atol=0.01))

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
    hanning_window_12 = PhaseSlopeIndex.hanning_fun(12)
    hanning_window_13 = PhaseSlopeIndex.hanning_fun(13)
    @test all(hanning_window_12 .≈ hann_12)
    @test all(hanning_window_13 .≈ hann_13)
    @test hanning_window_12[1] == hanning_window_12[end]
    @test hanning_window_13[1] == hanning_window_13[end]
    @test hanning_window_12[1] > 0.0
    @test hanning_window_13[1] > 0.0
end
