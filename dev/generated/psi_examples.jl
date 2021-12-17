using TimeSeriesCausality
using Plots: plot, heatmap, cgrad
using DSP: blackman

# data generation
n_channels = 4  # number of channels
n_samples = 2^16  # number of data points measured in each channel
fs = 128  # sampling frequency
time_array = Array{Float64}(range(1; step=1 / fs, length=n_samples))

# mixed data
range_c4 = 1:n_samples
range_c1 = range_c4 .+ 16
range_c3 = range_c1 .- 1

rand_data = randn(Float64, (n_samples + 16, 1)) # uniform noise
cause_source = rand_data[range_c1]  # channel 1
random_source = randn(Float64, n_samples)  # channel 2, uniform noise
effect_source = rand_data[range_c3]  # channel 3
weak_effect = rand_data[range_c4] .- (randn(Float64, (n_samples, 1)) / 5) # channel 4
mixed_data = hcat(cause_source, random_source, effect_source, weak_effect)

p1 = plot(
    time_array[1:64],
    mixed_data[1:64, :];
    title="Mixed data",
    label=["Cause" "Random" "Effect" "Noisy Effect"],
    linecolor=["red" "green" "blue" "magenta"],
)

plot(p1; layout=(1, 1), size=(800, 450))

segment_length = 100  # segment length
nboot = 256  # number of bootstrap iterations
method = "bootstrap"  # standard deviation estimation method

psi, psi_std = psi_est(mixed_data, segment_length; nboot=nboot, method=method)

p1 = heatmap(
    psi;
    ticks=false,
    yflip=true,
    yticks=([1, 2, 3, 4], ["Ch1", "Ch2", "Ch3", "Ch4"]),
    xticks=([1, 2, 3, 4], ["Ch1", "Ch2", "Ch3", "Ch4"]),
    color=cgrad(:bwr),
    title="Phase Slope Index",
)

p2 = heatmap(
    replace!(psi_std, NaN => 0);
    ticks=false,
    yflip=true,
    yticks=([1, 2, 3, 4], ["Ch1", "Ch2", "Ch3", "Ch4"]),
    xticks=([1, 2, 3, 4], ["Ch1", "Ch2", "Ch3", "Ch4"]),
    color=cgrad(:grays; rev=true),
    title="PSI standard deviation",
)

plot(p1, p2; layout=(1, 2), size=(720, 300))

segment_length = 100  # segment length
eplen = 200  # epoch length
method = "jackknife"  # standard deviation estimation method

# three frequency bands
freqlist = [[5:1:10;] [6:1:11;] [7:1:12;]]

segave = true  # average across CS segments
subave = true  # subtract average across CS segments
detrend = true  # performs a 0th-order detrend across raw segments
window = blackman  # blackman window function

psi, psi_std = psi_est(
    mixed_data,
    segment_length;
    subave=subave,
    segave=segave,
    detrend=detrend,
    freqlist=freqlist,
    eplen=eplen,
    method=method,
    window=blackman,
)

psi_normed = psi ./ (psi_std .+ eps())

p1 = heatmap(
    psi[:, :, 1];
    ticks=false,
    yflip=true,
    yticks=([1, 2, 3, 4], ["Ch1", "Ch2", "Ch3", "Ch4"]),
    xticks=([1, 2, 3, 4], ["Ch1", "Ch2", "Ch3", "Ch4"]),
    color=cgrad(:bwr),
    title="Phase Slope Index",
)

p2 = heatmap(
    psi_std[:, :, 1];
    ticks=false,
    yflip=true,
    yticks=([1, 2, 3, 4], ["Ch1", "Ch2", "Ch3", "Ch4"]),
    xticks=([1, 2, 3, 4], ["Ch1", "Ch2", "Ch3", "Ch4"]),
    color=cgrad(:grays; rev=true),
    title="PSI standard deviation",
)

p3 = heatmap(
    psi_normed[:, :, 1];
    ticks=false,
    yflip=true,
    yticks=([1, 2, 3, 4], ["Ch1", "Ch2", "Ch3", "Ch4"]),
    xticks=([1, 2, 3, 4], ["Ch1", "Ch2", "Ch3", "Ch4"]),
    color=cgrad(:bwr),
    title="Normalized PSI",
)

plot(p1, p2, p3; layout=(1, 3), size=(1110, 270))

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

