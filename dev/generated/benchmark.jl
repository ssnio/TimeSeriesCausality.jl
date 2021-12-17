using TimeSeriesCausality
using Distributions: MvNormal
using Plots: plot, bar

order = 1
n_samples = 1024  # number of data points per channel
segment_length = 128
epoch_length = 128
noise_scale = 0.25
plot_slice=128

# data generation
rand_data = randn(Float64, (n_samples + order))
signal = zeros(Float64, (n_samples, 2))
white_noise = noise_scale * randn(n_samples, 2)
signal[:, 1] = rand_data[order+1:end]
signal[:, 2] = rand_data[1:n_samples]
signal += white_noise

# Causal estimation
psi, psi_std = psi_est(signal, segment_length; method="jackknife", eplen=epoch_length)
granger_idx, err_std = granger_est(signal, segment_length; method="jackknife")

p1 = plot(
    1:plot_slice,
    signal[1:plot_slice, :];
    title="Unidirectional Flux with white noise",
    label=["Channel 1" "Channel 2"],
    linecolor=["red" "blue"],
    xlabel="Time [bins]",
    lw = 2,
);
p2 = bar(
    ["Granger" "PSI"],
    [psi[1, 2] granger_idx];
    title="Granger vs PSI",
    yerror=[psi_std[1, 2] err_std],
    legend=false
);
plot(p1, p2; layout=(1, 2), size=(800, 300))

# data generation
signal = zeros(Float64, (n_samples, 2))
signal[:, 1] = TimeSeriesCausality.pink_noise(n_samples, 1.0)
signal[:, 2] = TimeSeriesCausality.pink_noise(n_samples, 1.0)

# Causal estimation
psi, psi_std = psi_est(signal, segment_length; method="jackknife", eplen=epoch_length)
granger_idx, err_std = granger_est(signal, segment_length; method="jackknife")

p1 = plot(
    1:plot_slice,
    signal[1:plot_slice, :];
    title="Correlated noise",
    label=["Channel 1" "Channel 2"],
    linecolor=["red" "blue"],
    xlabel="Time [bins]",
    lw = 2,
);
p2 = bar(
    ["Granger" "PSI"],
    [psi[1, 2] granger_idx];
    title="Granger vs PSI",
    yerror=[psi_std[1, 2] err_std],
    legend=false
);
plot(p1, p2; layout=(1, 2), size=(800, 300))

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

