using TimeSeriesCausality
using Distributions: MvNormal
using Plots: plot
using Printf

# design (evolution) matrix
#            Channel 1->    Channel 2->
#           t-1  t-2  t-3  t-1  t-2  t-3
designer = [0.4 -0.6  0.8  0.0  0.0  0.0;  # -> Channel 1
            0.5  0.9  0.0  0.0  0.0  0.7]  # -> Channel 2

# data generation
time_steps = 1000000  # number of time steps
segment_length = 1000  # segment length
noise_cov = MvNormal([0.25 0.0; 0.0 0.64])  # uncorrelated Cov matrix of noise
noise = rand(noise_cov, time_steps)'  # sampling from the noise distribution

# pre-allocation and initial values
signal = zeros(time_steps, 2)
signal[1:3, :] = rand(3, 2) + noise[1:3, :]

# simulation
for t in 4:time_steps
    signal[t, :] = designer * reshape(signal[t-3:t-1, :], :, 1) + noise[t, :]
end

plot(1:segment_length, [signal[1:segment_length, 1], signal[1:segment_length, 2]],
     title = "Signals",
     label = ["Channel 1" "Channel 2"],
     xlabel = "Time steps",
     ylabel = "",
     lw = 2)

grager_idx, err_std = granger_est(signal, 3, segment_length, "jackknife")
@printf "Granger causality index is %.3f with std error of %.3f" grager_idx err_std

# the range of orders to look into
order_range = 1:7

# Akaike information criterion
aic = granger_aic(signal, order_range, segment_length)

# Bayesian information criterion
bic = granger_bic(signal, order_range, segment_length)

plot(order_range, [aic, bic],
     title = "Akaike vs Bayesian information critera",
     label = ["Akaike IC" "Bayesian IC"],
     xlabel = "Order",
     ylabel = "IC",
     lw = 2)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

