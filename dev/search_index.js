var documenterSearchIndex = {"docs":
[{"location":"generated/psi_examples/","page":"PSI Examples","title":"PSI Examples","text":"EditURL = \"https://github.com/ssnio/TimeSeriesCausality.jl/blob/master/docs/literate/psi_examples.jl\"","category":"page"},{"location":"generated/psi_examples/#Phase-Slope-Index","page":"PSI Examples","title":"Phase Slope Index","text":"","category":"section"},{"location":"generated/psi_examples/","page":"PSI Examples","title":"PSI Examples","text":"The notebook can be viewed here:","category":"page"},{"location":"generated/psi_examples/","page":"PSI Examples","title":"PSI Examples","text":"(Image: binder)\n(Image: nbviewer)","category":"page"},{"location":"generated/psi_examples/","page":"PSI Examples","title":"PSI Examples","text":"Robustly Estimating the Flow Direction of Information in Complex Physical Systems paper, by Guido Nolte, Andreas Ziehe, Vadim V. Nikulin, Alois Schlögl, Nicole Krämer, Tom Brismar, and Klaus-Robert Müller, (Nolte et al. 2008).","category":"page"},{"location":"generated/psi_examples/#Load-packages","page":"PSI Examples","title":"Load packages","text":"","category":"section"},{"location":"generated/psi_examples/","page":"PSI Examples","title":"PSI Examples","text":"using TimeSeriesCausality\nusing Plots: plot, heatmap, cgrad\nusing DSP: blackman","category":"page"},{"location":"generated/psi_examples/#Data","page":"PSI Examples","title":"Data","text":"","category":"section"},{"location":"generated/psi_examples/","page":"PSI Examples","title":"PSI Examples","text":"mixed_data contains four channels, where:","category":"page"},{"location":"generated/psi_examples/","page":"PSI Examples","title":"PSI Examples","text":"channels 1 and 2 are i.i.d. uniform 0 1 noise\nchannel 3 is delayed (by 1 sample) channel 1\nchannel 4 is delayed (by 16 samples) channel 1 plus i.i.d. uniform 0 02 noise","category":"page"},{"location":"generated/psi_examples/","page":"PSI Examples","title":"PSI Examples","text":"# data generation\nn_channels = 4  # number of channels\nn_samples = 2^16  # number of data points measured in each channel\nfs = 128  # sampling frequency\ntime_array = Array{Float64}(range(1; step=1 / fs, length=n_samples))\n\n# mixed data\nrange_c4 = 1:n_samples\nrange_c1 = range_c4 .+ 16\nrange_c3 = range_c1 .- 1\n\nrand_data = randn(Float64, (n_samples + 16, 1)) # uniform noise\ncause_source = rand_data[range_c1]  # channel 1\nrandom_source = randn(Float64, n_samples)  # channel 2, uniform noise\neffect_source = rand_data[range_c3]  # channel 3\nweak_effect = rand_data[range_c4] .- (randn(Float64, (n_samples, 1)) / 5) # channel 4\nmixed_data = hcat(cause_source, random_source, effect_source, weak_effect)\n\np1 = plot(\n    time_array[1:64],\n    mixed_data[1:64, :];\n    title=\"Mixed data\",\n    label=[\"Cause\" \"Random\" \"Effect\" \"Noisy Effect\"],\n    linecolor=[\"red\" \"green\" \"blue\" \"magenta\"],\n)\n\nplot(p1; layout=(1, 1), size=(800, 450))","category":"page"},{"location":"generated/psi_examples/#Example-1","page":"PSI Examples","title":"Example 1","text":"","category":"section"},{"location":"generated/psi_examples/","page":"PSI Examples","title":"PSI Examples","text":"PSI is calculated over all frequencies for segmented (segment_length = 100) but continuous data (single epoch, nep = 1) and estimation of error using Bootstrap method for 256 resampling iterations (nboot=256). The default window function (Hanning window) is used.","category":"page"},{"location":"generated/psi_examples/","page":"PSI Examples","title":"PSI Examples","text":"segment_length = 100  # segment length\nnboot = 256  # number of bootstrap iterations\nmethod = \"bootstrap\"  # standard deviation estimation method\n\npsi, psi_std = psi_est(mixed_data, segment_length; nboot=nboot, method=method)\n\np1 = heatmap(\n    psi;\n    ticks=false,\n    yflip=true,\n    yticks=([1, 2, 3, 4], [\"Ch1\", \"Ch2\", \"Ch3\", \"Ch4\"]),\n    xticks=([1, 2, 3, 4], [\"Ch1\", \"Ch2\", \"Ch3\", \"Ch4\"]),\n    color=cgrad(:bwr),\n    title=\"Phase Slope Index\",\n)\n\np2 = heatmap(\n    replace!(psi_std, NaN => 0);\n    ticks=false,\n    yflip=true,\n    yticks=([1, 2, 3, 4], [\"Ch1\", \"Ch2\", \"Ch3\", \"Ch4\"]),\n    xticks=([1, 2, 3, 4], [\"Ch1\", \"Ch2\", \"Ch3\", \"Ch4\"]),\n    color=cgrad(:grays; rev=true),\n    title=\"PSI standard deviation\",\n)\n\nplot(p1, p2; layout=(1, 2), size=(720, 300))","category":"page"},{"location":"generated/psi_examples/#Example-2","page":"PSI Examples","title":"Example 2","text":"","category":"section"},{"location":"generated/psi_examples/","page":"PSI Examples","title":"PSI Examples","text":"PSI is calculated over 3 frequency bands, for partitioned data to segments (segment_length = 100) and epochs (eplen = 200), estimation of error using Jackknife method (default). The window function is set to blackman (imported from DSP.jl). The plots are for only one of the frequency ranges.","category":"page"},{"location":"generated/psi_examples/","page":"PSI Examples","title":"PSI Examples","text":"We normalize the PSI by dividing it by estimated standard deviation.","category":"page"},{"location":"generated/psi_examples/","page":"PSI Examples","title":"PSI Examples","text":"segment_length = 100  # segment length\neplen = 200  # epoch length\nmethod = \"jackknife\"  # standard deviation estimation method\n\n# three frequency bands\nfreqlist = [[5:1:10;] [6:1:11;] [7:1:12;]]\n\nsegave = true  # average across CS segments\nsubave = true  # subtract average across CS segments\ndetrend = true  # performs a 0th-order detrend across raw segments\nwindow = blackman  # blackman window function\n\npsi, psi_std = psi_est(\n    mixed_data,\n    segment_length;\n    subave=subave,\n    segave=segave,\n    detrend=detrend,\n    freqlist=freqlist,\n    eplen=eplen,\n    method=method,\n    window=blackman,\n)\n\npsi_normed = psi ./ (psi_std .+ eps())\n\np1 = heatmap(\n    psi[:, :, 1];\n    ticks=false,\n    yflip=true,\n    yticks=([1, 2, 3, 4], [\"Ch1\", \"Ch2\", \"Ch3\", \"Ch4\"]),\n    xticks=([1, 2, 3, 4], [\"Ch1\", \"Ch2\", \"Ch3\", \"Ch4\"]),\n    color=cgrad(:bwr),\n    title=\"Phase Slope Index\",\n)\n\np2 = heatmap(\n    psi_std[:, :, 1];\n    ticks=false,\n    yflip=true,\n    yticks=([1, 2, 3, 4], [\"Ch1\", \"Ch2\", \"Ch3\", \"Ch4\"]),\n    xticks=([1, 2, 3, 4], [\"Ch1\", \"Ch2\", \"Ch3\", \"Ch4\"]),\n    color=cgrad(:grays; rev=true),\n    title=\"PSI standard deviation\",\n)\n\np3 = heatmap(\n    psi_normed[:, :, 1];\n    ticks=false,\n    yflip=true,\n    yticks=([1, 2, 3, 4], [\"Ch1\", \"Ch2\", \"Ch3\", \"Ch4\"]),\n    xticks=([1, 2, 3, 4], [\"Ch1\", \"Ch2\", \"Ch3\", \"Ch4\"]),\n    color=cgrad(:bwr),\n    title=\"Normalized PSI\",\n)\n\nplot(p1, p2, p3; layout=(1, 3), size=(1110, 270))","category":"page"},{"location":"generated/psi_examples/","page":"PSI Examples","title":"PSI Examples","text":"","category":"page"},{"location":"generated/psi_examples/","page":"PSI Examples","title":"PSI Examples","text":"This page was generated using Literate.jl.","category":"page"},{"location":"generated/granger_examples/","page":"Granger Examples","title":"Granger Examples","text":"EditURL = \"https://github.com/ssnio/TimeSeriesCausality.jl/blob/master/docs/literate/granger_examples.jl\"","category":"page"},{"location":"generated/granger_examples/#Granger-Causality-Estimation","page":"Granger Examples","title":"Granger Causality Estimation","text":"","category":"section"},{"location":"generated/granger_examples/","page":"Granger Examples","title":"Granger Examples","text":"The notebook can be viewed here:","category":"page"},{"location":"generated/granger_examples/","page":"Granger Examples","title":"Granger Examples","text":"(Image: binder)\n(Image: nbviewer)","category":"page"},{"location":"generated/granger_examples/#Load-packages","page":"Granger Examples","title":"Load packages","text":"","category":"section"},{"location":"generated/granger_examples/","page":"Granger Examples","title":"Granger Examples","text":"using TimeSeriesCausality\nusing Distributions: MvNormal\nusing Plots: plot\nusing Printf","category":"page"},{"location":"generated/granger_examples/#Data","page":"Granger Examples","title":"Data","text":"","category":"section"},{"location":"generated/granger_examples/","page":"Granger Examples","title":"Granger Examples","text":"We will generate (simulate) 1e6 two-channel samples using a design (evolution) matrix A of order 3. where:","category":"page"},{"location":"generated/granger_examples/","page":"Granger Examples","title":"Granger Examples","text":"channel 1 is the causal (independent) time series\nchannel 2 is the effect (dependent) time series","category":"page"},{"location":"generated/granger_examples/","page":"Granger Examples","title":"Granger Examples","text":"# design (evolution) matrix\n#            Channel 1->    Channel 2->\n#           t-1  t-2  t-3  t-1  t-2  t-3\ndesigner = [0.4 -0.6  0.8  0.0  0.0  0.0;  # -> Channel 1\n            0.5  0.9  0.0  0.0  0.0  0.7]  # -> Channel 2\n\n# data generation\nn_samples = 128*1024  # number of time steps\nsegment_length = 1024  # segment length\nnoise_cov = MvNormal([0.25 0.0; 0.0 0.64])  # uncorrelated Cov matrix of noise\nnoise = rand(noise_cov, n_samples)'  # sampling from the noise distribution\n\n# pre-allocation and initial values\nsignal = zeros(n_samples, 2)\nsignal[1:3, :] = rand(3, 2) + noise[1:3, :]\n\n# simulation\nfor t in 4:n_samples\n    signal[t, :] = designer * reshape(signal[t-3:t-1, :], :, 1) + noise[t, :]\nend\n\nplot(1:segment_length, [signal[1:segment_length, 1], signal[1:segment_length, 2]],\n     title = \"Signals\",\n     label = [\"Channel 1\" \"Channel 2\"],\n     xlabel = \"Time steps\",\n     ylabel = \"\",\n     lw = 2)","category":"page"},{"location":"generated/granger_examples/#Granger-method-of-causal-estimation","page":"Granger Examples","title":"Granger method of causal estimation","text":"","category":"section"},{"location":"generated/granger_examples/","page":"Granger Examples","title":"Granger Examples","text":"Granger method can estimate the causal index for a 2-channel signal. The larger the value, the stronger the causal dependence. A positive value indicates the flow of information from channel 1 to 2, and vice versa. We have also included JackKnife method for calculate the standard deviation of error for the estimation.","category":"page"},{"location":"generated/granger_examples/","page":"Granger Examples","title":"Granger Examples","text":"granger_idx, err_std = granger_est(signal, segment_length; order=3, method=\"jackknife\")\n@printf \"Granger causality index is %.3f with std error of %.3f\" granger_idx err_std","category":"page"},{"location":"generated/granger_examples/#Order-as-a-hyperparameter","page":"Granger Examples","title":"Order as a hyperparameter","text":"","category":"section"},{"location":"generated/granger_examples/","page":"Granger Examples","title":"Granger Examples","text":"Since the order (time step delay) in which the signals interact is unknown by us, we have implemented Akaike and Bayesian information criteria to estimate the most informative order.","category":"page"},{"location":"generated/granger_examples/","page":"Granger Examples","title":"Granger Examples","text":"# the range of orders to look into\norder_range = 1:7\n\n# Akaike information criterion\naic = granger_aic(signal, segment_length, order_range)\n\n# Bayesian information criterion\nbic = granger_bic(signal, segment_length, order_range)\n\nplot(order_range, [aic, bic],\n     title = \"Akaike vs Bayesian information criteria\",\n     label = [\"Akaike IC\" \"Bayesian IC\"],\n     xlabel = \"Order\",\n     ylabel = \"IC\",\n     lw = 2)","category":"page"},{"location":"generated/granger_examples/","page":"Granger Examples","title":"Granger Examples","text":"","category":"page"},{"location":"generated/granger_examples/","page":"Granger Examples","title":"Granger Examples","text":"This page was generated using Literate.jl.","category":"page"},{"location":"#TimeSeriesCausality.jl","page":"Home","title":"TimeSeriesCausality.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This is a Julia implementation of the Phase Slope Index and Granger Causality methods. Please refer to http://doc.ml.tu-berlin.de/causality for more information.","category":"page"},{"location":"#Outline","page":"Home","title":"Outline","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pages = [\"index.md\", \"generated/psi_examples.md\", \"generated/granger_examples.md\", \"generated/benchmark.md\"]","category":"page"},{"location":"#Functions","page":"Home","title":"Functions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Phase Slope Index (PSI) estimation psi_est:","category":"page"},{"location":"","page":"Home","title":"Home","text":"psi_est","category":"page"},{"location":"#TimeSeriesCausality.psi_est","page":"Home","title":"TimeSeriesCausality.psi_est","text":"psi_est(data, seglen ; segshift, eplen, freqlist, method,\n        nboot, segave, subave, detrend)\n\ncalculates phase slope index (PSI)\n\nArguments\n\ndata::AbstractArray: NxM array for N data points in M channels\nseglen::Integer: segment length (determines the frequency resolution). If defining freqlist, seglen must be the same as sampling frequency.\n\noptional arguments\n\nsegshift::Integer: number of bins by which neighboring segments are shifted (default is seglen / 2)\neplen::Integer: length of epochs (default is number of samples)\nfreqlist::AbstractArray: a UnitRange or 2D-Array where each column is a frequency band (default is full range). Note that the DC component (0th frequency of FFT) is discarded, and freqlist shall only include integer values.\nmethod::String: standard deviation estimation method (default is \"jackknife\")\nsubave::Bool: if true, subtract average across Cross Spectra segments (default is false)\nsegave::Bool: if true, average across Cross Spectra segments (default is true)\nnboot::Integer: number of bootstrap resampling iterations (default is 100)\ndetrend::Bool: if true, performs a 0th-order detrend across raw segments (default is false)\nwindow::Function: window function with interval length as sole necessary argument (default is Hanning)\nverbose::Bool: if true, warnings and info logs would be echoed. (default is false)\n\nReturns\n\npsi::AbstractArray: Phase Slope Index with shape (channel, channel, frequency bands)\npsi_std::AbstractArray: PSI estimated standard deviation with shape (channel, channel, frequency bands)\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"Granger Causality estimation granger_est:","category":"page"},{"location":"","page":"Home","title":"Home","text":"granger_est","category":"page"},{"location":"#TimeSeriesCausality.granger_est","page":"Home","title":"TimeSeriesCausality.granger_est","text":"granger_est(data, seglen; order, method, verbose)\n\nGranger's method of causal relation approximation\n\nArguments\n\ndata::AbstractArray: Nx2 array for N data points in 2 channels.\nseglen::Integer: segment length.\n\noptional arguments\n\nmethod::String: standard deviation estimation method (default is \"jackknife\")\norder::Int: Model order. Assumed time delay order of interest (default is n_samples / 2)\nverbose::Bool: if true, warnings and info logs would be echoed.\n\nReturns\n\nGrind::Float64: Granger causality index\nGrind_std::Float64: estimated standard deviation of error\n\nInternal variables:\n\nCovs: Concatenated covariance mats of different orders\nAcoef: Design (A) matrix coefficients\nPerr: Prediction Error\nΣ: Measures the accuracy of the autoregressive prediction\n\nReferences:\n\nC. W. J. Granger (1969). Investigating causal relations by econometric models and cross-spectral methods, Econometrica, 37, 424-438.\nM. Ding et al. (2006). Granger Causality: Basic Theory and Application to Neuroscience available online: https://arxiv.org/abs/q-bio/0608035\n\n\n\n\n\n","category":"function"},{"location":"#Citation","page":"Home","title":"Citation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Please cite the following paper if you use the PSI code in published work:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Nolte, G., Ziehe, A., Nikulin, V., Schlögl, A., Krämer, N., Brismar, T., & Müller, K.R. (2008), Robustly Estimating the Flow Direction of Information in Complex Physical Systems, Phys. Rev. Lett., 100, 234101. ","category":"page"},{"location":"#Acknowledgement","page":"Home","title":"Acknowledgement","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This work was funded by the German Federal Ministry of Education and Research (BMBF) in the project ALICE III under grant ref. 01IS18049B.","category":"page"},{"location":"generated/benchmark/","page":"Benchmark","title":"Benchmark","text":"EditURL = \"https://github.com/ssnio/TimeSeriesCausality.jl/blob/master/docs/literate/benchmark.jl\"","category":"page"},{"location":"generated/benchmark/#Granger-method-vs.-Phase-Slope-Index","page":"Benchmark","title":"Granger method vs. Phase-Slope-Index","text":"","category":"section"},{"location":"generated/benchmark/","page":"Benchmark","title":"Benchmark","text":"Here we attempt at reproducing the Figure 1 of Nolte et al. 2008","category":"page"},{"location":"generated/benchmark/","page":"Benchmark","title":"Benchmark","text":"The notebook can be viewed here:","category":"page"},{"location":"generated/benchmark/","page":"Benchmark","title":"Benchmark","text":"(Image: binder)\n(Image: nbviewer)","category":"page"},{"location":"generated/benchmark/","page":"Benchmark","title":"Benchmark","text":"Robustly Estimating the Flow Direction of Information in Complex Physical Systems paper, by Guido Nolte, Andreas Ziehe, Vadim V. Nikulin, Alois Schlögl, Nicole Krämer, Tom Brismar, and Klaus-Robert Müller, (Nolte et al. 2008).","category":"page"},{"location":"generated/benchmark/#Load-packages","page":"Benchmark","title":"Load packages","text":"","category":"section"},{"location":"generated/benchmark/","page":"Benchmark","title":"Benchmark","text":"using TimeSeriesCausality\nusing Distributions: MvNormal\nusing Plots: plot, bar\nusing Printf\nusing FFTW","category":"page"},{"location":"generated/benchmark/#Unidirectional-flux-with-white-noise","page":"Benchmark","title":"Unidirectional flux with white noise","text":"","category":"section"},{"location":"generated/benchmark/","page":"Benchmark","title":"Benchmark","text":"channel 1 is the causal (independent) time series\nchannel 2 is the effect (dependent) time series","category":"page"},{"location":"generated/benchmark/","page":"Benchmark","title":"Benchmark","text":"parameters","category":"page"},{"location":"generated/benchmark/","page":"Benchmark","title":"Benchmark","text":"order = 2\nn_samples = 16*128  # number of data points per channel\nsegment_length = 64\nepoch_length = 128\nnoise_scale = 0.5\nplot_slice=128\n\n# data generation\nrand_data = randn(Float64, (n_samples + order))\nsignal = zeros(Float64, (n_samples, 2))\nwhite_noise = noise_scale * randn(n_samples, 2)\nsignal[:, 1] = rand_data[order+1:end]\nsignal[:, 2] = rand_data[1:n_samples]\nsignal += white_noise","category":"page"},{"location":"generated/benchmark/","page":"Benchmark","title":"Benchmark","text":"Causal estimation","category":"page"},{"location":"generated/benchmark/","page":"Benchmark","title":"Benchmark","text":"psi, psi_std = psi_est(signal, segment_length; method=\"jackknife\", eplen=epoch_length)\ngranger_idx, err_std = granger_est(signal, segment_length; method=\"jackknife\")\n\np1 = plot(\n    1:plot_slice,\n    signal[1:plot_slice, :];\n    title=\"Unidirectional Flux with white noise\",\n    label=[\"Channel 1\" \"Channel 2\"],\n    linecolor=[\"red\" \"blue\"],\n    xlabel=\"Time [bins]\",\n    lw = 2,\n);\np2 = bar(\n    [\"Granger\" \"PSI\"],\n    [psi[1, 2] granger_idx];\n    title=\"Granger vs PSI\",\n    yerror=[psi_std[1, 2] err_std],\n    legend=false\n);\nplot(p1, p2; layout=(1, 2), size=(800, 300))","category":"page"},{"location":"generated/benchmark/#Pink-(correlated)-noise","page":"Benchmark","title":"Pink (correlated) noise","text":"","category":"section"},{"location":"generated/benchmark/","page":"Benchmark","title":"Benchmark","text":"# data generation\nsignal = zeros(Float64, (n_samples, 2))\nsignal[:, 1] = TimeSeriesCausality.pink_noise(n_samples, 1.0)\nsignal[:, 2] = TimeSeriesCausality.pink_noise(n_samples, 1.0)","category":"page"},{"location":"generated/benchmark/","page":"Benchmark","title":"Benchmark","text":"Causal estimation","category":"page"},{"location":"generated/benchmark/","page":"Benchmark","title":"Benchmark","text":"psi, psi_std = psi_est(signal, segment_length; method=\"jackknife\", eplen=epoch_length)\ngranger_idx, err_std = granger_est(signal, segment_length; method=\"jackknife\")\n\np1 = plot(\n    1:plot_slice,\n    signal[1:plot_slice, :];\n    title=\"Correlated noise\",\n    label=[\"Channel 1\" \"Channel 2\"],\n    linecolor=[\"red\" \"blue\"],\n    xlabel=\"Time [bins]\",\n    lw = 2,\n);\np2 = bar(\n    [\"Granger\" \"PSI\"],\n    [psi[1, 2] granger_idx];\n    title=\"Granger vs PSI\",\n    yerror=[psi_std[1, 2] err_std],\n    legend=false\n);\nplot(p1, p2; layout=(1, 2), size=(800, 300))","category":"page"},{"location":"generated/benchmark/","page":"Benchmark","title":"Benchmark","text":"","category":"page"},{"location":"generated/benchmark/","page":"Benchmark","title":"Benchmark","text":"This page was generated using Literate.jl.","category":"page"}]
}
