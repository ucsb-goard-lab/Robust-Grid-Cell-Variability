%% Decoding analysis - real data
%-------------------------------------------------------------------------%
%   This script computes the positional decoding error on real data (from
%   Garnder et al., 2022), using the same method that is used in the
%   syntethic data analysis.
%
%   Written by WTR 03/18/2024 // Last updated by WTR 05/28/2024
%-------------------------------------------------------------------------%
close all
clear all

%% Set random seed 
rng(1)

%% Globals
nSamples = 25;                                              % number of subpopulations to do the decoding analysis on           
nDecoding = 10;                                             % number of splits of the data   
nShuffles = 10;                                             % number of shuffles used for train and test data 
train_test_split = 0.5;                                     % percent of data used for train                  

arenaSize = [1.5, 1.5];                                     % size of arena - in meters       
nNeurons = [2.^(1:6), 82];                                  % size of subpopulations of grid cells to be tested

parent_path = '/Users/redmawt1/Documents/Grid cell variability/Decoding analysis/Real data/';
data_path = strcat(parent_path, 'Data/');
result_path = strcat(parent_path, 'Results/');
recording_id = 'r12'; 
norm = 'cell';                                              % normalize each ratemap or each cell

arenaResolution = 15;
n_x = floor(arenaResolution * arenaSize(1));                % number of bins in the x-direction
n_y = floor(arenaResolution * arenaSize(2));                % number of bins in the y-direction
bin_size = arenaSize ./ [n_x, n_y];

save_flag = 1; 

%% Loading data
load(strcat(data_path, recording_id, 'accepted_cells.mat'))
load(strcat(data_path, recording_id, 'spikes.mat'))
load(strcat(data_path, recording_id, 'x.mat'))
load(strcat(data_path, recording_id, 'y.mat'))

accepted_cell_ids = find(accepted_cells == 1);
spikes = spikes(accepted_cell_ids, :);
[N, T] = size(spikes);

%% Splitting data in nDecoding chunks
X = zeros(N, n_x, n_y, nDecoding);
split_bins = round(linspace(0, T, nDecoding + 1));
x_bins = linspace(-arenaSize(1)/2 , arenaSize(1)/2, n_x + 1);
y_bins = linspace(-arenaSize(2)/2 , arenaSize(2)/2, n_y + 1);

for ii = 1:nDecoding
    ii
    spikes_ii = spikes(:, (split_bins(ii) + 1):split_bins(ii + 1));
    x_ii = x((split_bins(ii) + 1):split_bins(ii + 1));
    y_ii = y((split_bins(ii) + 1):split_bins(ii + 1));

    X_ii = zeros(N, n_x, n_y);
    occupancy_ii = zeros(n_x, n_y);

    for jj = 1:length(x_ii)
        x_bin_id = find(x_ii(jj) > x_bins(1:(end-1)) & x_ii(jj) < x_bins(2:end));
        y_bin_id = find(y_ii(jj) > y_bins(1:(end-1)) & y_ii(jj) < y_bins(2:end));

        X_ii(:, x_bin_id, y_bin_id) =  X_ii(:, x_bin_id, y_bin_id) + spikes_ii(:, jj);
        occupancy_ii(x_bin_id, y_bin_id) = occupancy_ii(x_bin_id, y_bin_id) + 1;
    end

    for jj = 1:N
        X(jj, :, :, ii) = squeeze(X_ii(jj, :, :)) ./ occupancy_ii;
        if strcmp(norm, 'ratemap')
            X(jj, :, :, ii) = X(jj, :, :, ii) / max(X(jj, :, :, ii), [], 'all');
        end
    end

end

if strcmp(norm, 'cell')
    for ii = 1:N
        X(ii, :, :, :) = X(ii, :, :, :) / max(X(ii, :, :, :), [], 'all');
    end
end

%% Decoding 
error = zeros(length(nNeurons), nSamples);
n_tr = ceil(train_test_split * nDecoding); 

for nn = 1:length(nNeurons)
    for ss = 1:nSamples
        neuron_ids = randperm(N, nNeurons(nn)); % randsample(N, nNeurons(nn), true); %

        E = zeros(n_x, n_y, nShuffles);
        for ii = 1:nShuffles
            shuff_ids = randperm(nDecoding); 
            tr_ids = shuff_ids(1:n_tr);
            te_ids = shuff_ids((n_tr + 1):end);

            X_tr = nanmean(X(neuron_ids, :, :, tr_ids), 4);
            X_te = nanmean(X(neuron_ids, :, :, te_ids), 4); 

            if strcmp(norm, 'train_test')
                X_tr = X_tr ./ max(X_tr, [], [2, 3]);
                X_te = X_te ./ max(X_te, [], [2, 3]);
            end

            e = zeros(n_x, n_y);
            for jj = 1:n_x
                for kk = 1:n_y
                    if sum(X_te(:, jj, kk)) > 0
                        w = sum(X_te(:, jj, kk) .* X_tr, 1);
                        w = squeeze(w);
                        [jj_decode, kk_decode] = find(w == nanmax(w, [], 'all'));
                
                        if length(jj_decode) > 1
                            coin_flip = randi(length(jj_decode));
                            jj_decode = jj_decode(coin_flip);
                            kk_decode = kk_decode(coin_flip);
                        end
                
                        jj_error = bin_size(1) * (jj_decode - jj);
                        kk_error = bin_size(2) * (kk_decode - kk);
                        e(jj, kk) = sqrt(sum(jj_error^2 + kk_error^2));
                    end
                end     
            end
            E(:, :, ii) = e;
        end
        error(nn, ss) = nanmean(nanmean(E, [1, 2]));
    end
end

decoding_error = error;
mean(decoding_error, 2)

if save_flag == 1
    save(strcat(result_path, 'decoding_error_normalized_', norm, '.mat'), 'decoding_error')
end




