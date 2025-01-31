function [ E ] = MLE_DECODING(rateMap, n_s, arena_size)
%-------------------------------------------------------------------------%
%   This function generates random population vectors, drawn from a Poisson
%   distribution, and performs MLE decoding in a cross-validated manner.
%   Decoding is performed on a 50-50 train-test split.
%
%   Written by WTR 04/25/2023 // Last updated by WTR 05/28/2024
%-------------------------------------------------------------------------%
%% Constructing "observed" Poisson activity
[n_x, n_y, n_N] = size(rateMap); 

X = zeros(n_x, n_y, n_N, n_s); 
for ii = 1:n_N
    for jj = 1:n_s
        X(:, :, ii, jj) = poissrnd(rateMap(:, :, ii)); 
    end
end

%% Decoding using leave one out
n_tr = n_s / 2; 
E = nan(n_x, n_y, n_s);
bin_size = arena_size ./ [n_x, n_y]; 

for ii = 1:n_s
    shuff_ids = randperm(n_s); 
    tr_ids = shuff_ids(1:n_tr);
    te_ids = shuff_ids((n_tr + 1):end);

    X_tr = mean(X(:, :, :, tr_ids), 4); 
    X_te = mean(X(:, :, :, te_ids), 4); 
    
    e = zeros(n_x, n_y);
    for jj = 1:n_x
        for kk = 1:n_y
            if sum(X_te(jj, kk, :)) > 0
                w = sum(X_te(jj, kk, :) .* X_tr, 3);
                [jj_decode, kk_decode] = find(w == max(w, [], 'all'));
                
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

E = nanmean(E, [1, 2]);

%% Random baseline 
% E_rand = nan(n_x, n_y, n_s);
% Z_rand = sqrt(sum(n_x^2 + n_y^2));
% 
% for ii = 1:n_s
%     e = zeros(n_x, n_y);
%     for jj = 1:n_x
%         for kk = 1:n_y
%             rand_x = randi(n_x, 1);
%             rand_y = randi(n_y, 1);
%                 
%             e(jj, kk) = sqrt(sum((rand_x - jj)^2 + (rand_y - kk)^2)) / Z_rand;
%         end
%     end
%     E_rand(:, :, ii) = e;
%     
% end
% 
% E_rand = nanmean(E_rand, [1, 2]);
