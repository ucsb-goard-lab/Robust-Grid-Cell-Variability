%% Plotting real vs. synthetic data
%-------------------------------------------------------------------------%
%   This script plots the decoding error done on the real data and compares
%   to that on the synthetic data
%
%   Written by WTR 03/18/2024 // Last updated by WTR 05/28/2024
%-------------------------------------------------------------------------%
%% Globals
nNeurons = 2.^[1:6];
arenaSize = [1.5, 1.5];
saveFlag = 1;

%% Loading results
parent_path = '/Users/redmawt1/Documents/Grid cell variability/Decoding analysis Github/';
real_results_path = strcat(parent_path, 'Real data/Results/');
synthetic_results_path = strcat(parent_path, 'Simulations/Results/');
save_path = strcat(parent_path, 'Real vs Simulation results/');

load(strcat(real_results_path, 'decoding_error_normalized_cell.mat'));
decoding_error_cell_norm = decoding_error;
load(strcat(synthetic_results_path, 'fixed_error_vs_pop_size_grid_spacing_0.85.mat'));
load(strcat(synthetic_results_path, 'variable_error_vs_pop_size_grid_spacing_0.85.mat'));

%% Generating random baseline
n_x = 22;
n_y = 22; 
bin_size = arenaSize ./ [n_x, n_y];

nSamples = 10;
nShuffles = 25;
random_error = zeros(1, nSamples);

for ii = 1:nSamples
    E = zeros(n_x, n_y, nShuffles);
    for jj = 1:nShuffles
        for xx = 1:n_x
            for yy = 1:n_y
                rand_x = randi(n_x);
                rand_y = randi(n_y);
                x_error = bin_size(1) * (xx - rand_x);
                y_error = bin_size(2) * (yy - rand_y);
                E(xx, yy, jj) = sqrt(sum(x_error^2 + y_error^2));
            end
        end
    end
    random_error(ii) = nanmean(nanmean(E, [3]), [1, 2]);
end


%% Plotting
figure
x = nNeurons; 
yFix = nanmean(fixError); yFix = yFix(1:length(nNeurons));
yVar = nanmean(varError); yVar = yVar(1:length(nNeurons));
yReal_cell_norm = nanmean(decoding_error_cell_norm'); yReal_cell_norm = yReal_cell_norm(1:length(nNeurons));
yRandom = nanmean(random_error);

semFix = nanstd(fixError); semFix = semFix(1:length(nNeurons)); % / sqrt(size(fixError, 2))
semVar = nanstd(varError); semVar = semVar(1:length(nNeurons)); %/ sqrt(size(varError, 2))
semReal_cell_norm = nanstd(decoding_error_cell_norm'); semReal_cell_norm = semReal_cell_norm(1:length(nNeurons)); % / sqrt(size(decoding_error_cell_norm, 2))

colorFixed = [0.8, 0.8, 0.8];
colorVar = [0.9, 0.4, 0.4];
colorReal_cell_norm = [0.4, 0.4, 0.9];

fill([x, fliplr(x)], [yFix + semFix , fliplr(yFix - semFix)], colorFixed, 'EdgeColor', colorFixed, 'FaceAlpha', 0.5); hold on
fill([x, fliplr(x)], [semVar + yVar, fliplr(yVar - semVar)], colorVar, 'EdgeColor', colorVar, 'FaceAlpha', 0.5); 
fill([x(1:(length(yReal_cell_norm))), fliplr(x(1:(length(yReal_cell_norm))))], [semReal_cell_norm + yReal_cell_norm, fliplr(yReal_cell_norm - semReal_cell_norm)], colorReal_cell_norm, 'EdgeColor', colorReal_cell_norm, 'FaceAlpha', 0.5); 

plot([x(1), x(end)], [yRandom, yRandom], 'k--');
plot(x, yFix, 'k-', 'LineWidth', 1.5);
plot(x, yVar, 'r-', 'LineWidth', 1.5); 
plot([x(1:(length(yReal_cell_norm)))], yReal_cell_norm(1:length(yReal_cell_norm)), 'b-', 'LineWidth', 1.5);

legend('Fixed ori. and spacing', 'Variable ori. and spacing', 'Real data');
xlabel('Neurons');
ylabel('Decoding error (m.)');
ylim([0.5, 0.9])
xlim([2, 64])
xticks(x)
xlabels = {};
for ii = 1:length(nNeurons)
    xlabels{ii} = num2str(nNeurons(ii));
end
xticklabels(xlabels)

if saveFlag == 1
    savefig(strcat(save_path, 'decoding_error_vs_pop_size_real_vs_synthetic.fig'));
end
