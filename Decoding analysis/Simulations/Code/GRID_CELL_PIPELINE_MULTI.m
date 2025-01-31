%% GRID CELL PIPELINE MULTI
%-------------------------------------------------------------------------%
%   This script generates a population of grid cells, with varying
%   properties, and computes the error in reconstructing the position
%   represented by the activity of such a population. 
%
%   Written by WTR 01/19/2022 // Last updated by WTR 01/20/2025
%-------------------------------------------------------------------------%
%% Set random seed 
rng(1)

%% Globals      
gridSpacing = [0.85];             % mean spacing of grid cells - in meters    
gridOrientation = [6];            % mean orientation of grid cells - in degrees
arenaSize = [1.5, 1.5];           % size of arena - in meters       
nNeurons = 2.^6; %(1:10);             % size of populations of grid cells to be tested
nSamples = 25;                    % number of independent populations to test per size
nDecoding = 10;                   % number of random spiking vectors to draw per position
gridFiring_max_mean = 13;         % mean maximum firing rate for idealized model (Poisson neurons can fire more)
gridFiring_max_std = 8;           % mean maximum firing rate for idealized model (Poisson neurons can fire more)
oriStd = [1];                  % standard deviation used for sampling grid orientation
spacingStd = [0.05];        % standard deviation used for sampling grid spacing

saveFlag = 1;                     % whether or not to save all plots and matrices
savePath = 'Results/';

%% Role of size of grid population on decoding accuracy
varOriError = zeros(nSamples, length(nNeurons)); 
varSpacingError = zeros(nSamples, length(nNeurons)); 
varError = zeros(nSamples, length(nNeurons));
fixError = zeros(nSamples, length(nNeurons)); 

for ii = 1:nSamples
    ii
    for nn = 1:length(nNeurons)
        nn
        [rateMap_Fixed, spacing_Fixed, orientation_Fixed, phase_Fixed] = ...
            RATE_MAP_MULTI_FOURIER(gridFiring_max_mean, gridFiring_max_std, gridSpacing, zeros(1, length(gridSpacing)), arenaSize, gridOrientation, zeros(1, length(gridOrientation)), nNeurons(nn));
        [eFix] = MLE_DECODING(rateMap_Fixed, nDecoding, arenaSize);
        
        [rateMap_Variable, spacing_Variable, orientation_Variable, phase_Variable] = ...
            RATE_MAP_MULTI_FOURIER(gridFiring_max_mean, gridFiring_max_std, gridSpacing, spacingStd, arenaSize, gridOrientation, oriStd, nNeurons(nn));
        [eVar] = MLE_DECODING(rateMap_Variable, nDecoding, arenaSize);

        fixError(ii, nn) = mean(eFix);  
        varError(ii, nn) = mean(eVar);

    end
end

% Saving data
spacing_tag = '';
for ii = 1:length(gridSpacing)
    spacing_tag = strcat(spacing_tag, '_', num2str(gridSpacing(ii)));
end

if saveFlag == 1
    save(strcat(savePath, '/fixed_error_vs_pop_size_grid_spacing', spacing_tag, '.mat'), "fixError");
    save(strcat(savePath, '/variable_error_vs_pop_size_grid_spacing', spacing_tag, '.mat'), "varError");
end

% Generating random baseline
n_x = 22;
n_y = 22; 
bin_size = arenaSize ./ [n_x, n_y];
nShuffles = 10;

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

% Plotting
figure
x = nNeurons; 
yFix = nanmean(fixError);
yVar = nanmean(varError);
yRandom = nanmean(random_error);

stdFixed = nanstd(fixError);
stdVar = nanstd(varError);      

colorFixed = [0.8, 0.8, 0.8];
colorVar = [0.4, 0.4, 0.9];

fill([x, fliplr(x)], [yFix + stdFixed , fliplr(yFix - stdFixed)], colorFixed, 'EdgeColor', colorFixed, 'FaceAlpha', 0.5); hold on
fill([x, fliplr(x)], [stdVar + yVar, fliplr(yVar - stdVar)], colorVar, 'EdgeColor', colorVar, 'FaceAlpha', 0.5); 

plot(x, yFix, 'k-', 'LineWidth', 1.5);
plot(x, yVar, 'b-', 'LineWidth', 1.5); 
plot([x(1), x(end)], [yRandom, yRandom], 'k--', 'LineWidth', 1.5)

legend('Fixed ori. and spacing', 'Variable ori. and spacing');
xlabel('Neurons');
ylabel('Decoding error');
ylim([0, 1.0])
xlim([2, 1024])
xticks(x)
xlabels = {};
for ii = 1:length(nNeurons)
    xlabels{ii} = num2str(nNeurons(ii));
end
xticklabels(xlabels)

if saveFlag == 1
    savefig(strcat(savePath, '/decoding_error_vs_pop_size_grid_spacing', spacing_tag, '.fig'));
end

%% Role of variability in grid orientation and spacing
gridOrientation_variability = 0:1:5; 
gridSpacing_variability = 0:0.01:0.05;
errorVariable = zeros(nSamples, length(gridOrientation_variability), length(gridSpacing_variability));

for ii = 1:nSamples
    ii
    for oo = 1:length(gridOrientation_variability)
        oo
        for rr = 1:length(gridSpacing_variability)
            [rateMap_Variable] = RATE_MAP_MULTI_FOURIER(gridFiring_max_mean, gridFiring_max_std, gridSpacing, gridSpacing_variability(rr), arenaSize, gridOrientation, gridOrientation_variability(oo), nNeurons(end));
            [e] = MLE_DECODING(rateMap_Variable, nDecoding, arenaSize);

            errorVariable(ii, oo, rr) = mean(e);

        end
    end
end

% Saving data
if saveFlag == 1
    save(strcat(savePath, '/variable_spacing_orientation_grid_spacing', num2str(gridSpacing), '.mat'), "errorVariable");
end

% Plotting
figure
imagesc(squeeze(mean(errorVariable, 1)))
xlabels = {};
for ii = 1:length(gridSpacing_variability)
    xlabels{ii} = num2str(gridSpacing_variability(ii));
end
xticklabels(xlabels)
xlabel('\sigma_\theta (meters)')
ylabels = {};
for ii = 1:length(gridOrientation_variability)
    ylabels{ii} = num2str(gridOrientation_variability(ii));
end
yticklabels(ylabels)
ylabel('\sigma_\lambda (degrees)')
clim([0, 0.7])
if saveFlag == 1
    savefig(strcat(savePath, '/variable_spacing_orientation_grid_spacing', num2str(gridSpacing), '.fig'));
end


