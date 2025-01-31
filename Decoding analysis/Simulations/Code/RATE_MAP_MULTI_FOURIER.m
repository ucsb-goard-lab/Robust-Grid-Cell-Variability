function [rateMap, spacingVec, orientationVec, phaseVec] = RATE_MAP_MULTI_FOURIER(gridFiring_max_mean, gridFiring_max_std, gridSpacing, ...
    gridSpacing_variability, arenaSize, gridOrientation, gridOrientation_variability, nNeurons)
%-------------------------------------------------------------------------%
%   This function generate the synthetic (idealized) rate maps via the sum
%   of three two-dimensional sinusoid gratings (as Solstad et al., 2006).
%   Grid orientations and spacings are sampled randomly, according to the
%   variability defined by gridSpacing_variability and
%   gridOrientation_variability. Grid phase is uniformly sampled. 
%
%   Written by WTR 06/05/2023 // Last updated by WTR 01/17/2025
%-------------------------------------------------------------------------%
%% Globals
arenaResolution = 15; % bins per meter
arenaX = linspace(-(arenaSize(1)/2), (arenaSize(1)/2), arenaSize(1) * arenaResolution);
arenaY = linspace(-(arenaSize(2)/2), (arenaSize(2)/2), arenaSize(2) * arenaResolution);

if length(gridSpacing) ~= length(gridOrientation) || length(gridSpacing_variability) ~= length(gridOrientation_variability)
    error("Number of grid modules is not equal for spacing and orientation");
end
n_modules = length(gridSpacing);

%% Ratemaps
rateMap = zeros(length(arenaX), length(arenaY), nNeurons); 
spacingVec = zeros(1, nNeurons);
orientationVec = zeros(1, nNeurons);
phaseVec = zeros(2, nNeurons);

for nn = 1:nNeurons
    grid_module_id = ceil(nn * n_modules / nNeurons);

    orientation = normrnd(gridOrientation(grid_module_id), gridOrientation_variability(grid_module_id)) * pi / 180;
    spacing = normrnd(gridSpacing(grid_module_id), gridSpacing_variability(grid_module_id));
    phase = rand(1, 2) .* arenaSize;

    orientationVec(nn) = orientation;
    spacingVec(nn) = spacing;
    phaseVec(:, nn) = phase;
    
    gridFiring_max = normrnd(gridFiring_max_mean, gridFiring_max_std);
    if gridFiring_max < 2
        gridFiring_max = 2;
    elseif gridFiring_max > 30
        gridFiring_max = 30; 
    end
    
    for xx = 1:length(arenaX)
        for yy = 1:length(arenaY)
            k = 4 * pi / (sqrt(3) * spacing);
            k1 = k / sqrt(2) * [cos(orientation + pi / 12) + sin(orientation + pi / 12), cos(orientation + pi / 12) - sin(orientation + pi / 12)]; 
            k2 = k / sqrt(2) * [cos(orientation + 5 * pi / 12) + sin(orientation + 5 * pi / 12), cos(orientation + 5 * pi / 12) - sin(orientation + 5 * pi / 12)];
            k3 = k / sqrt(2) * [cos(orientation + 3 * pi / 4) + sin(orientation + 3 * pi / 4), cos(orientation + 3 * pi / 4) - sin(orientation + 3 * pi / 4)]; 
            waveVectors = [k1; k2; k3];
            rateMap(xx, yy, nn) = gridFiring_max * 2/3 * (1/3 * sum(cos(waveVectors * [arenaX(xx) + phase(1); arenaY(yy) + phase(2)])) + 1/2);
        end
    end

end