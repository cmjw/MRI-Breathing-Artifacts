%% 
N = 256;                  
num_bins = 5; % = number of respiratory phases, try 5
num_breaths = 5;
expansion_level = 0.4;

num_spokes = 400; % try 400   
golden_angle = 111.246;   

% simulate expansion/compression during breathing
t = linspace(0, 1, num_spokes);
breathing_signal = sin(2 * pi * num_breaths * t); 

figure(10); plot(t, breathing_signal); title('Breathing Signal');

%% Generate data
% create a dynamic phantom that changes as the "patient" "breathes"

sample_p = zeros(N, N);
sample_projection = radon(sample_p, 0); % match the dimensions
len = length(sample_projection);

sinogram = zeros(len, num_spokes); 
angles = mod((0:num_spokes-1) * golden_angle, 180);

for i = 1:num_spokes
    current_scale = 1.0 + expansion_level * ((breathing_signal(i) + 1) / 2);
    
    p = phantom('Modified Shepp-Logan', N);
    p_scaled = imresize(p, current_scale, 'bilinear');
    
    % center = floor(size/2)
    % start = c = (n/2) + 1, end = c + (n/2)
    sz = size(p_scaled);
    p_final = zeros(N, N);
    r_idx = max(1, floor((sz(1)-N)/2)):min(sz(1), floor((sz(1)+N)/2)-1);
    c_idx = max(1, floor((sz(2)-N)/2)):min(sz(2), floor((sz(2)+N)/2)-1);
  
    p_final = p_scaled(floor(sz(1)/2)-N/2+1:floor(sz(1)/2)+N/2, ...
                       floor(sz(2)/2)-N/2+1:floor(sz(2)/2)+N/2);

    % can fix cropping to project onto a larger image than NxN
    
    % a single radial projection at the golden angle
    res = radon(p_final, angles(i));
    sinogram(:, i) = res;
end

figure(20); imagesc(sinogram); colormap("gray"); axis equal off;
title('Sinogram');

%% Binning 

% group spokes into bins based on the breathing signal amplitude

[~, bin_edges] = histcounts(breathing_signal, num_bins);
bin_indices = discretize(breathing_signal, bin_edges);

%% iradon method
reconstructed_bins_iradon = zeros(N, N, num_bins);

for b = 1:num_bins
    % Extract spokes belonging to a specific respiratory phase
    idx = (bin_indices == b);
    bin_sinogram = sinogram(:, idx);
    bin_angles = angles(idx);
    
    reconstructed_bins_iradon(:,:,b) = iradon(bin_sinogram, bin_angles, 'Linear', 'Ram-Lak', 1, N);
    % very undersampled
end

% average reconstruction (no binning)
average_image_iradon = iradon(sinogram, angles, 'Linear', 'Ram-Lak', 1, N);

figure(30);
subplot(1, 3, 1); imshow(average_image_iradon, []);
title('No Binning (iradon)');
subplot(1, 3, 2); imshow(reconstructed_bins_iradon(:,:,1), []);
title('Full Exhale (iradon)');
subplot(1, 3, 3); imshow(reconstructed_bins_iradon(:,:,num_bins), []);
title('Full Inhale (iradon)');

% can use iterative reconstruction to decrease streaking artifact

figure(31);
for i=1:num_bins
   subplot(1, num_bins, i);
   imshow(reconstructed_bins_iradon(:,:,i));
   title(['Bin ' num2str(i) ' (iradon)']);
end
%% 

% rudimentary denoising function just to test
% update later

function u = basic_tv_denoise(u, lambda, iters)
    dt = 0.2; % stability step size
    
    for i = 1:iters
        % fradients (forward differences)
        [ux, uy] = gradient(u);
        
        % magnitude of gradient
        mag = sqrt(ux.^2 + uy.^2 + 1e-6);
        
        % normalized gradient
        nx = ux ./ mag;
        ny = uy ./ mag;
        
        % divergence, where the noise it
        [nxx, ~] = gradient(nx);
        [~, nyy] = gradient(ny);
        div = nxx + nyy;
        
        u = u + dt * lambda * div;
    end
end
%% Total variation regularization

% parameters
num_iterations = 20;
lambda_tv = 0.005; % higher is more smoothed
step_size = 0.1; % alpha, learning rate
denoise_iterations = 5;

reconstructed_bins_tv = zeros(N, N, num_bins);
all_residuals = zeros(num_iterations, num_bins); % store error

for b = 1:num_bins
    % Extract spokes belonging to a specific respiratory phase
    idx = (bin_indices == b);
    bin_sinogram = sinogram(:, idx);
    bin_angles = angles(idx);

    % initial guess, with iradon
    u = reconstructed_bins_iradon(:,:,b);

    local_step = step_size / length(bin_angles);

    bin_norm = norm(bin_sinogram(:));
    
    for i = 1:num_iterations
        % forward project current guess (radon space)
        current_sinogram = radon(u, bin_angles);

        if size(current_sinogram, 1) ~= size(bin_sinogram, 1)
            error('Dimension mismatch, check scaling.');
        end
        error_sinogram = current_sinogram - bin_sinogram;

        % monitor convergence
        all_residuals(i, b) = norm(error_sinogram(:)) / bin_norm;

        % object space
        backproj_error = iradon(error_sinogram, bin_angles, 'none', 'Linear', 1, N);

        % data fidelity
        u = u - local_step * backproj_error;

        % TV regularization
        u = basic_tv_denoise(u, lambda_tv, denoise_iterations);

        % should be non-negative
        u(u<0) = 0;

        fprintf('Bin %2.d, Iter %2.d, Max Value: %.2f\n', b, i, max(u(:)));
    end

    reconstructed_bins_tv(:,:,b) = u;
end

figure(40);
for i=1:num_bins
   subplot(1, num_bins, i);
   imshow(reconstructed_bins_tv(:,:,i));
   title(['Bin ' num2str(i) ' (TV)']);
end

%% Convergence plots

figure(50);
plot(1:num_iterations, all_residuals, '-o', 'LineWidth', 2);
grid on;
xlabel('Iteration Number');
ylabel('Relative Residual (Data Mismatch)');
title('Convergence Plot of TV Reconstruction');
legend_labels = arrayfun(@(x) sprintf('Bin %d', x), 1:num_bins, 'UniformOutput', false);
legend(legend_labels);

%% TV Regularization 2

% rudimentary denoising function just to test
% update later

function u = basic_tv_denoise(u, lambda, iters)
    dt = 0.2; % stability step size
    
    for i = 1:iters
        % fradients (forward differences)
        [ux, uy] = gradient(u);
        
        % magnitude of gradient
        mag = sqrt(ux.^2 + uy.^2 + 1e-6);
        
        % normalized gradient
        nx = ux ./ mag;
        ny = uy ./ mag;
        
        % divergence, where the noise it
        [nxx, ~] = gradient(nx);
        [~, nyy] = gradient(ny);
        div = nxx + nyy;
        
        u = u + dt * lambda * div;
    end
end

% parameters
num_iterations = 20;
lambda_tv = 0.005; % higher is more smoothed
step_size = 0.1; % alpha, learning rate
denoise_iterations = 5;

reconstructed_bins_tv = zeros(N, N, bins);
all_residuals = zeros(num_iterations, bins); % store error

for b = 1:bins
    % Extract spokes belonging to a specific respiratory phase
    idx = (bin_indices == b);
    bin_sinogram = sinogram(:, idx);
    bin_angles = angles(idx);

    % initial guess, with iradon
    u = reconstructed_iradon(:,:,b);

    local_step = step_size / length(bin_angles);

    bin_norm = norm(bin_sinogram(:));
    
    for i = 1:num_iterations
        % forward project current guess (radon space)
        current_sinogram = radon(u, bin_angles);

        if size(current_sinogram, 1) ~= size(bin_sinogram, 1)
            error('Dimension mismatch, check scaling.');
        end
        error_sinogram = current_sinogram - bin_sinogram;

        % monitor convergence
        all_residuals(i, b) = norm(error_sinogram(:)) / bin_norm;

        % object space
        backproj_error = iradon(error_sinogram, bin_angles, 'none', 'Linear', 1, N);

        % data fidelity
        u = u - local_step * backproj_error;

        % TV regularization
        u = basic_tv_denoise(u, lambda_tv, denoise_iterations);

        % should be non-negative
        u(u<0) = 0;

        fprintf('Bin %2.d, Iter %2.d, Max Value: %.2f\n', b, i, max(u(:)));
    end

    reconstructed_bins_tv(:,:,b) = u;
end

figure;
for i=1:bins
   subplot(1, bins, i);
   imshow(reconstructed_bins_tv(:,:,i));
   title(['Bin ' num2str(i) ' (TV)']);
end

figure;
plot(1:num_iterations, all_residuals, '-o', 'LineWidth', 2);
grid on;
xlabel('Iteration Number');
ylabel('Relative Residual (Data Mismatch)');
title('Convergence Plot of TV Reconstruction');
legend_labels = arrayfun(@(x) sprintf('Bin %d', x), 1:bins, 'UniformOutput', false);
legend(legend_labels);