%% Load XCAT data
data = load("data/sampling_300ms_compressed.mat");
whos
disp(fieldnames(data)) 
N = 400; % phantom dimensions are 400x400
NUM_SLICES = 50; % slices per sample
NUM_SAMPLES = 100; 
% data.data2(:,:,slice,sample)

%% Radial MRI
golden_angle = 111.246;
frames = 100; % change later
slice = 10;

tr = 0.004; % repetition time: time to acquire 1 spoke [s]
breaths = 2; 
cycle_time = 5; % time for one breath cycle [s]
spokes = (breaths * cycle_time) / tr; 

% sample projection to obtain sinogram dimensions
sample_phantom = zeros(N,N);
sample_projection = radon(sample_phantom, 0);

% using radial MRI sampling
angles = mod((0:spokes-1) * golden_angle, 180);
sinogram = zeros(length(sample_projection), spokes);

% 1 spoke per phase, probably unrealistic
for i=1:spokes
    idx = 1 + floor(mod((i-1)*frames/(cycle_time/tr), ...
        frames)); % 1-indexed
    %disp(num2str(idx));
    p = data.data2(:,:,slice,idx);
    sino = radon(p, angles(i));
    sinogram(:,i) = sino;
end

%% Display iradon with no method to reconstruct
figure; imagesc(sinogram); colormap("gray"); axis equal off;
title('Sinogram');

bad_recon = iradon(sinogram, angles, 'Linear', 'Ram-Lak', 1, N);
figure; imshow(bad_recon, []); 
title(['Direct Reconstruction (' num2str(frames) ' frames, slice=' num2str(slice) ')']);

%% Binning

% want to bin more informed by respiratory stage. perhaps uneven binning or
% explicitly defined binning per stage for clearer reconstruction.
% for now I have simply divided the whole set of frames (100) into even
% bins.

bins = 10;

% map each spoke to a phase, like in forward projection
spoke_indices = 1:spokes;
frame_assignments = 1 + floor(mod((spoke_indices-1)*frames/(cycle_time/tr), frames));
bin_indices = ceil(frame_assignments / (frames / bins));
%disp(bin_indices)

%% iradon

reconstructed_iradon = zeros(N, N, bins);

for b = 1:bins
    indices = (bin_indices == b);
    bin_sino = sinogram(:,indices);
    bin_angles = angles(indices);
    reconstructed_iradon(:,:,b) = iradon(bin_sino, bin_angles, 'Linear', 'Ram-Lak', 1, N);
    
    % display iradon vs GT
    figure; subplot(1,2,1);
    imshow(reconstructed_iradon(:,:,b), []);
    title(['Bin ' num2str(b) ' (iradon)']);
    subplot(1,2,2);
    temp = ((b-1)*10 + 1) + floor((frames/bins)/2);
    f = mod(temp, frames);
    imshow(data.data2(:,:,slice,f), []);
    title('Approximate Ground Truth');
end

%% Total Variation (TV) Regularization

num_iters = 30; 
lambda = 0.005 / length(bin_angles); % higher is smoother
alpha = 0.001; % step size / learning rate
tv_iters = 10;
epsilon = 1e-6; % to avoid division by 0

% halo at i=20, alpha = 0.1.

reconstructed_bins_tv = zeros(N, N, bins);
all_residuals = zeros(num_iters, bins);

for b=1:bins
    % extract spokes from bin b
    indices = (bin_indices == b);
    bin_sinogram = sinogram(:, indices);
    bin_angles = angles(indices);

    % initial guess with iradon
    x = reconstructed_iradon(:,:,b);
    x(x<0) = 0; % constrain non-negative

    bin_norm = norm(bin_sinogram(:));

    % forward project current guess
    for i=1:num_iters
        current_sinogram = radon(x, bin_angles);

        if size(current_sinogram, 1) ~= size(bin_sinogram, 1)
            error('Sinogram dimension mismatch.');
        end

        % error in radon space
        residual = current_sinogram - bin_sinogram;
        all_residuals(i,b) = norm(residual(:)) / bin_norm;
        
        % backward project to image domain
        backproj_error = iradon(residual, bin_angles, 'none', 'none', 1, N);

        x = x - lambda * backproj_error;

        % TV regularization / denoising
        for t=1:tv_iters
            [dfdx, dfdy] = gradient(x);

            magnitude = sqrt(dfdx.^2 + dfdy.^2 + epsilon);

            [d2fdx2, ~] = gradient(dfdx ./ magnitude);
            [~, d2fdy2] = gradient(dfdy ./ magnitude);
            tv_gradient = d2fdx2 + d2fdy2;

            x = x - alpha * tv_gradient;
        end

        x(x<0) = 0; % constrain again
    end

    % update step
    reconstructed_bins_tv(:,:,b) = x;

    figure; subplot(1,3,1);
    imshow(reconstructed_iradon(:,:,b), []);
    title(['Bin ' num2str(b) ' (iradon)']);
    subplot(1,3,2);
    temp = ((b-1)*10 + 1) + floor((frames/bins)/2);
    f = mod(temp, frames);
    imshow(data.data2(:,:,slice,f), []);
    title('Approximate Ground Truth');
    subplot(1,3,3); imshow(reconstructed_bins_tv(:,:,b), []);
    title(['Bin ' num2str(b) ' (TV)']);
end

% combine iterative reconstruction


%% 
figure;
plot(1:num_iters, all_residuals, '-o', 'LineWidth', 2);
grid on;
xlabel('Iteration Number');
ylabel('Relative Residual (Data Mismatch)');
title('Convergence Plot of TV Reconstruction');
legend_labels = arrayfun(@(x) sprintf('Bin %d', x), 1:bins, 'UniformOutput', false);
legend(legend_labels);

%% Future work

% XD-GRASP - use correlation between different time frames
% temporal TV
% pixels between phases are correlated
% temporal penalty term
% grain / streaks is removed, keeping the anatomy correlated between frames
% sharper
% 3D reconstruction