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

bins = 20;

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
    
    figure; subplot(1,2,1);
    imshow(reconstructed_iradon(:,:,b), []);
    title(['Bin ' num2str(b) ' (iradon)']);

    subplot(1,2,2);
    temp = ((b-1)*10 + 1) + floor((frames/bins)/2);
    f = mod(temp, frames);
    imshow(data.data2(:,:,slice,f), []);
    title('Approximate Ground Truth');
end

