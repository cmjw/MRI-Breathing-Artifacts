% Load XCAT data
data = load("data/sampling_300ms_compressed.mat");
whos
disp(fieldnames(data)) 
N = 400; % phantom dimensions are 400x400
NUM_SLICES = 50; % slices per sample
NUM_SAMPLES = 100; 
% data.data2(:,:,slice,sample)

%% 
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

%%
figure; imagesc(sinogram); colormap("gray"); axis equal off;
title('Sinogram');

bad_recon = iradon(sinogram, angles, 'Linear', 'Ram-Lak', 1, N);
figure; imshow(bad_recon, []); 
title(['Direct Reconstruction (' num2str(frames) ' frames, slice=' num2str(slice) ')']);