% Load XCAT data
data = load("data/sampling_300ms_compressed.mat");
whos
disp(fieldnames(data)) 
N = 400; % phantom dimensions are 400x400
NUM_SLICES = 50; % slices per sample
NUM_SAMPLES = 100; 
% data.data2(:,:,slice,sample)

%%
% Sample these frames (maybe a smaller subset - start with 10 of 100
for i=1:8 % for now let one breath be t=1 to 8
    figure; clf;
    imshow(data.data2(:,:,10,i), []);
end

%% 
golden_angle = 111.246;
num_breathing_phases = 100; % change later
slice = 10;

repetition_time = 0.004; % time to acquire 1 spoke [s]
breaths = 5; 
breath_cycle_time = 5; % time for one breath cycle [s]
spokes = (breaths * breath_cycle_time) / repetition_time; 

% sample projection to obtain sinogram dimensions
sample_phantom = zeros(N,N);
sample_projection = radon(sample_phantom, 0);

% using radial MRI sampling
angles = mod((0:num_spokes-1) * golden_angle, 180);
sinogram = zeros(length(sample_projection), num_spokes);

% 1 spoke per phase, probably unrealistic
for i=1:spokes
    %idx = mod(i, num_breathing_phases) + 1; % 1-indexed
    idx = 1 + mod(floor(spokes / (breath_cycle_time/num_breathing_phases)), num_breathing_phases);
    disp(num2str(idx));
    p = data.data2(:,:,1,idx);
    sino = radon(p, angles(i));
    sinogram(:,i) = sino;
end

%%
figure; imagesc(sinogram); colormap("gray"); axis equal off;
title('Sinogram');

bad_recon = iradon(sinogram, angles, 'Linear', 'Ram-Lak', 1, 400);
figure; imshow(bad_recon, []); 
title(['Direct Reconstruction (' num2str(num_breathing_phases) ' frames, slice=' num2str(slice) ')']);