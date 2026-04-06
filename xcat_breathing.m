% Load XCAT samples
data = load("data/sampling_300ms_compressed.mat");
whos
disp(fieldnames(data)) 

%%
% Sample these frames (maybe a smaller subset - start with 10 of 100
for i=1:8 % for now let one breath be t=1 to 8
    figure; clf;
    imshow(data.data2(:,:,10,i), []);
end

%% 
num_breaths = 5;
num_breathing_phases = 100; % change later
num_spokes = 200;
golden_angle = 111.246;
slice = 10;

% should num spokes necessarily equal time for all breaths?
% since not creating a breathing signal, no need for num breaths
% should depend now on num spokes

% sample projection
sample_phantom = zeros(400,400);
sample_projection = radon(sample_phantom, 0);

angles = mod((0:num_spokes-1) * golden_angle, 180);
%disp(['Num angles:' length(angles)]);
sinogram = zeros(length(sample_projection), num_spokes);

% 1 spoke per phase, probably unrealistic
for i=1:num_spokes
    idx = mod(i, num_breathing_phases) + 1; % 1-indexed
    %disp(['Index ' num2str(idx)]);
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