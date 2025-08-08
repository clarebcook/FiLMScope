%% Read in files
% replace this block with reading in your height map and image
% these should be 2D arrays of the same size
heightMap;
grayImage;

%% generate xy information
mag = 0.12; % magnification of the system 
downsample = 4; % downsample amount of the image/height map
pix_size = 1.1e-6; % pixel size

scale = pix_size * downsample / mag * 1e3;
[x_pix, y_pix] = meshgrid(1:size(heightMap, 2), 1:size(heightMap, 1));
x_pix = fliplr(x_pix);
y_mm = y_pix * scale;
x_mm = x_pix * scale;

%% show the mesh
fig = figure();

X = x_mm;
Y = y_mm;
Z = medfilt2(heightMap, [5, 5]);
values = grayImage;

% % uncomment this block to overlay height map and grayscale image
% cmap = turbo(256);
% % Normalize Z to [0, 1]
% z_normalized = (Z - min(Z(:))) / (max(Z(:)) - min(Z(:)));
% 
% cmap_idx = round(z_normalized * (size(cmap,1)-1)) + 1;
% cmap_rgb = ind2rgb(cmap_idx, cmap);  % Converts to MxNx3 RGB array
% gray_rgb = double(values)/255;          % Normalize to [0, 1]
% gray_rgb = repmat(gray_rgb, [1 1 3]);   % Replicate across RGB channels
% alpha = 0.8;                            % 0.5 = equal mix
% values = alpha*gray_rgb + (1-alpha)*cmap_rgb;

h = mesh(X, Y, Z, values);


daspect([1.6 1 1]);

set(gcf, "Color", "k");
set(gca, 'Color', 'k');
set(gca, 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');

% Freeze the tick locations so they don't change on interaction
xticks([0, 10, 20, 30])
xticks('manual');
yticks([0, 10, 20])
yticks('manual');
zticks([0, 4, 8, 12]);
zticks('manual')

shading interp;
