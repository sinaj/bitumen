clear;
clc;

root_dir = 'data/';
patch_dir = 'patches/';

names = {'B0', 'B1', 'B2', 'B3', 'B4', 'M0', 'M1', 'M2', 'M3', 'M4'};

images = {};
labels = {};

fprintf('Loading...');
for i = 1:length(names)
    image = imread(strcat(root_dir, names{i}, '.jpg'));
    images{end+1} = double(rgb2gray(image)) ./ 255;
    label = imread(strcat(root_dir, names{i}, '_label.png'));
    labels{end+1} = uint8((label ./ 255));
end
fprintf('done!\n');

patch_size = 200;

dim1 = size(images{end}, 1);
dim2 = size(images{end}, 2);

max_x = floor(dim1 / patch_size);
max_y = floor(dim2 / patch_size);

patch_num = max_x * max_y * length(names);

patches = zeros(patch_num, patch_size, patch_size);
patches_label = zeros(patch_num, patch_size, patch_size);

fprintf('Extracting...\n');
ind = 1;
for im_ind = 1:length(names)
    for x_ind = 1:max_x
        for y_ind = 1:max_y
            x_int = (x_ind-1) * patch_size + 1 : x_ind * patch_size;
            y_int = (y_ind-1) * patch_size + 1 : y_ind * patch_size;
            image = images{im_ind};
            label = labels{im_ind};
            patches(ind, :, :) = image(x_int, y_int);
            patches_label(ind, :, :) = label(x_int,y_int) == 1;
            ind = ind + 1;
            
        end
    end
    fprintf('Image %s is done!\n', names{im_ind});
end

fprintf('Saving...');
data.patches = patches;
data.labels = patches_label;

save('data/patches.mat', 'data');
fprintf('done!\n');
