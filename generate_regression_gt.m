clear;
clc;

root_dir = 'data/';
patch_dir = 'patches/';

draw_img = false;

% 'B0', 'B1', 'B2', 'B3', 'B4'
% names = {'M0', 'M1', 'M2', 'M3', 'M4'};
names = {'B0'};

images = {};
labels = {};

fprintf('Loading...');
for i = 1:length(names)
    image = imread(strcat(root_dir, names{i}, '.jpg'));
    images{end+1} = double(rgb2gray(image));
    label = imread(strcat(root_dir, names{i}, '_label.png'));
    labels{end+1} = logical((label) ./ 255);
end
fprintf('done!\n');

% patch_size = 64;
patch_size = 64;

dim1 = size(images{end}, 1);
dim2 = size(images{end}, 2);

max_x = floor(dim1 / patch_size);
max_y = floor(dim2 / patch_size);

patch_num = max_x * max_y * length(names);

patches = zeros(patch_num, patch_size, patch_size);
patches_label = zeros(patch_num, 1);

fprintf('Extracting...\n');
ind = 1;
for im_ind = 1:length(names)
    fprintf('Image %d\n', im_ind);
    for x_ind = 1:max_x
        for y_ind = 1:max_y
            fprintf('%d, %d\n', x_ind, y_ind);
            x_int = (x_ind-1) * patch_size + 1 : x_ind * patch_size;
            y_int = (y_ind-1) * patch_size + 1 : y_ind * patch_size;
            image = images{im_ind};
            label = labels{im_ind};
            patches(ind, :, :) = image(x_int, y_int);
            acc_vec = zeros(255, 1);
            filename = 'fig.gif';
            image = image(x_int, y_int);
            label = label(x_int, y_int);
            for thr = 1:1:255
%                 fprintf('%d, ', thr);
                tmp_label = image;
                tmp_label = tmp_label > thr;
                if draw_img
                    imshow(tmp_label(end-80:end, end-60:end) * 255);
                    drawnow;
                    frame = getframe(1);
                    im = frame2im(frame);
                    [imind, cm] = rgb2ind(im, 256);
                    if thr == 1
                        imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
                    else
                        imwrite(imind,cm,filename,'gif','WriteMode','append', 'DelayTime', 0.01);
                    end
                end
                acc_vec(thr) = mean(tmp_label(:) == label(:));
            end
            [~, patches_label(ind)] = max(acc_vec);
            ind = ind + 1;
%             fprintf('\n');
        end
    end
    fprintf('Image %s is done!\n', names{im_ind});
end

fprintf('Saving...');
data.patches = patches;
data.labels = patches_label;

save('data/thr_gt.mat', 'data');
fprintf('done!\n');

a = reshape(data.labels, [length(names), max_x, max_y]);
surf(squeeze(a(1, :, :)) ./ 255);

[x, y] = meshgrid(32+(0:max_x-1)*64, 32+(0:max_y-1)*64);
[xq, yq] = meshgrid(1:20:size(images{1}, 1), 1:20:size(images{1}, 2));
vq = interp2(x,y,l,xq,yq,'cubic');
surf(xq,yq,vq);
