setup() ;
% setup('useGpu', true); % Uncomment to initialise with a GPU support
clear;
clc;
%% Part 3.1: Prepare the data

% Load a database of blurred images to train from
% imdb = load('data/text_imdb.mat') ;


load('data/thr_gt.mat');

patch_per_image = 966;

train_num = patch_per_image * 4;
test_num = patch_per_image;

imdb.images.id = 1:train_num+test_num;
imdb.images.set = [ones(1, train_num), 2*ones(1, test_num)];

images = data.patches(1:train_num+test_num, :, :);
labels = data.labels(1:train_num+test_num);

imdb.images.data = single(permute(images, [2, 3, 4, 1]));
imdb.images.label = single(permute(labels, [2, 1]));

train = find(imdb.images.set == 1) ;
val = find(imdb.images.set == 2) ;

%% Part 3.2: Create a network architecture

net = initializeRegCNN() ;
% net = initializeLargeCNN() ;


%% Part 3.3: learn the model

% Add a loss (using a custom layer)
net = addCustomLossLayer(net, @l2LossForward, @l2LossBackward) ;

% Train
trainOpts.expDir = 'data/epoches' ;
trainOpts.gpus = [] ;
trainOpts.batchSize = 16;
trainOpts.learningRate = 2e-10 ;
trainOpts.plotDiagnostics = false ;
trainOpts.numEpochs = 100 ;
trainOpts.errorFunction = 'binary' ;
trainOpts.regression = false;

a = 1;

net = cnn_train(net, imdb, @getBatch_reg, trainOpts) ;

% Deploy: remove loss
net.layers(end) = [] ;

%% Part 3.4: evaluate the model

% Evaluate network on an image
res = vl_simplenn(net, imdb.images.data(:,:,:,val(1))) ;

root_dir = 'data/';

% 'B0', 'B1', 'B2', 'B3', 'B4'
% names = {'M0', 'M1', 'M2', 'M3', 'M4'};
names = {'M0'};

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
    tic;
    fprintf('Image %d\n', im_ind);
    for x_ind = 1:max_x
        for y_ind = 1:max_y
%             fprintf('%d, %d\n', x_ind, y_ind);
            i = imdb.images.data(:, :, 1, val(ind));
            res = vl_simplenn(net, i);
            patches_label(ind) = gather(res(end).x);
            ind = ind + 1;
            
        end
    end
    fprintf('Image %s is done!\n', names{im_ind});
    [x, y] = meshgrid(32+(0:max_x-1)*64, 32+(0:max_y-1)*64);
    [xq, yq] = meshgrid(1:size(images{1}, 1), 1:size(images{1}, 2));
    l = patches_label(1:max_x*max_y);
    l = reshape(l, [max_y, max_x]);
    vq = interp2(x,y,l,xq,yq,'cubic');
    vq = vq';
    image = images{im_ind};
    p = image > vq;
    l = labels{im_ind};
    p = p(32:end-62, 32:end-82);
    l = l(32:end-62, 32:end-82);
    acc = mean(p(:) == l(:));
    pos = p(:) == 1;
    neg = p(:) == 0;
    gt_pos = l(:) == 1;
    gt_neg = l(:) == 0;
    tp = sum(pos & gt_pos);
    fp = sum(pos & gt_neg);
    tn = sum(neg & gt_neg);
    fn = sum(neg & gt_pos);
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    fprintf('ACC: %.4f, precision: %.4f recall: %.4f time: %dsf\n', acc, precision, recall, toc);
%     figure, imshow(p);
end

data.patches = patches;
data.labels = patches_label;

a = reshape(data.labels, [length(names), max_x, max_y]);
figure(10), surf(squeeze(a(1, :, :)) ./ 255);

[x, y] = meshgrid(32+(0:max_x-1)*64, 32+(0:max_y-1)*64);
[xq, yq] = meshgrid(1:20:size(images{1}, 1), 1:20:size(images{1}, 2));
l = patches_label(1:max_x*max_y);
l = reshape(l, [max_y, max_x]);
vq = interp2(x,y,l,xq,yq,'cubic');
figure(20), surf(xq,yq,vq);
