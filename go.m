setup() ;
% setup('useGpu', true); % Uncomment to initialise with a GPU support
clc;
%% Part 3.1: Prepare the data

% Load a database of blurred images to train from
imdb = load('data/text_imdb.mat') ;

replace_images = true;

if replace_images
    load('data/patches.mat');

    train_num = 300;
    test_num = 130;

    imdb.images.id = 1:train_num+test_num;
    imdb.images.set = [ones(1, train_num), 2*ones(1, test_num)];

    images = data.patches(1:train_num+test_num, :, :);
    labels = data.labels(1:train_num+test_num, :, :);

    imdb.images.data = single(permute(images, [2, 3, 4, 1]));
    imdb.images.label = single(permute(labels, [2, 3, 4, 1]));
end

train = find(imdb.images.set == 1) ;
val = find(imdb.images.set == 2) ;


% % Visualize the first image in the database
figure(31) ; set(gcf, 'name', 'Data: Input & Output') ; clf ;

subplot(1,2,1) ; imagesc(imdb.images.data(:,:,:,val(1))) ;
axis off image ; title('Input (image)') ;

subplot(1,2,2) ; imagesc(imdb.images.label(:,:,:,val(1))) ;
axis off image ; title('Desired output (binary segmentation)') ;

colormap gray ;

%% Part 3.2: Create a network architecture
%
% The expected input size (a single 64 x 64 x 1 image patch). This is
% used for visualization purposes.

net = initializeSmallCNN() ;
% net = initializeLargeCNN() ;

% Display network
% vl_simplenn_display(net) ;


%% Part 3.3: learn the model

% Add a loss (using a custom layer)
net = addCustomLossLayer(net, @l2LossForward, @l2LossBackward) ;

% Train
trainOpts.expDir = 'data/text-small' ;
trainOpts.gpus = [] ;
trainOpts.batchSize = 16;
trainOpts.learningRate = 0.002 ;
trainOpts.plotDiagnostics = false ;
% trainOpts.plotDiagnostics = true ; % Uncomment to plot diagnostics
trainOpts.numEpochs = 5 ;
trainOpts.errorFunction = 'binary' ;

a = 1;

net = cnn_train(net, imdb, @getBatch, trainOpts) ;

% Deploy: remove loss
net.layers(end) = [] ;

%% Part 3.4: evaluate the model

% Evaluate network on an image
res = vl_simplenn(net, imdb.images.data(:,:,:,val(1))) ;

figure(32) ; clf ; colormap gray ;
set(gcf,'name', 'Initial Network') ;
subplot(1,2,1) ;
imagesc(res(1).x) ; axis image off  ;
title('CNN input') ;

set(gcf,'name', 'Part 3.2: network output') ;
subplot(1,2,2) ;
imagesc(round(res(end).x)) ; axis image off  ;
title('CNN output (not trained yet)') ;

% figure(33) ; set(gcf, 'name', 'Part 3.4: Results on the training set') ;
% showDeblurringResult(net, imdb, train(1:30:151)) ;
% 
% figure(34) ; set(gcf, 'name', 'Part 3.4: Results on the validation set') ;
% showDeblurringResult(net, imdb, val(1:30:151)) ;
% 
% figure(35) ;
% set(gcf, 'name', 'Part 3.4: Larger example on the validation set') ;
% colormap gray ;
% subplot(1,2,1) ; imagesc(imdb.examples.blurred{1}, [-1, 0]) ;
% axis image off ;
% title('CNN input') ;
% res = vl_simplenn(net, imdb.examples.blurred{1}) ;
% subplot(1,2,2) ; imagesc(res(end).x, [-1, 0]) ;
% axis image off ;
% title('CNN output') ;
