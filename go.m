setup() ;

clc;
%% Part 1: Prepare the data

load_images = true;

if load_images
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

%% Part 2: Create a network architecture

net = initializeSmallCNN() ;
% net = initializeLargeCNN() ;


%% Part 3: learn the model

% Add a loss (using a custom layer)
net = addCustomLossLayer(net, @l2LossForward, @l2LossBackward) ;

% Train
trainOpts.expDir = 'data/epoches' ;
trainOpts.gpus = [] ;
trainOpts.batchSize = 16;
trainOpts.learningRate = 0.002 ;
trainOpts.plotDiagnostics = false ;
trainOpts.numEpochs = 2 ;
trainOpts.errorFunction = 'binary' ;
trainOpts.regression = true;

net = cnn_train(net, imdb, @getBatch, trainOpts) ;

% Deploy: remove loss
net.layers(end) = [] ;

%% Part 4: evaluate the model

% Evaluate network on an image
res = vl_simplenn(net, imdb.images.data(:,:,:,val(1))) ;

figure(32) ; clf ; colormap gray ;
set(gcf,'name', 'Initial Network') ;
subplot(1,2,1) ;
imagesc(res(1).x) ; axis image off  ;
title('CNN input') ;

set(gcf,'name', 'Samlpe: Network Output') ;
subplot(1,2,2) ;
imagesc(round(res(end).x)) ; axis image off  ;
title('CNN output (not trained yet)') ;
