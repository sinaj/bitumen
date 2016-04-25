setup() ;

clc;
%% Part 1: Prepare the data

load_images = true;

if load_images
    load('data/patches.mat');
    
    patch_per_image = 966;
    
    train_num = patch_per_image * 4;
    test_num = patch_per_image;

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

% net = initializeSmallCNN() ;
net = initializeLargeCNN() ;


%% Part 3: learn the model

% Add a loss (using a custom layer)
net = addCustomLossLayer(net, @l2LossForward, @l2LossBackward) ;

% Train
trainOpts.expDir = 'data/epoches' ;
trainOpts.gpus = [] ;
trainOpts.batchSize = 16;
trainOpts.learningRate = 2e-3 ;
trainOpts.plotDiagnostics = false ;
trainOpts.numEpochs = 100 ;
trainOpts.errorFunction = 'binary' ;
trainOpts.regression = true;

net = cnn_train(net, imdb, @getBatch, trainOpts) ;

% Deploy: remove loss
net.layers(end) = [] ;

%% Part 4: evaluate the model

fprintf('Evaluating Test Set...\n');
% Evaluate network on an image
res = vl_simplenn(net, imdb.images.data(:,:,:,val)) ;

prediction = round(res(end).x);

label = imdb.images.label(:, :, :, val);

prediction = prediction(:);
label = label(:);

pos = prediction == 1;
neg = prediction == 0;

l_pos = label == 1;
l_neg = label == 0;

tp = sum(pos & l_pos);
tn = sum(neg & l_neg);
fp = sum(pos & l_neg);
fn = sum(neg & l_pos);

acc = (tp + tn) / (tp + tn + fp + fn);
precision = tp / (tp + fp);
recall = tp / (tp + fn);
f_score = 2 * precision * recall / (precision + recall);

fprintf('ACC: %.4f, PREC: %.4f, REC: %.4f, F_SCORE: %.4f\n', acc, precision, recall, f_score);

% figure(32) ; clf ; colormap gray ;
% set(gcf,'name', 'Initial Network') ;
% subplot(1,2,1) ;
% imagesc(res(1).x) ; axis image off  ;
% title('CNN input') ;
% 
% set(gcf,'name', 'Samlpe: Network Output') ;
% subplot(1,2,2) ;
% imagesc(round(res(end).x)) ; axis image off  ;
% title('CNN output (not trained yet)') ;


%% Create Image


% res = vl_simplenn(net, imdb.images.data(:,:,:,val)) ;


out = gather(res(end).x);

pred = round(out);

recreated_image = zeros(patch_size*max_x, patch_size*max_y);

ind = 1;
for x_ind = 1:max_x
    for y_ind = 1:max_y
        x_int = (x_ind-1) * patch_size + 1 : x_ind * patch_size;
        y_int = (y_ind-1) * patch_size + 1 : y_ind * patch_size;
        recreated_image(x_int, y_int) = squeeze(pred(:, :, 1, ind));
        ind = ind + 1;

    end
end

recreated_image = 1 - imfill(1-recreated_image, 'holes');
fprintf('done!\n');

figure(100), imshow(recreated_image);


