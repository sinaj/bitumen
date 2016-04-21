function y = l2LossForward(x,r)
r = reshape(r, size(x));
delta = x - r ;
y = sum(delta(:).^2) ;
y = y / (size(x,1) * size(x,2)) ;  % normalize by image size
