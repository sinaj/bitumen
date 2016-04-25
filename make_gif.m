filename = 'animation.gif';
for thr = 1:20
%     tmp_label = image;
%     tmp_label = tmp_label > thr;
%     imshow(tmp_label(end-199:end, end-199:end) * 255);
    drawnow;
    frame = getframe(1);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    if thr == 1
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
    else
        imwrite(imind,cm,filename,'gif','WriteMode','append', 'DelayTime', 0.01);
    end
    a = 1;
end