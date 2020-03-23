clc; clear all; close all;

%% Get user to identify bolts in training set

videoFileReader_train = vision.VideoFileReader('IMG_1208.mp4');
videoPlayer_train = vision.VideoPlayer();

num_samples = 16;

fig = uifigure;
btn = uibutton(fig,'state','Text','Next Image', 'Value', false);
num_bolt = zeros(1, num_samples);

for i = 1:num_samples
    for j = 1:80
        objectFrame1_1 = videoFileReader_train();
    end
    figure; imshow(objectFrame1_1);
    btn.Value = false;
    pause(5)
    while true
        drawnow()
        stop_state = get(btn, 'Value');
        if stop_state
            break
        end
        num_bolt(i) = num_bolt(i) + 1;
        objectRegion_test(:,num_bolt(i),i)=round(getPosition(imrect));
    end
end

%% Create training images of bolts from user input

v = VideoReader('IMG_1208.mp4');
bolt_train = zeros(720, 1280, 3, max(num_bolt), num_samples);

for i = 1:num_samples
    frames(:, :, :, i) = read(v, 80*i);
    for j = 1:num_bolt(i)
        object_Region = objectRegion_test(:,j,i);
        bolt_train(object_Region(2):object_Region(2)+object_Region(4),...
            object_Region(1):object_Region(1)+object_Region(3), :, j, i) = ...
            permute(frames(object_Region(1):object_Region(1)+object_Region(3),...
            720-object_Region(2)-object_Region(4):720-object_Region(2), :, i), [2 1 3]);
    end
end

total_bolts = 0;

for i = 1:num_samples
    for j = 1:num_bolt(i)
        object_Region = objectRegion_test(:,j,i);
        total_bolts = total_bolts + 1;
        bolt_images(:, :, :, total_bolts) = uint8(bolt_train(:, :, :, j, i));
        eval(sprintf('Bolt_%d = bolt_images(:, :, :, total_bolts);', total_bolts));
        eval(sprintf('Boltcropped_%d = Bolt_%d(object_Region(2):object_Region(2)+object_Region(4), object_Region(1):object_Region(1)+object_Region(3), :);', total_bolts, total_bolts));
        figure; imshow(eval(sprintf('Boltcropped_%d', total_bolts)));
        pause(0.5)    
    end
end
    
cd 'training'

for i = 1:total_bolts
    q1 = sprintf("Boltcropped_%d", i);
    q2 = sprintf("Boltcropped_%d.png", i);
    imwrite(eval(q1), q2);
end

%% Creating cropped images of same size

for i = 1:total_bolts
    name = sprintf("Boltcropped_%d", i);
    sz = size(eval(name));
    sizex(i) = sz(1); sizey(i) = sz(2);
end

minx = min(sizex); miny = min(sizey);

for i = 1:total_bolts
    name = sprintf("Boltcropped_%d", i);
    sz = size(eval(name));
    adjustx = sz(1) - minx;
    adjusty = sz(2) - miny;
    rx = rem(adjustx,2); qx = floor(adjustx/2);
    ry = rem(adjusty,2); qy = floor(adjusty/2);
    eval(sprintf('Boltadjust(:, :, :, i) = Boltcropped_%d(qx+rx+1:sz(1)-qx,qy+ry+1:sz(2)-qy , :);', i, i));   
end

%% SVD on adjusted bolts

close all

bolts_reshaped = reshape(Boltadjust, [minx*miny*3, total_bolts]);

[U, S, V] = svd(double(bolts_reshaped), 'econ');

figure;
subplot(2,1,1) 
plot(diag(S),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 80]) 
subplot(2,1,2) 
semilogy(diag(S),'ko','Linewidth',[2]) 
set(gca,'Fontsize',[14],'Xlim',[0 80])


figure;
for i = 1:total_bolts
    subplot(8,8,i)
    imshow(squeeze(uint8(Boltadjust(:, :, :, i))))
end


for j = 8
    figure;
    im_rank1 = U(:, 2:j)*S(2:j, 2:j)*V(:, 2:j)';
    im_rank1s = reshape(im_rank1, [minx, miny, 3, total_bolts]);
    for i = 1:total_bolts
        subplot(8,8,i)
        imshow(squeeze(uint8(im_rank1s(:, :, :, i))))
    end
end

figure;
for i = 1:8
    subplot(2, 4, i)
    Ut1 = reshape(U(:, i), minx, miny, 3);
    Ut2=Ut1(minx:-1:1,:); 
    pcolor(Ut2), shading interp, colormap hot
    set(gca,'Xtick',[],'Ytick',[])
end

%% Loading test video

video_test = VideoReader('IMG_1208.mp4');
video_test = read(video_test);
video_test = permute(video_test, [2 1 3 4]);

%% Detecting bolts

clear bolts_positions x_detected y_detected
close all; clc

size_test = size(video_test);
sz_testx = size_test(1);
sz_testy = size_test(2);

numwindows_x = floor(sz_testx/(0.5*minx));
numwindows_y = floor(sz_testy/(0.5*miny));

window_x = minx;
window_y = miny;

shift_x = floor(sz_testx/numwindows_x);
shift_y = floor(sz_testy/numwindows_y);


numframes = 12;

threshold = 2*10^4;

for i = 1:numframes
    frame = video_test(:, :, :, 80*i);
    bolts_detected = 0;
    for j = 0:numwindows_y-2
        for k = 0:numwindows_x-2
            window = frame(k*shift_x+1:k*shift_x+window_x, j*shift_y+1:j*shift_y+window_y, :);
            window_reshape = double(reshape(window, [minx*miny*3, 1]));
            projection = U(:, 1:4)'*window_reshape;
            projection_score = norm(projection);
            if projection_score > threshold
                bolts_detected = bolts_detected+1;
                x_detected = k*shift_x+1 + (k*shift_x+window_x)/2;
                y_detected = j*shift_y+1 + (j*shift_y+window_y)/2;
                bolts_positions(bolts_detected, 1, i) = x_detected;
                bolts_positions(bolts_detected, 2, i) = y_detected;
            end
            bolt_classifier(j+1, k+1, i) = projection_score(1);
        end
    end
    if bolts_detected == 0
        bolts_positions(1, 1, i) = 0;
        bolts_positions(1, 2, i) = 0;
    end
    %     bolt_classifier = projection_score(:, :, i);
    %     [M, I] = max(bolt_classifier,[],'all','linear');
    %     bolt_classifier_norm = bolt_classifier/M;
    %     [row,col] = ind2sub([numwindows_x-1 numwindows_y-1],I);
    %     bolts_positions(1,1,i) = shift_y*(row+0.5);
    %     bolts_positions(1,2,i) = shift_x*(col+0.5);
    figure;
    imshow(frame)
    hold on
    scatter(bolts_positions(:, 1, i), bolts_positions(:, 2, i), 'r*')
end

