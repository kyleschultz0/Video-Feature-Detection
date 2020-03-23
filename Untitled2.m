%% Detecting bolts

size_test = size(video_test);
sz_testx = size_test(1);
sz_testy = size_test(2);

numwindows_x = floor(sz_testx/minx);
numwindows_y = floor(sz_testy/miny);

numframes = 10;

threshold = 2*10^4;

for i = 1:numframes
    frame = video_test(:, :, :, 80*i);
    bolts_detected = 0;
%     figure; imshow(frame)
    for j = 0:numwindows_y-1
        for k = 0:numwindows_x-1
            window = frame(720-(k+1)*minx:720-k*minx-1, j*miny+1:(j+1)*miny, :);
            window_reshape = double(reshape(window, [minx*miny*3, 1]));
            projection = U(:, 1:4)'*window_reshape;
            projection_score = norm(projection);
            if projection_score > threshold
                bolts_detected = bolts_detected+1;
                y_detected = 720-(k+1)*minx + ((720-k*minx-1)- (720-(k+1)*minx))/2;
                x_detected = j*miny+1 + (j+1)*miny/2;
                bolts_positions(bolts_detected, 1, i) = x_detected;
                bolts_positions(bolts_detected, 2, i) = y_detected;
            end
            bolt_classifier(j+1, k+1, i) = projection_score(1);
%             figure;
%             imshow(window)        
        end
    end
    figure;
    imshow(frame)
    hold on 
    scatter(bolts_positions(:, 1, i), bolts_positions(:, 2, i), 'r*')
    pause(1)
end
        