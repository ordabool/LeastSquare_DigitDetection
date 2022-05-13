%% ====================== Prepare New Test Set ======================
load('mnist.mat');
load('classifiers.mat');

num_images = test.count;
new_test_images = shiftdim(test.images, 2);
A_new_test = reshape(new_test_images,num_images,28*28);
A_new_test = [A_new_test, ones(num_images,1)];
true_labels = test.labels;

%% ============================ Predict ==============================
UNCLASSIFIED = -1;
pred = UNCLASSIFIED * ones(num_images, 1);

clc;
images = test.images;

unclassifiedResultsCount = 0;

% Iterate over each image and multiply by every classifier
for i=1:num_images
    classifiedNumber = -1;
    bestResultDistance = 999;
    currentImage = A_new_test(i,:,:);

    for j=0:9
        % Set x to x0, x1 ... x9
        eval(['x = x' num2str(j)  ';']);

        result = currentImage*x;
        if result > 0
            % Calculate distance from 1
            distanceFrom1 = abs((result-1)^2);
            if distanceFrom1 < bestResultDistance
                bestResultDistance = distanceFrom1;
                classifiedNumber = j;
            end
        end
    end

    % Count the unclassified number
    if classifiedNumber == UNCLASSIFIED
        unclassifiedResultsCount = unclassifiedResultsCount + 1;
    end

    % Display the current image if it was classified wrong
    if classifiedNumber ~= true_labels(i,1)
        imagesc(images(:,:,i));
        colormap(gray(256))
        axis image; axis off; 
        title(['Classified image as ',num2str(classifiedNumber),'. Should be ',num2str(true_labels(i,1))]); 
        pause;
    end

    pred(i,1) = classifiedNumber;
end


%% =========================== Evaluate ==============================
acc = mean(pred == true_labels)*100;
disp(['Accuracy=',num2str(acc),'% (',num2str((1-acc/100)*num_images),' wrong examples)']); 
disp(['Couldnt classify ', num2str(unclassifiedResultsCount), ' results']); 
