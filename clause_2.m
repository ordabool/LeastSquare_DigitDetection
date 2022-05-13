clc; clear;
 
%%  ==================== Load MNIST dataset ======================
load('mnist.mat');

%% ======================= Parameters ===========================
N = 4000;

%%  ========= Find the classifier 'x' for every digit ===========
for wantedDigit=0:9

    % Reset the iteritive variables
    clear("imagesPerWantedDigit","imagesPerOtherDigits","A_all","b_all","A_train","b_train","x", "A_test", "b_test", "predC", "trueC", "acc", "error");

    disp(['------ Building classifier for the digit ', num2str(wantedDigit), ' ------']);

    % Seperate the images into imagesPerWantedDigit and imagesPerOtherDigits
    imagesPerWantedDigit = training.images(:,:,training.labels == wantedDigit);
    imagesPerOtherDigits = training.images(:,:,training.labels ~= wantedDigit);
    
    % Create A, b
    A_all = zeros(2*N,28^2);
    b_all = zeros(2*N,1);
    for i=1:N
        A_all(2*i-1,:) = reshape(imagesPerWantedDigit(:,:,i),1,28*28);
        A_all(2*i,:)   = reshape(imagesPerOtherDigits(:,:,i),1,28*28);
        b_all(2*i-1)   = +1;
        b_all(2*i)     = -1; 
    end
    A_all = [A_all, ones(2*N,1)];

    % Solve LS
    A_train = A_all(1:N,:); 
    b_train = b_all(1:N); 
    x=pinv(A_train)*b_train; 
    % Save x to a variable named x0, x1 ... x9 so we would have all of the digit classifiers
    eval(['x' num2str(wantedDigit) '= x;']);

    % Prepare Test Set
    A_test = A_all(N+1:2*N,:); 
    b_test = b_all(N+1:2*N); 
    
    % Check Performance
    predC = sign(A_train*x); 
    trueC = b_train; 
    disp('Train Error:'); 
    acc=mean(predC == trueC)*100;
    disp(['Accuracy=',num2str(acc),'% (',num2str((1-acc/100)*N),' wrong examples)']); 
    
    predC = sign(A_test*x); 
    trueC = b_test; 
    disp('Test Error:'); 
    acc=mean(predC == trueC)*100;
    disp(['Accuracy=',num2str(acc),'% (',num2str((1-acc/100)*N),' wrong examples)']); 

    % Show 5 of the Problematic Images
    error = find(predC~=trueC); 
    for k=1:1:5
        % Change to k=1:1:length(error) to see all problematic images
        figure(2);
        imagesc(reshape(A_test(error(k),1:28^2),[28,28]));
        colormap(gray(256))
        axis image; axis off; 
        title(['problematic digit number ',num2str(k),' :',num2str(A_test(error(k),:)*x)]); 
        pause;  
    end

end










