%% clear
clc
close all
clear all

%% generate matrix to store all data about images
% neural nwtwork data
% all features
nn_features=[];
% classes of all training photos
nn_classes=[];
% features after PCA code
nn_reduced_features=[];
% all normalized features for NN
nn_normal_features=[];
% normalized features after PCA code
nn_reduced_normal_features=[];

% SVM data
% all features
svm_features=[];
% classes of all training photos
svm_classes=[];
% features after PCA code
svm_reduced_features=[];
% to store ABCD features
ABCD_features=[];

%% acquisition and loading images
% load images matrix, containig images and class of each image
load training_database.mat;
% get the length of image matrix(number of images)
number_of_photos=length(photos);

%% prosess all image in loop
% to extract features for all 76 training images
for photo_num=1:number_of_photos
    %% read photo from matrix of photos
    im=photos(photo_num).photo;
    %figure, imshow(im);

    %% image resize to 512*512 using bilinear interpolation method
    im=imresize(im,[512, 512]);
    %figure, imshow(im);
    im_original=im; % to be used in crop after detectiong thr lesion
    
    %% convert image from RGB to graylevel
    im_gray=rgb2gray(im);
    %figure, imshow(im_gray);

    % from ISSN: 2249-7951
    
    %% contrast and brightness adjustment, gamma correction
%     H = vision.ContrastAdjuster;
%     im_gray1=step(H,im_gray);
%     %figure, imshow(im_gray1);
% %     from 1877-0509
    
    im_gray=imadjust(im_gray);
    %figure, imshow(im_gray);
    
%     ii=histeq(im_gray);
%     %figure, imshow(ii);
    
    %% median filter 
    im_gray=medfilt2(im_gray);
    %figure, imshow(im_gray);
    
    % from ISSN(p): 2249-684x
    
    %% Segmentation to detect a lesion region   
    
    % maximun entropy
    [level im_bw]=max_entropy(im_gray);
    %figure, imshow(im_bw);
    
    % outso method
%     im_bw=im2bw(im_gray);
    %figure, imshow(im_bw);

    %% complement of the image
    im_bw=imcomplement(im_bw);
    %figure, imshow(im_bw);

    %% filled the holes in the objects
    im_filled=imfill(im_bw,'holes');
    %figure, imshow(im_filled);
    
    %% opening
    im_opened = imopen(im_filled, strel('disk', 10));
    %figure, imshow(im_opened);
    im_opened=im2bw(im_opened);

    %% labeled image to get informations about objects 
    [im_labeled, num_of_objects]=bwlabel(im_opened,8);
    %store to data matrix
    data(photo_num).photos.im_labeled=im_labeled;

    %% extract properties of objects in labeled image
    im_info=regionprops(im_labeled,'all');
    %store to data matrix
    data(photo_num).im_info=im_info;

    %% using centroid to detect the central object(lesion)
    % size of image
    [n,m]=size(im_opened);
    % contre of the image and round to nearest integer
    centre_bw=[round(m/2),round(n/2)];
    if (im_labeled(round(m/2),round(n/2))>0)
        central_object_index=im_labeled(round(m/2),round(n/2));
    else 
        % minimum distance to the centre of the image
        min_distance=inf;
        % for loop to get the minimum didtance of objects
        for obj=1:num_of_objects
            distance=sqrt((im_info(obj).Centroid(1)-centre_bw(1))^2+(im_info(obj).Centroid(2)-centre_bw(2))^2);
            if distance<min_distance
                min_distance=distance;
                central_object_index=obj;
            end
        end
    end 

    %% return the central object (lesion)
    central_bounding_box=im_info(central_object_index).BoundingBox;
    % crop the central object
    im_bw_croped=imcrop(im_labeled,central_bounding_box);
    %figure, imshow(im_bw_croped);

    %% calculate labeled image for lesion
    [im_bw_croped_labeled, num_of_objects]=bwlabel(im_bw_croped,8);
    lesion_info=regionprops(im_bw_croped_labeled,'all');
    
    %% determine the largest area b/t objects (lesion)
    % calculate all areas of objects in one matrix after filling the holes;
    objects_areas=cat(1,lesion_info.Area);

    %% determine the largest object in the image
    max_area=0;
    large_object_index=0;
    for i=1:length(objects_areas)
        if(objects_areas(i)>max_area)
            max_area=objects_areas(i);
            large_object_index=i;
        end
    end

    %% delete the objects that has areas smaller than the largest one
    im_bw_croped=bwareaopen(im_bw_croped,max_area);
    %figure, imshow(im_bw_croped); 
    
    %% crop lesion area from original image
    % crop the lesion area from the original image after detection the lesion
    %   by using binary image and objects region properties;
    im_original_croped=imcrop(im_original,central_bounding_box);
    %figure, imshow(im_original_croped);

    %% remove any things except the tumor object
    r=im_original_croped(:,:,1);
    g=im_original_croped(:,:,2);
    b=im_original_croped(:,:,3);
    rs=(r).*uint8(im_bw_croped);
    gs=(g).*uint8(im_bw_croped);
    bs=(b).*uint8(im_bw_croped);
    im_original_croped=uint8(zeros(size(im_original_croped)));
    im_original_croped(:,:,1)=rs;
    im_original_croped(:,:,2)=gs;
    im_original_croped(:,:,3)=bs;
    %figure, imshow(im_original_croped)

    %% return croped image to gray levels
    im_gray_croped=rgb2gray(im_original_croped);
    %figure, imshow(im_gray_croped);
    
    %% ABCD features
    %% A: asymmetry index (AI)
    lesion_area=bwarea(im_bw_croped);
    [height, width]=size(im_bw_croped);
    image_area=height*width;
    dA=image_area-lesion_area;
    %area indes
    AI=(dA/image_area)*100;
    
    %% B: border irregularity (campact index CI)
    %perimeter oa the lesion(number of edge pixles)
    im_bw_bounadaries = bwperim(im_bw_croped);
    %figure, imshow(im_bw_bounadaries);
%     perimeter=bwarea(im_bw_bounadaries);
    perimeter=0;
    [x,y]=size(im_bw_bounadaries);
    for i=1:x
        for j=1:y
            if(im_bw_bounadaries(i,j)==1)
                perimeter=perimeter+1;
            end
        end
    end
    %compaxt index (CI)
    CI=(power(perimeter,2))/(4*pi*lesion_area);
    
    %% C: color variation (CV)
    % colors values
    white=[255, 255, 255];
    black=[0, 0, 0];
    red=[255, 0, 0];
    light_broun=[205, 133, 63];
    dark_broun=[101, 67, 33];
    blue_gray=[0, 134, 139];
    
    %convert RGB image to double
    im_double=im2double(im_original_croped);
    
    % calculate the distance from any pixle to each color-value
    % get im_bw_croped size to check the object pixle just
    [r,c]=size(im_bw_croped);
    color_count=zeros(1,6);
    for x=1:r
        for y=1:c
            if(im_bw_croped(x,y)==1)
                %distance
                %white
                dist1=sqrt((im_double(x,y,1)-white(1))^2+(im_double(x,y,2)-white(2))^2+(im_double(x,y,3)-white(3))^2);
                if(dist1==0)
                    color_count(1,1)=1;
                end
                %black
                dist2=sqrt((im_double(x,y,1)-black(1))^2+(im_double(x,y,2)-black(2))^2+(im_double(x,y,3)-black(3))^2);
                if(dist2==0)
                    color_count(1,2)=1;
                end
                %red
                dist3=sqrt((im_double(x,y,1)-red(1))^2+(im_double(x,y,2)-red(2))^2+(im_double(x,y,3)-red(3))^2);
                if(dist3==0)
                    color_count(1,3)=1;
                end
                %light broun
                dist4=sqrt((im_double(x,y,1)-light_broun(1))^2+(im_double(x,y,2)-light_broun(2))^2+(im_double(x,y,3)-light_broun(3))^2);
                if(dist4==0)
                    color_count(1,4)=1;
                end
                %dark broun
                dist5=sqrt((im_double(x,y,1)-dark_broun(1))^2+(im_double(x,y,2)-dark_broun(2))^2+(im_double(x,y,3)-dark_broun(3))^2);
                if(dist5==0)
                    color_count(1,5)=1;
                end
                %blue gray
                dist6=sqrt((im_double(x,y,1)-blue_gray(1))^2+(im_double(x,y,2)-blue_gray(2))^2+(im_double(x,y,3)-blue_gray(3))^2);
                if(dist6==0)
                    color_count(1,6)=1;
                end
            end
        end
    end
    color_feature=sum(color_count);
    
    %% D: diameter (D) (equivalent diameter)
    diameter_pixles=[lesion_info(1).EquivDiameter];
    diameter=diameter_pixles*0.264583333;
    
    %% store ABCD features to matrix
    ABCD_features(photo_num,:)=[AI, CI, color_feature, diameter];
    
    %% TDS features
    TDS=(AI*1.3)+(CI*0.1)+(color_feature*0.5)+(diameter*0.5);
    
    %% boundaries and circulation
    boundaries= bwboundaries(im_bw_croped,'noholes');
    %Display the label matrix and draw each boundary
    %figure, imshow(im_bw_croped);
%     hold on
    for k = 1:length(boundaries)
        boundary = boundaries{k};
%         plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
    end
%     stats = regionprops(L,'Area','Centroid');
%     threshold = 0.94;
    %
    % % loop over the boundaries
    for k = 1:length(boundaries)
        % obtain (X,Y) boundary coordinates corresponding to label 'k'
        boundary = boundaries{k};
        % compute a simple estimate of the object's perimeter
        delta_sq = diff(boundary).^2;
        perimeter = sum(sqrt(sum(delta_sq,2)));
        % obtain the area calculation corresponding to label 'k'
        area = lesion_info(k).Area;
        % compute the roundness metric
        metric = 4*pi*area/perimeter^2;
        % display the results
        circulation_string = sprintf('%2.2f',metric);
        % mark objects above the threshold with a black circle
%         if metric > threshold
%             centroid = stats(k).Centroid;
%             plot(centroid(1),centroid(2),'ko');
%         end
%         text(boundary(1,2)-35,boundary(1,1)+13,circulation_string,'Color','y',...
%         'FontSize',14,'FontWeight','bold');
    end
    circulation=str2double(circulation_string);
    
    %%
    im_histogram = histeq(im_gray_croped);
    %figure, imshow(im_histogram), colormap gray
    % mean
    M = mean2(im_histogram(:));
    % standard deviation
    SD =std(double(im_histogram(:)));
    % skewness
    im_double= im2double(im_histogram);
    S=skewness(im_double(:));
    % kurtosis
    K=kurtosis(double(im_histogram(:)));
    % entropy
    Ent=entropy(im_histogram(:));
    
    
    %% features using GLCM
    %% creates a gray-level co-occurrence matrix (GLCM)
    GLCM = graycomatrix(im_gray_croped);

    %% statistical properties of the GLCM.
    % GLCM main features
    GLCM_features = graycoprops(GLCM,'all');
    
    %store to data matrix
    data(photo_num).info.glcm=GLCM;
    % //note
    %     GLCM in binary image, give us the same result as in graySale
    % from ISSN: 2249-7951
    
    %% set all features in one matrix
    % features
    nn_features(1,photo_num)=GLCM_features.Contrast;
    nn_features(2,photo_num)=S; % skewness
    nn_features(3,photo_num)=K; % kurtosis
    nn_features(4,photo_num)=Ent; %entropy
    nn_features(5,photo_num)=M; % mean
    nn_features(6,photo_num)=SD;% dtandard deviation
    nn_features(7,photo_num)=circulation;
    nn_features(8,photo_num)=GLCM_features.Energy; % energy
    nn_features(9,photo_num)=GLCM_features.Correlation;
    nn_features(10,photo_num)=GLCM_features.Homogeneity;
    nn_features(11,photo_num)=TDS;
    
    %% set the class to the image
    % in neural network
    % nn_classes=[1,0]: abnormal
    % nn_classes=[0,1]: normal
    % in support vector machine
    % svm_classes=1 : abnormal
    % svm_classes=0 : normal
    
    nn_classes(:,photo_num)=photos(photo_num).nn_class;
    svm_classes(photo_num,1)=photos(photo_num).svm_class;

    %% plot all images
%     %figure
%     subplot 321
%     imshow(im);
%     title('original image');
% 
%     subplot 322
%     imshow(im_gray);
%     title('gray image');
% 
%     subplot 323
%     imshow(im_opened);
%     title('segmented image');
% 
%     subplot 324
%     imshow(im_bw_croped);
%     title('binary lesion');
% 
%     subplot 325
%     imshow(im_original_croped);
%     title('lesion original image');
% 
%     subplot 326
%     imshow(im_gray_croped);
%     title('lesion gray image');

      close all
end % end of for loop

%% SVM data
% convert to SVM features (column features)
svm_features=nn_features';

%% normalizing nn_features
nn_normal_features=[];
for i=1:11
    for j=1:76
        nn_normal_features(i,j)=(nn_features(i,j)-min(nn_features(i,:)))/(max(nn_features(i,:))-min(nn_features(i,:)));
    end
end

%% PCA
reduced_features_index=PCA(nn_features');

% select features after PCA-code
nn_reduced_features=[];
nn_reduced_normal_features=[];
svm_reduced_features=[];
i=length(reduced_features_index);
while i>=(length(reduced_features_index)-4)    
    nn_reduced_normal_features=[nn_reduced_normal_features; nn_normal_features(reduced_features_index(i),:)];
    nn_reduced_features=[nn_reduced_features; nn_features(reduced_features_index(i),:)];
    svm_reduced_features=[svm_reduced_features, svm_features(:,reduced_features_index(i))];
    i=i-1;
end

%% saving all data

% % neural network data
% save('nn_features.mat','nn_features');
% save('nn_classes.mat','nn_classes');
% save('nn_reduced_features.mat','nn_reduced_features');
% save('nn_normal_features.mat','nn_normal_features');
% save('nn_reduced_normal_features.mat','nn_reduced_normal_features');
% % support vector machine data
% save('svm_classes.mat','svm_classes');
% save('svm_features.mat','svm_features');
% save('svm_reduced_features.mat','svm_reduced_features');
% % PCA data
% save('features_to_pca.mat','features_to_pca');
% save('reduced_features_index.mat','reduced_features_index');
% save('ABCD_features.mat','ABCD_features');


