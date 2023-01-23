function varargout = GUI(varargin)
% GUI MATLAB code for GUI.fig
%      GUI, by itself, creates a new GUI or raises the existing
%      singleton*.
%
%      H = GUI returns the handle to a new GUI or the handle to
%      the existing singleton*.
%
%      GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUI.M with the given input arguments.
%
%      GUI('Property','Value',...) creates a new GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GUI

% Last Modified by GUIDE v2.5 13-Feb-2017 21:41:57

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT
% clear command
clc

%% define global varilables
% iamges 
global im_original;
global im_original_croped;
global im_bw_croped;
global im_gray_croped;
global im_histogram;
% features
global nn_features;
global svm_features;
% classes
global svm_class;
global nn_class;


% --- Executes just before GUI is made visible.
function GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GUI (see VARARGIN)

% Choose default command line output for GUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in select_image_button.
function select_image_button_Callback(hObject, eventdata, handles)
% hObject    handle to select_image_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA) 
%% load classifiers
% load neural network classifier
load('nn_net.mat');
% load support vector machine classifier
load('svm_net.mat');
% load reduced features index, after PCA code
load('reduced_features_index.mat');

%% reset all matrices
nn_features=[];
nn_class=[];
nn_reduced_features=[];
svm_features=[];
svm_class=[];
svm_reduced_features=[];
svm_class=[];
nn_class=[];

%% select image
[image_directory,user_calceller]=imgetfile;
im=imread(image_directory);

%% image resize to 512*512
im=imresize(im,[512, 512]);
im_original=im; % to be used in crop after detectiong thr lesion
%show original image
axes(handles.im_original_axes);
imshow(im_original);
title('Original image');

%% convert image from RGB to graylevel
im_gray=rgb2gray(im);
% from ISSN: 2249-7951

%% contrast and brightness adjustment, gamma correction
%     H = vision.ContrastAdjuster;
%     im_gray=step(H,im_gray);
%     %figure, imshow(im_gray);
% from 1877-0509

im_gray=imadjust(im_gray);
%figure, imshow(im_gray);

%% median filter 
im_gray=medfilt2(im_gray);
% from ISSN(p): 2249-684x

%% Segmentation to detect a lesion region   

% maximun entropy
%     [level im_bw]=max_entropy(im_gray);
%     %figure, imshow(im_bw);

% outso method
im_bw=im2bw(im_gray);

%% complement of the image
im_bw=imcomplement(im_bw);

%% filled the holes in the objects
im_filled=imfill(im_bw,'holes');

  %% erosion
%     SE = strel('disk',5);    
%     im_eroded = imerode(im_filled,SE);
%     %figure, imshow(im_eroded);
%     im_eroded=im2bw(im_eroded);

%% opening
im_opened = imopen(im_filled, strel('disk', 10));

%% labeled image to get informations about objects 
[im_labeled,num_of_objects]=bwlabel(im_opened,8);

%% extract properties of objects in labeled image
im_info=regionprops(im_labeled,'all');

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
%show croped binary image
axes(handles.im_bw_croped_axes);
imshow(im_bw_croped);
title('Segmented lesion');

%% crop lesion area from original image
% crop the lesion area from the original image after detection the lesion
%   by using binary image and objects region properties;
im_original_croped=imcrop(im_original,central_bounding_box);


%% remove any things except the tumor object
% [rows,columns]=size(im_bw_croped);
% for x=1:rows
%     for y=1:columns
%         if(im_bw_croped(x,y)==0)
%             im_original_croped(x,y,1)=255;
%             im_original_croped(x,y,2)=255;
%             im_original_croped(x,y,3)=255;
%         end
%     end
% end

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
%show original croped image
axes(handles.im_original_croped_axes);
imshow(im_original_croped);
title('Original lesion');

%% return croped image to gray levels
im_gray_croped=rgb2gray(im_original_croped);
%show gray image croped
axes(handles.im_gray_croped_axes);
imshow(im_gray_croped);
title('Gray lesion');

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
diameter=diameter_pixles/3.779527559055;

%% TDS features
TDS=(AI*1.3)+(CI*0.1)+(color_feature*0.5)+(diameter*0.5);

%% boundaries and circulation
boundaries= bwboundaries(im_bw_croped,'noholes');
%Display the label matrix and draw each boundary
%figure, imshow(im_bw_croped);
% hold on
for k = 1:length(boundaries)
    boundary = boundaries{k};
%     plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
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
%     text(boundary(1,2)-35,boundary(1,1)+13,circulation_string,'Color','y',...
%     'FontSize',14,'FontWeight','bold');
end
circulation=str2double(circulation_string);

%%
im_histogram = histeq(im_gray_croped);
%show gray scale image
axes(handles.im_histogram_axes);
imagesc(im_histogram);
colormap gray;
title('Scaled image');

% mean
M = mean2(im_histogram(:));
% standard deviation
SD =std(double(im_histogram(:)));
% skewness
im_double= im2double(im_histogram);
S=skewness(im_double(:));
% kurtosis
K=kurtosis(double(im_histogram(:)));
% energy
im_histogram_info=graycoprops(im_histogram,{'energy'});
Enr=im_histogram_info(1).Energy;
% entropy
Ent=entropy(im_histogram(:));


%% features using GLCM
%% creates a gray-level co-occurrence matrix (GLCM)
GLCM = graycomatrix(im_gray_croped);

%% statistical properties of the GLCM.
% GLCM main features
GLCM_features = graycoprops(GLCM,'all');

% GLCM 22-features 
%     glcm_features=GLCM_features(GLCM);

% //note
%     GLCM in binary image, give us the same result as in graySale
% from ISSN: 2249-7951

%% set all features in one matrix
% features
nn_features(1,1)=GLCM_features.Contrast;
nn_features(2,1)=S; % skewness
nn_features(3,1)=K; % kurtosis
nn_features(4,1)=Ent; %entropy
nn_features(5,1)=M; % mean
nn_features(6,1)=SD;% dtandard deviation
nn_features(7,1)=circulation;
nn_features(8,1)=Enr; % energy
nn_features(9,1)=GLCM_features.Correlation;
nn_features(10,1)=GLCM_features.Homogeneity;
nn_features(11,1)=TDS;

%% SVM data
% cconvert to SVM features
svm_features=[];
[r,c]=size(nn_features);
for i=1:r
    svm_features(:,i)=nn_features(i,:);
end

% %convert classes to svm_classes
% svm_classes=[];
% [r,c]=size(nn_classes);
% for i=1:c
%     svm_classes(i)=nn_classes(1,i);
% end

%% assign values to features-boxes
set(handles.contrast_box,'string',nn_features(1,1));
set(handles.skewness_box,'string',nn_features(2,1));
set(handles.kurtosis_box,'string',nn_features(3,1));
set(handles.entropy_box,'string',nn_features(4,1));
set(handles.mean_box,'string',nn_features(5,1));
set(handles.standard_deviation_box,'string',nn_features(6,1));
set(handles.circulation_box,'string',nn_features(7,1));
set(handles.energy_box,'string',nn_features(8,1));
set(handles.correlation_box,'string',nn_features(9,1));
set(handles.homogenity_box,'string',nn_features(10,1));
set(handles.TDS_box,'string',nn_features(11,1));

%% reduce features
nn_reduced_features=[];
svm_reduced_features=[];

i=length(reduced_features_index);
while i>=(length(reduced_features_index)-4)    
    nn_reduced_features=[nn_reduced_features; nn_features(reduced_features_index(i),:)];
    svm_reduced_features=[svm_reduced_features, svm_features(:,reduced_features_index(i))];
    i=i-1;
end

%% classify image
% neural network
nn_class=[];
nn_result=nn_net(nn_reduced_features);
if(nn_result(1,1)>nn_result(2,1))
    nn_class='Abnormal';
else 
    nn_class='Normal';
end

% SVM
svm_class=[];
svm_result=svmclassify(svm_net,svm_reduced_features);
if(svm_result==1)
    svm_class='Abnormal';
else
    svm_class='Normal';
end

%assign to text Box
set(handles.nn_result_box,'string',nn_class);
set(handles.svm_result_box,'string',svm_class);

function contrast_box_Callback(hObject, eventdata, handles)
% hObject    handle to contrast_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of contrast_box as text
%        str2double(get(hObject,'String')) returns contents of contrast_box as a double


% --- Executes during object creation, after setting all properties.
function contrast_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to contrast_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function skewness_box_Callback(hObject, eventdata, handles)
% hObject    handle to skewness_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of skewness_box as text
%        str2double(get(hObject,'String')) returns contents of skewness_box as a double


% --- Executes during object creation, after setting all properties.
function skewness_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to skewness_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function kurtosis_box_Callback(hObject, eventdata, handles)
% hObject    handle to kurtosis_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of kurtosis_box as text
%        str2double(get(hObject,'String')) returns contents of kurtosis_box as a double


% --- Executes during object creation, after setting all properties.
function kurtosis_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to kurtosis_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function entropy_box_Callback(hObject, eventdata, handles)
% hObject    handle to entropy_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of entropy_box as text
%        str2double(get(hObject,'String')) returns contents of entropy_box as a double


% --- Executes during object creation, after setting all properties.
function entropy_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to entropy_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function mean_box_Callback(hObject, eventdata, handles)
% hObject    handle to mean_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of mean_box as text
%        str2double(get(hObject,'String')) returns contents of mean_box as a double


% --- Executes during object creation, after setting all properties.
function mean_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to mean_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function circulation_box_Callback(hObject, eventdata, handles)
% hObject    handle to circulation_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of circulation_box as text
%        str2double(get(hObject,'String')) returns contents of circulation_box as a double


% --- Executes during object creation, after setting all properties.
function circulation_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to circulation_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function energy_box_Callback(hObject, eventdata, handles)
% hObject    handle to energy_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of energy_box as text
%        str2double(get(hObject,'String')) returns contents of energy_box as a double


% --- Executes during object creation, after setting all properties.
function energy_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to energy_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function correlation_box_Callback(hObject, eventdata, handles)
% hObject    handle to correlation_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of correlation_box as text
%        str2double(get(hObject,'String')) returns contents of correlation_box as a double


% --- Executes during object creation, after setting all properties.
function correlation_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to correlation_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function standard_deviation_box_Callback(hObject, eventdata, handles)
% hObject    handle to standard_deviation_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of standard_deviation_box as text
%        str2double(get(hObject,'String')) returns contents of standard_deviation_box as a double


% --- Executes during object creation, after setting all properties.
function standard_deviation_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to standard_deviation_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function TDS_box_Callback(hObject, eventdata, handles)
% hObject    handle to TDS_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of TDS_box as text
%        str2double(get(hObject,'String')) returns contents of TDS_box as a double


% --- Executes during object creation, after setting all properties.
function TDS_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TDS_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function svm_result_box_Callback(hObject, eventdata, handles)
% hObject    handle to svm_result_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of svm_result_box as text
%        str2double(get(hObject,'String')) returns contents of svm_result_box as a double


% --- Executes during object creation, after setting all properties.
function svm_result_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to svm_result_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function nn_result_box_Callback(hObject, eventdata, handles)
% hObject    handle to nn_result_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of nn_result_box as text
%        str2double(get(hObject,'String')) returns contents of nn_result_box as a double


% --- Executes during object creation, after setting all properties.
function nn_result_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nn_result_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function homogenity_box_Callback(hObject, eventdata, handles)
% hObject    handle to homogenity_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of homogenity_box as text
%        str2double(get(hObject,'String')) returns contents of homogenity_box as a double


% --- Executes during object creation, after setting all properties.
function homogenity_box_CreateFcn(hObject, eventdata, handles)
% hObject    handle to homogenity_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in clear_button.
function clear_button_Callback(hObject, eventdata, handles)
% hObject    handle to clear_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%%
set(handles.contrast_box,'string','');
set(handles.skewness_box,'string','');
set(handles.kurtosis_box,'string','');
set(handles.entropy_box,'string','');
set(handles.mean_box,'string','');
set(handles.standard_deviation_box,'string','');
set(handles.circulation_box,'string','');
set(handles.energy_box,'string','');
set(handles.correlation_box,'string','');
set(handles.homogenity_box,'string','');
set(handles.TDS_box,'string','');
set(handles.nn_result_box,'string','');
set(handles.svm_result_box,'string','');
emp=[1,1;1,1];
axes(handles.im_original_axes)
imshow(emp);
axes(handles.im_bw_croped_axes)
imshow(emp);
axes(handles.im_original_croped_axes)
imshow(emp);
axes(handles.im_gray_croped_axes)
imshow(emp);
axes(handles.im_histogram_axes)
imshow(emp);
nn_class=[];
svm_class=[];
nn_features=[];
nn_reduced_features=[];
svm_features=[];
svm_reduced_features=[];
nn_result=[];
svm_result=[];