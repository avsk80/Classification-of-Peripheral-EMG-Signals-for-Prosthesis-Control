function feature = extract_feature(data,win_size,win_inc)

if nargin < 3
    if nargin < 2
        win_size = 256;
    end
    win_inc = 32;
end

deadzone = 0.01;
feature1 = getrmsfeat(data,win_size,win_inc); % RMS
feature2 = getmavfeat(data,win_size,win_inc); % MAV
feature3 = getzcfeat(data,deadzone,win_size,win_inc); % ZC
feature4 = getsscfeat(data,deadzone,win_size,win_inc); % SSC
feature5 = getwlfeat(data,win_size,win_inc); % WL

ar_order = 6;
feature6 = getarfeat(data,ar_order,win_size,win_inc); % AR

%sizes = [size(feature1,1) size(feature2,1) size(feature3,1) size(feature4,1) size(feature5,1) size(feature6,1)]

%m = max(sizes)

%feature = [padarray(feature1,1) 
%padarray(feature2,m-size(feature2,1)) 
%padarray(feature3,m-size(feature3,1)) 
%padarray(feature4,m-size(feature4,1)) 
%padarray(feature5,m-size(feature5,1)) 
%padarray(feature6,m-size(feature6,1))];

feature = [feature1 feature2 feature3 feature4 feature5 feature6];

