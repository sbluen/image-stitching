%This project uses Renuka's SIFT code. The keypoint locations are slightly
%approximated during the process of normalizing them. To save time, only
%the first 20 pairs of matching keypoints are used.
clear all
close all
clc


%function [kpts,kp_descriptor] = prog5 (im,sigma_orig,num_octaves,num_scales,r)
%constants
 sigma_orig= 1;
 num_octaves= 3;
 num_scales= 2;
 r= 10;
 match_threshold=0.75;
 
 im1=imresize(imread(input('Enter filename of image 1:', 's')), 0.5);
 im2=imresize(imread(input('Enter filename of image 2:', 's')), 0.5);

im=im1;
im_store=im;

% Get grayscale, double precision image normalized to lie between 0 and 1
if size(im,3)==3
    im=rgb2gray(im);
end
im=double(im);
im=im-min(min(im));
im=im/max(max(im));

% Anti-aliasing followed by resizing and pre-blurring
im=conv2(im,fspecial('gaussian',[9 9],0.5),'same');
im=imresize(im,2);
im=conv2(im,fspecial('gaussian',[9 9],1),'same');

% Get value of k
k=2^(1/num_scales);

% Build Gaussian stack
for i=1:num_octaves
    sigma=sigma_orig;
    temp=[];
    for j=1:num_scales+3
        sigma_f=sqrt(k^2-1)*sigma;
        sigma=k*sigma;
        sigmas(i,j)=sigma*0.5*2^(i-1);
        im=conv2(im,fspecial('gaussian',[15 15],sigma_f),'same');
        temp(:,:,j)=im;
    end
    G_octave{i}=temp;
    im=temp(:,:,end-2); % Must downsample the image two images from the top of the stack
    im=imresize(im,0.5,'nearest');
end

% Build Difference of Gaussian (DoG) stack
for i=1:num_octaves
    temp=[];
    for j=2:num_scales+3
        temp(:,:,j-1)=G_octave{i}(:,:,j-1)-G_octave{i}(:,:,j);
    end
    DoG_octave{i}=temp;
end

% Extrema Detection
for i=1:num_octaves
    temp=[];
    for j=2:num_scales+1
        tempim=DoG_octave{i}(:,:,j-1:j+1);
        tempd=(imregionalmax(tempim,26)+imregionalmin(tempim,26))>0;
        tempd=tempd(:,:,2);
        % Eliminate the detected extrema which are too close to the border
        border=5*2^(num_octaves-i-1);
        tempd(1:border,:)=0;
        tempd(end-border+1:end,:)=0;
        tempd(:,1:border)=0;
        tempd(:,end-border+1:end)=0;
        temp(:,:,j-1)=tempd;
    end
    extrema{i}=temp;
end

% Eliminate Low Contrast points
for i=1:num_octaves
    temp=extrema{i};
    check=DoG_octave{i};
    for j=1:num_scales
        checkd=check(:,:,j+1);
        checkd=abs(checkd)>(0.03); % Can change threshold to see the difference
        temp(:,:,j)=temp(:,:,j).*checkd;       
    end
    extrema{i}=temp;
end

% Elminate Edges
for i=1:num_octaves
    temp=extrema{i};
    check=DoG_octave{i};
    for j=1:num_scales
        checkd=check(:,:,j+1);
        Dx=conv2(checkd,[1 0 -1],'same');
        Dy=conv2(checkd,[1 0 -1]','same');
        Dxx=conv2(Dx,[1 0 -1],'same');
        Dxy=conv2(Dx,[1 0 -1]','same');
        Dyy=conv2(Dy,[1 0 -1]','same');
        trace=Dxx+Dyy;
        detmt=Dxx.*Dyy-Dxy.^2;
        ratio=(trace.^2)./detmt;
        checkd=ratio<((r+1)^2/r);
        temp(:,:,j)=temp(:,:,j).*checkd;
    end
    extrema{i}=temp;
end

% Assign Orientation and Scale
for i=1:num_octaves
    temp=G_octave{i};
    tempm=[];
    tempo=[];
    for j=2:num_scales+1
        gaussim=temp(:,:,j);
        Dx=conv2(gaussim,[1 0 -1],'same');
        Dy=conv2(gaussim,[1 0 -1]','same');
        mag=sqrt(Dx.^2+Dy.^2);
        ori=180/pi*atan2(Dy,Dx);
        ori=ori+360*(ori<0);
        tempm(:,:,j-1)=mag;
        tempo(:,:,j-1)=ori;
    end
    mag_octave{i}=tempm;
    ori_octave{i}=tempo;
end

% Find location of all keypoints
kpts=[];
kp_count=0;
for i=1:num_octaves
    tempm=mag_octave{i};
    tempo=ori_octave{i};
    tempe=extrema{i};
    for j=1:num_scales
        tempim=tempm(:,:,j);
        sigma_temp=1.5*sigmas(i,j);
        wsize=2*round(sigma_temp);
        % Define the weighing filter
        weigh=fspecial('gaussian',[2*wsize+1 2*wsize+1],sigma_temp);
        kp_mag=tempm(:,:,j);
        kp_ori=tempo(:,:,j);
        keyps=tempe(:,:,j);
        [row,col]=find(keyps);
        [L,B]=size(keyps);
        for count=1:numel(row)
            % Start building histogram
            vect=zeros(1,36);
            for l=max(1,row(count)-wsize):min(L,row(count)+wsize)
                for b=max(1,col(count)-wsize):min(B,col(count)+wsize)
                    wt=kp_mag(l,b)*weigh(l-row(count)+wsize+1,b-col(count)+wsize+1);
                    bin=ceil(kp_ori(l,b)/10);
                    vect(bin)=vect(bin)+wt;
                end
            end
            peak=max(vect);
            for step=1:36
                % Store all keypoints whose magnitude is greater than 80%
                % of the maximum magnitude - store full location
                % information, scale and angle information
                if vect(step)>0.8*peak
                    kp_count=kp_count+1;
                    kpts(kp_count,:)=[i j row(count) col(count) vect(step) (step-1)*10+5;];
                end
            end
        end
    end
end

% Find descriptors
for count=1:kp_count
    kp_current=kpts(count,:);
    i=kp_current(1);
    j=kp_current(2);
    ori=kp_current(6);
    % Define the Gaussian weighing filter
    fil=fspecial('gaussian',[16 16],8);
    tempm=mag_octave{i};
    tempm=tempm(:,:,j);
    tempo=ori_octave{i};
    tempo=tempo(:,:,j);
    [l,b]=size(tempm);
    % Zero-pad the magnitude and orientation matrices
    tempm=[zeros(l,25) tempm zeros(l,25)];
    tempm=[zeros(25,b+50); tempm; zeros(25,b+50)];
    tempo=[zeros(l,25) tempo zeros(l,25)];
    tempo=[zeros(25,b+50); tempo; zeros(25,b+50)];
    % Rotate the matrices by -orientation of the keypoint instead of 
    % rotating window by +orientation for simplicity    
    tempm=imrotate(tempm,-ori);
    tempo=imrotate(tempo,-ori);
    [l,b]=size(tempm);
    % Get 16x16 arrays and weigh the magnitude array with the Gaussian
    tempm=tempm(floor(l/2)-7:floor(l/2)+8,floor(b/2)-7:floor(b/2)+8);
    tempo=tempo(floor(l/2)-7:floor(l/2)+8,floor(b/2)-7:floor(b/2)+8);
    tempm=tempm.*fil;
    % Initialize descriptors
    des=[];
    % Get 4x4 sub-arrays
    for i=1:4:13
        for j=1:4:13
            des_temp=zeros(1,8);
            win_mag=tempm(i:i+3,j:j+3);
            win_ori=tempo(i:i+3,j:j+3);
            for x=1:4
                for y=1:4
                    % See which bin the given orientation corresponds to
                    bin=ceil(win_ori(x,y)/45);
                    % Get the distance weighing factor
                    trilin_wt=abs(win_ori(x,y)-(2*bin-1)*22.5)/45;
                    % Increment the required bin
                    des_temp(bin)=des_temp(bin)+win_mag(x,y)*(1-trilin_wt);
                end
            end
            des=[des des_temp];
        end
    end
    % Normalize the descriptor
    des=des/norm(des,2);
    % Truncate to lie below 0.2
    des(des>0.2)=0.2;
    % Renormalize
    des=des/norm(des,2);
    %add coordinate information for homographies
    cur_octave=i;
%     row=floor(kp_current(3)*(2^(cur_octave-1))+0.5*(2^(cur_octave-1)));
%     col=floor(kp_current(4)*(2^(cur_octave-1))+0.5*(2^(cur_octave-1)));
    row=kp_current(1,3)*2^(kp_current(1,1)-1);
    col=kp_current(1,4)*2^(kp_current(1,1)-1);
    kp_des1(count,:)=[des row col];
end

im=im2;

im_store=im;

% Get grayscale, double precision image normalized to lie between 0 and 1
if size(im,3)==3
    im=rgb2gray(im);
end
im=double(im);
im=im-min(min(im));
im=im/max(max(im));

% Anti-aliasing followed by resizing and pre-blurring
im=conv2(im,fspecial('gaussian',[9 9],0.5),'same');
im=imresize(im,2);
im=conv2(im,fspecial('gaussian',[9 9],1),'same');

% Get value of k
k=2^(1/num_scales);

% Build Gaussian stack
for i=1:num_octaves
    sigma=sigma_orig;
    temp=[];
    for j=1:num_scales+3
        sigma_f=sqrt(k^2-1)*sigma;
        sigma=k*sigma;
        sigmas(i,j)=sigma*0.5*2^(i-1);
        im=conv2(im,fspecial('gaussian',[15 15],sigma_f),'same');
        temp(:,:,j)=im;
    end
    G_octave{i}=temp;
    im=temp(:,:,end-2); % Must downsample the image two images from the top of the stack
    im=imresize(im,0.5,'nearest');
end

% Build Difference of Gaussian (DoG) stack
for i=1:num_octaves
    temp=[];
    for j=2:num_scales+3
        temp(:,:,j-1)=G_octave{i}(:,:,j-1)-G_octave{i}(:,:,j);
    end
    DoG_octave{i}=temp;
end

% Extrema Detection
for i=1:num_octaves
    temp=[];
    for j=2:num_scales+1
        tempim=DoG_octave{i}(:,:,j-1:j+1);
        tempd=(imregionalmax(tempim,26)+imregionalmin(tempim,26))>0;
        tempd=tempd(:,:,2);
        % Eliminate the detected extrema which are too close to the border
        border=5*2^(num_octaves-i-1);
        tempd(1:border,:)=0;
        tempd(end-border+1:end,:)=0;
        tempd(:,1:border)=0;
        tempd(:,end-border+1:end)=0;
        temp(:,:,j-1)=tempd;
    end
    extrema{i}=temp;
end

% Eliminate Low Contrast points
for i=1:num_octaves
    temp=extrema{i};
    check=DoG_octave{i};
    for j=1:num_scales
        checkd=check(:,:,j+1);
        checkd=abs(checkd)>(0.03); % Can change threshold to see the difference
        temp(:,:,j)=temp(:,:,j).*checkd;       
    end
    extrema{i}=temp;
end

% Elminate Edges
for i=1:num_octaves
    temp=extrema{i};
    check=DoG_octave{i};
    for j=1:num_scales
        checkd=check(:,:,j+1);
        Dx=conv2(checkd,[1 0 -1],'same');
        Dy=conv2(checkd,[1 0 -1]','same');
        Dxx=conv2(Dx,[1 0 -1],'same');
        Dxy=conv2(Dx,[1 0 -1]','same');
        Dyy=conv2(Dy,[1 0 -1]','same');
        trace=Dxx+Dyy;
        detmt=Dxx.*Dyy-Dxy.^2;
        ratio=(trace.^2)./detmt;
        checkd=ratio<((r+1)^2/r);
        temp(:,:,j)=temp(:,:,j).*checkd;
    end
    extrema{i}=temp;
end

% Assign Orientation and Scale
for i=1:num_octaves
    temp=G_octave{i};
    tempm=[];
    tempo=[];
    for j=2:num_scales+1
        gaussim=temp(:,:,j);
        Dx=conv2(gaussim,[1 0 -1],'same');
        Dy=conv2(gaussim,[1 0 -1]','same');
        mag=sqrt(Dx.^2+Dy.^2);
        ori=180/pi*atan2(Dy,Dx);
        ori=ori+360*(ori<0);
        tempm(:,:,j-1)=mag;
        tempo(:,:,j-1)=ori;
    end
    mag_octave{i}=tempm;
    ori_octave{i}=tempo;
end

% Find location of all keypoints
kpts=[];
kp_count=0;
for i=1:num_octaves
    tempm=mag_octave{i};
    tempo=ori_octave{i};
    tempe=extrema{i};
    for j=1:num_scales
        tempim=tempm(:,:,j);
        sigma_temp=1.5*sigmas(i,j);
        wsize=2*round(sigma_temp);
        % Define the weighing filter
        weigh=fspecial('gaussian',[2*wsize+1 2*wsize+1],sigma_temp);
        kp_mag=tempm(:,:,j);
        kp_ori=tempo(:,:,j);
        keyps=tempe(:,:,j);
        [row,col]=find(keyps);
        [L,B]=size(keyps);
        for count=1:numel(row)
            % Start building histogram
            vect=zeros(1,36);
            for l=max(1,row(count)-wsize):min(L,row(count)+wsize)
                for b=max(1,col(count)-wsize):min(B,col(count)+wsize)
                    wt=kp_mag(l,b)*weigh(l-row(count)+wsize+1,b-col(count)+wsize+1);
                    bin=ceil(kp_ori(l,b)/10);
                    vect(bin)=vect(bin)+wt;
                end
            end
            peak=max(vect);
            for step=1:36
                % Store all keypoints whose magnitude is greater than 80%
                % of the maximum magnitude - store full location
                % information, scale and angle information
                if vect(step)>0.8*peak
                    kp_count=kp_count+1;
                    kpts(kp_count,:)=[i j row(count) col(count) vect(step) (step-1)*10+5;];
                end
            end
        end
    end
end

% Find descriptors
for count=1:kp_count
    kp_current=kpts(count,:);
    i=kp_current(1);
    j=kp_current(2);
    ori=kp_current(6);
    % Define the Gaussian weighing filter
    fil=fspecial('gaussian',[16 16],8);
    tempm=mag_octave{i};
    tempm=tempm(:,:,j);
    tempo=ori_octave{i};
    tempo=tempo(:,:,j);
    [l,b]=size(tempm);
    % Zero-pad the magnitude and orientation matrices
    tempm=[zeros(l,25) tempm zeros(l,25)];
    tempm=[zeros(25,b+50); tempm; zeros(25,b+50)];
    tempo=[zeros(l,25) tempo zeros(l,25)];
    tempo=[zeros(25,b+50); tempo; zeros(25,b+50)];
    % Rotate the matrices by -orientation of the keypoint instead of 
    % rotating window by +orientation for simplicity    
    tempm=imrotate(tempm,-ori);
    tempo=imrotate(tempo,-ori);
    [l,b]=size(tempm);
    % Get 16x16 arrays and weigh the magnitude array with the Gaussian
    tempm=tempm(floor(l/2)-7:floor(l/2)+8,floor(b/2)-7:floor(b/2)+8);
    tempo=tempo(floor(l/2)-7:floor(l/2)+8,floor(b/2)-7:floor(b/2)+8);
    tempm=tempm.*fil;
    % Initialize descriptors
    des=[];
    % Get 4x4 sub-arrays
    for i=1:4:13
        for j=1:4:13
            des_temp=zeros(1,8);
            win_mag=tempm(i:i+3,j:j+3);
            win_ori=tempo(i:i+3,j:j+3);
            for x=1:4
                for y=1:4
                    % See which bin the given orientation corresponds to
                    bin=ceil(win_ori(x,y)/45);
                    % Get the distance weighing factor
                    trilin_wt=abs(win_ori(x,y)-(2*bin-1)*22.5)/45;
                    % Increment the required bin
                    des_temp(bin)=des_temp(bin)+win_mag(x,y)*(1-trilin_wt);
                end
            end
            des=[des des_temp];
        end
    end
    % Normalize the descriptor
    des=des/norm(des,2);
    % Truncate to lie below 0.2
    des(des>0.2)=0.2;
    % Renormalize
    des=des/norm(des,2);
    %add coordinate information for homographies
    cur_octave=i;
%     row=floor(kp_current(3)*(2^(cur_octave-1))+0.5*(2^(cur_octave-1)));
%     col=floor(kp_current(4)*(2^(cur_octave-1))+0.5*(2^(cur_octave-1)));
    row=kp_current(1,3)*2^(kp_current(1,1)-1);
    col=kp_current(1,4)*2^(kp_current(1,1)-1);
    kp_des2(count,:)=[des row col];
end

disp=0;
if disp
    tic;
% For visualization
for i=1:size(kpts,1)
kpdraw(i,1)=kpts(i,3)*2^(kpts(i,1)-1);
kpdraw(i,2)=kpts(i,4)*2^(kpts(i,1)-1);
kpdraw(i,3)=kpts(i,5)*0.5*10^3;
kpdraw(i,4)=kpts(i,6);
end

figure;
imshow(imresize(im_store,2));
hold on
for i=1:size(kpts,1)
plot(gca,[kpdraw(i,2) kpdraw(i,2)+kpdraw(i,3)*sin(kpdraw(i,4)*pi/180)],[kpdraw(i,1) kpdraw(i,1)+kpdraw(i,3)*cos(kpdraw(i,4)*pi/180)]);
end
scatter(kpdraw(:,2),kpdraw(:,1),'g.')
toc;
end

%size1 and size2 are the numbers of keypoints in the descriptors
size1=size(kp_des1, 1);
size2=size(kp_des2, 1);

%dists will store the euclidean distance along with the coordinates of the
%keypoints involved.
dists=[];

%matches will store the matching pairs of keypoints.
matches=[];
for i=1:size1
    dists=[];
    for j=1:size2
        dist=[sqrt(sum((kp_des1(i, 1:128)-kp_des2(j,1:128)).^2)), kp_des1(i, 129:130), kp_des2(j, 129:130)];
        dists=[dists;dist];
    end
    dists=sortrows(dists, 1);
    %size(dists)
    if dists(2,1)==0
        continue
    elseif dists(1,1)/dists(2,1)<match_threshold
        continue
    else
        matches=[matches;dists(1, 2:5)];
    end
end

% %print the keypoint numbers
% figure(1),imshow(im1);
% hold on
% for i=1:size(matches)
%     text(matches(i,2)/2, matches(i,1)/2, num2str(i));
% end
% hold off;
% return

%X is ransac input
X=[];

for i=1:size(matches, 1);
    %need to convert from col_base row_base col_target row_target format
    X=[X [matches(i, 4);matches(i, 3);matches(i, 2); matches(i, 1)]];
end

% number of points
N = size(X, 2);
% inilers percentage
p = 0.25;
% noise
sigma = 1;

% set RANSAC options
options.epsilon = 1e-6;
options.P_inlier = 1-1e-4;
options.sigma = sigma;
options.validateMSS_fun = @validateMSS_homography;
options.est_fun = @estimate_homography;
options.man_fun = @error_homography;
options.mode = 'RANSAC';
options.Ps = [];
options.notify_iters = [];
options.min_iters = 1000;
options.fix_seed = false;
options.reestimate = true;
options.stabilize = false;


%call ransac and transform using its homography
%IT=Image after Transformation.
[results, options] = RANSAC(X, options);
H = (reshape(results.Theta, 3, 3))';
T = maketform('projective', H);
IT = imtransform(im2, T);

%Calculate the offset.
%For this code, I could have used any point. I used the first matching
%point only because it was the first idea that came to mind.
[xout, yout]=tformfwd(T, 0, 0);
offset=[xout, yout];

%account of the initial upsampling
offset=offset/2;
offset=[floor(offset(1,1)), floor(offset(1,2))];

%swap x and y coordinates
offset=[offset(1,2) offset(1,1)];

%Apply the offset, while putting the picture in the middle of a matrix of
%zeroes.
%In other words, we average IT[p2] with im1[p1] into
%result[size(im1)+size(IT)+p1].
%This applies the points that are within im2.
[rows, cols]=size(IT);
[orows, ocols]=size(im1);
padding=[rows+orows, cols+ocols];
result=zeros(padding*3);
for i=1:rows
    for j=1:cols
        color=0;
        %p1 and p2 are (i, j) on picture 1 and picture 2.
        p2=[i j];
        p1=p2+offset;
        if IT(p2(1,1), p2(1,2))==0
            %out of bounds on IT
            %Let the color remain 0 here.
            
            %This might be unneeded.
            %color=im1(p1(1,1), p1(1,2));
        elseif p1(1,1)<1 || p1(1,2)<1
            %out of bounds on im1
            color=IT(p2(1,1), p2(1,2));
        elseif p1(1,1)>orows || p1(1,2)>ocols
            %out of bounds on im1
            color=IT(p2(1,1), p2(1,2));
        else
            %within the bounds of both pictures
            color=0.5*im1(p1(1,1), p1(1,2))+0.5*IT(p2(1,1), p2(1,2));
        end
        %Put color into result[padding + p1]
        result(p1(1,1)+(rows+orows), p1(1,2)+(cols+ocols))=color;
    end
end

%This block of code handles pixels that are in im1 only.
for i=1:orows
    for j=1:ocols
        p1=[i, j];
        color=im1(p1(1,1), p1(1,2));
        %Check if result[padding + p1] is not already processed.
        if result(p1(1,1)+(rows+orows), p1(1,2)+(cols+ocols))==0
            %Put color into result[padding + p1]
            result(p1(1,1)+(rows+orows), p1(1,2)+(cols+ocols)) = color;
        end
    end
end

%Display the resulting image.
figure(1),imshow(result,[]);
% hold on
% plot(cols+ocols+matches(:,2)/2, rows+orows+matches(:,1)/2, 'ro', 'markersize', 5);
% hold off;

%end