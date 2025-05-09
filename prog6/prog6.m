


% num = match(image1, image2)
%
% This function reads two images, finds their SIFT features, and
%   displays lines connecting the matched keypoints.  A match is accepted
%   only if its distance is less than distRatio times the distance to the
%   second closest match.
% It returns the number of matches displayed.
%
% Example: match('scene.pgm','book.pgm');

function num = prog6(image1, image2)

%if no args supplied
if nargin<2
    image1=input('Enter filename of image 1:', 's');
    image2=input('Enter filename of image 2:', 's');
end

% Find SIFT keypoints for each image
[im1, des1, loc1] = sift(image1);
[im2, des2, loc2] = sift(image2);

% For efficiency in Matlab, it is cheaper to compute dot products between
%  unit vectors rather than Euclidean distances.  Note that the ratio of 
%  angles (acos of dot products of unit vectors) is a close approximation
%  to the ratio of Euclidean distances for small angles.
%
% distRatio: Only keep matches in which the ratio of vector angles from the
%   nearest to second nearest neighbor is less than distRatio.
distRatio = 0.6;   

% For each descriptor in the first image, select its match to second image.
des2t = des2';                          % Precompute matrix transpose
for i = 1 : size(des1,1)
   dotprods = des1(i,:) * des2t;        % Computes vector of dot products
   [vals,indx] = sort(acos(dotprods));  % Take inverse cosine and sort results

   % Check if nearest neighbor has angle less than distRatio times 2nd.
   if (vals(1) < distRatio * vals(2))
      match(i) = indx(1);
   else
      match(i) = 0;
   end
end

% % Create a new image showing the two images side by side.
% im3 = appendimages(im1,im2);
% 
% % Show a figure with lines joining the accepted matches.
% figure('Position', [100 100 size(im3,2) size(im3,1)]);
% colormap('gray');
% imagesc(im3);
cols1 = size(im1,2);

%X is an addition that is used as from our class section notes.
X=[];
matches=[];
for i = 1: size(des1,1)
  if (match(i) > 0)
%     line([loc1(i,2) loc2(match(i),2)+cols1], ...
%          [loc1(i,1) loc2(match(i),1)], 'Color', 'c');
    %need to convert to col_base row_base col_target row_target format
    x1=loc1(i,2);
    x2=loc2(match(i),2);
    y1=loc1(i,1);
    y2=loc2(match(i),1);
    X=[X, [x2;y2;x1;y1]];
  end
end
num = sum(match > 0);

if num>4
    fprintf('Found %d matches.\n', num);
else
    fprintf('Found an insufficient number of matches for overlaying.');
    return
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
%For this code, I could have used any point. I am using the average of the
%top left and bottom right corners.
point=[1 1];
[xout, yout]=tformfwd(T, point);
offset1=[xout, yout]-point;
point=[size(image2, 2) size(image2, 1)];
[xout, yout]=tformfwd(T, point);
offset2=[xout, yout]-point;

%Average the offsets, make them discrete, and swap the row and col
%coordinates.
offset=0.5*(offset1+offset2);
offset=[floor(offset(1,1)), floor(offset(1,2))];
offset=[offset(1,2) offset(1,1)];

%Subtract 100 to the y coordinate of the offset to correct for something in
%lowe's code.
offset=offset-[100, 0];

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

[cropped,t,b,l,r] = imtrim(result);

%Display the resulting image.
figure(1),imshow(cropped,[]);
% hold on
% plot(cols+ocols+matches(:,2)/2, rows+orows+matches(:,1)/2, 'ro', 'markersize', 5);
% hold off;

end

%source: http://www.alecjacobson.com/weblog/?p=1479
function [out1,out2,out3,out4,out5] = imtrim(im,location)
  %IMTRIM auto-crop an image like Photoshop's Edit>Trim feature, as of yet only
  %grayscale imagse are supported
  %
  %   cropped = IMTRIM(IM) crop image based on top left corner
  %
  %   [cropped,t,b,l,r] = IMTRIM(IM) return cropped image and indices used to
  %   crop the image. So cropped = im(t:b,l:r);
  %
  %   [t,b,l,r] = IMTRIM(IM) return only indices used to crop the image. So
  %   cropped = im(t:b,l:r);
  %
  %   [...] = IMTRIM(IM,location) same as above but location may specify
  %   top-left corner ('NorthWest') or bottom-right corner ('SouthEast') to be
  %   the picel used in determining the auto-crop
  %
  %   Copyright Alec Jacobson, 2010
  %

  if(~exist('location'))
    location = 'NorthWest';
  end

  % gather corner value to which the image is compared
  if(strcmp(location,'NorthWest'))
    corner_value = im(1,1);
  elseif(strcmp(location,'SouthEast'))
    corner_value = im(1,1);
  else
    error([location ' is not a valid location']);
  end

  % hard-coded threshold parameter, works equivalently with Photoshop's
  % hardcoded parameter
  threshold = 0.1;

  % get difference of image with corner value
  %difference = abs(im - corner_value)>0.1;
  % should work for any number of channels
  difference = sqrt(sum((im - corner_value).^2,3)) > ...
    sqrt(threshold^2*size(im,3));
  [left_i,left_j] = ind2sub(size(difference),find(difference,1));
  [right_i,right_j] = ind2sub(size(difference),find(difference,1,'last'));
  [top_j,top_i] = ind2sub(size(difference'),find(difference',1));
  [bottom_j,bottom_i] = ind2sub(size(difference'),find(difference',1,'last'));
  if(nargout == 1)
    out1 = im(top_i:bottom_i,left_j:right_j);
  elseif(nargout == 5)
    out1 = im(top_i:bottom_i,left_j:right_j);
    out2 = top_i;
    out3 = bottom_i;
    out4 = left_j;
    out5 = right_j;
  else
    out1 = top_i;
    out2 = bottom_i;
    out3 = left_j;
    out4 = right_j;
  end
end
