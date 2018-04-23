%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File to create training and validation set       %
% for ShanghaiTech Dataset Part A and B. 10% of    %
% the training set is set aside for validation     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc; clear all;
seed = 95461354;
rng(seed)
N = 9;
dataset = 'A';
%dataset = 'B';
dataset_name = ['shanghaitech_part_' dataset '_patches_' num2str(N)];
path = ['../data/original/shanghaitech/part_' dataset '_final/train_data/images/'];
output_path = '../data/formatted_trainval/';
train_path_img = strcat(output_path, dataset_name,'/mscnn_train/');
train_path_den = strcat(output_path, dataset_name,'/mscnn_train_den/');
val_path_img = strcat(output_path, dataset_name,'/mscnn_val/');
val_path_den = strcat(output_path, dataset_name,'/mscnn_val_den/');
gt_path = ['../data/original/shanghaitech/part_' dataset '_final/train_data/ground_truth/'];

mkdir(output_path)
mkdir(train_path_img);
mkdir(train_path_den);
mkdir(val_path_img);
mkdir(val_path_den);
if (dataset == 'A')
    num_images = 300;
else
    num_images = 400;
end
num_val = ceil(num_images*0.1);
indices = randperm(num_images);

for idx = 1:num_images
    i = indices(idx);
    if (mod(idx,10)==0)
        fprintf(1,'Processing %3d/%d files\n', idx, num_images);
    end
    load(strcat(gt_path, 'GT_IMG_',num2str(i),'.mat')) ;
    input_img_name = strcat(path,'IMG_',num2str(i),'.jpg');
    im = imread(input_img_name);
    [h, w, c] = size(im);
    if (c == 3)
        im = rgb2gray(im);
    end
    scale = 0.9;
    wn2 = w * scale; hn2 = h * scale;
    %wn2 =8 * floor(wn2/8);
    %hn2 =8 * floor(hn2/8);
    w_s = int32(floor((w - wn2) / 2));
    h_s = int32(floor((h - hn2) / 2));
    annPoints =  image_info{1}.location;
    
%     if( w <= 2*wn2 )
%         im = imresize(im,[ h,2*wn2+1]);
%         annPoints(:,1) = annPoints(:,1)*2*wn2/w;
%     end
%     if( h <= 2*hn2)
%         im = imresize(im,[2*hn2+1,w]);
%         annPoints(:,2) = annPoints(:,2)*2*hn2/h;
%     end
%     [h, w, c] = size(im);
%     
    im_density = get_density_map_gaussian(im,annPoints);
    
    for w_i = 0:2
        for h_i = 0:2
            %w_i表示x1增加多少倍的0.05x的大小，起始为0
            %h_i表示y1增加多少倍的0.05y的大小，起始为0
            x1 = 1 + w_i * w_s;
            y1 = 1 + h_i * h_s;

            im_sampled = im(y1:y1+hn2-1, x1:x1+wn2-1,:);
            im_density_sampled = im_density(y1:y1+hn2-1, x1:x1+wn2-1);

            img_idx = strcat(num2str(i), '_',num2str(w_i*3+h_i+1));        

            if(idx < num_val)
                imwrite(im_sampled, [val_path_img num2str(img_idx) '.jpg']);
                csvwrite([val_path_den num2str(img_idx) '.csv'], im_density_sampled);
            else
                imwrite(im_sampled, [train_path_img num2str(img_idx) '.jpg']);
                csvwrite([train_path_den num2str(img_idx) '.csv'], im_density_sampled);
            end
        end
    end
    
end

