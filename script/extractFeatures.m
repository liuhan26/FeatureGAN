function feature = extractFeatures(model,weights,path,is_imresize)
%root_dir = '/media/liuhan/xiangziBRL/Lock3DFace/TestImage';
%root_dir='/media/scw4750/cicilu/Realsense_station/wholeface_64_64/train/crop_image_realsense_128_128';
fid=fopen(path,'r');
addpath(genpath('/home/liuhan/caffe-master/matlab'));
use_gpu=true;
if use_gpu
    caffe.set_mode_gpu();
    caffe.set_device(0);
else
    caffe.set_mode_cpu();
end

net=caffe.Net(model,weights,'test');
f1_struct=[];

i=1;
while ~feof(fid)
    file= fgetl(fid);
    if exist('root_dir','var')
        img = imread([root_dir filesep file(1:end-2)]);
    else
        img = imread(file(1:end-2));
    end
    
    if is_imresize
        img=imresize(img,[128 128]);
    end
    
    img=img';
    
    data = zeros(128,128,1,1);
    data = single(data);
    data(:,:,:,1) = (single(img)/255);
    
    net.blobs('data').set_data(data);
    net.forward_prefilled();
    ip_f1=net.blobs('eltwise_fc1').get_data();
    ip_f1=ip_f1(:);
    f1_struct = [f1_struct ip_f1];   
    i=i+1
end
fclose(fid);
feature = f1_struct';