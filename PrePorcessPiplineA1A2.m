clc;
clear;
close all;

empty_dataName_list1 = {'54_gt.dat'};
intruded_dataName_list1 = {'45_dyp_1.dat'	'45_dyp_2.dat'	'45_dyp_3.dat'	'45_dyp_4.dat'   '54_dyp_1.dat'	'54_dyp_2.dat'	'54_dyp_3.dat' '54_dyp_4.dat'};
intruded_dataName_list2 = {'45_lyj_1.dat'	'45_lyj_2.dat'	'45_lyj_3.dat' '45_lyj_4.dat' '54_lyj_1.dat'	'54_lyj_2.dat'	'54_lyj_3.dat' '54_lyj_4.dat' };
all_dataName_list = [empty_dataName_list1,  ...
                     intruded_dataName_list1,intruded_dataName_list2];
%% 归一化值
CSI = {};
for dataName = all_dataName_list
    csi = read_bf_file(dataName{:});
    csi = copy_empty_csi_v2(csi);
    CSI=[CSI;csi];
end
amp = get_amplitude(CSI);
amp = amp_hampel_v2(amp,20);
amp = amp_DWT(amp);
min_val = min(amp(:));
max_val = max(amp(:));

%% 空房间

% PA,TX,RX,SC

amp_cut_all = [];
for dataName = empty_dataName_list1

    % CSI
    csi = read_bf_file(dataName{:});
    csi = copy_empty_csi_v2(csi);% 补全缺失天线，tx 1缺1
    csi = scale_csi_5300(csi,"5300");% 以绝对单位计算CSI

    % AMP
    amp = get_amplitude(csi);
    amp = amp_hampel_v2(amp,20);
    amp = amp_DWT(amp);
    amp = normalize_data_new(amp,min_val,max_val);

    % window length 6
    n_timestamps = 6;
    step_size = 1;
    amp_cut = get_window(amp(:,:,:,:),n_timestamps,step_size);
    amp_cut_all = cat(1,amp_cut_all,amp_cut);
end
num_samples = size(amp_cut_all, 1);
num_subset = floor(num_samples / 3);
empty_amp1 = amp_cut_all(1:num_subset, :, :, :, :);
empty_amp2 = amp_cut_all(num_subset:2*num_subset, :, :, :, :);


amp_cut_all = [];
for dataName = intruded_dataName_list1
    % CSI
    csi = read_bf_file(dataName{:});
    csi = copy_empty_csi_v2(csi);
    csi = scale_csi_5300(csi,"5300");

    % AMP
    amp = get_amplitude(csi);
    amp = amp_hampel_v2(amp,20);
    amp = amp_DWT(amp);
    amp = normalize_data_new(amp,min_val,max_val);

    % window length 6
    n_timestamps = 6;
    step_size = 1;
    amp_cut = get_window(amp(:,:,:,:),n_timestamps,step_size);
    amp_cut_all = cat(1,amp_cut_all,amp_cut);
end
intruded_amp1 = amp_cut_all;


amp_cut_all = [];
for dataName = intruded_dataName_list2
    % CSI
    csi = read_bf_file(dataName{:});
    csi = copy_empty_csi_v2(csi);
    csi = scale_csi_5300(csi,"5300");

    % AMP
    amp = get_amplitude(csi);
    amp = amp_hampel_v2(amp,20);
    amp = amp_DWT(amp);
    amp = normalize_data_new(amp,min_val,max_val);

    % window length 6
    n_timestamps = 6;
    step_size = 1;
    amp_cut = get_window(amp(:,:,:,:),n_timestamps,step_size);
    amp_cut_all = cat(1,amp_cut_all,amp_cut);
end
intruded_amp2 = amp_cut_all;

save('A1A2amp.mat','empty_amp1','empty_amp2','intruded_amp1','intruded_amp2');

%% 压缩
load('A1A2amp.mat')

% empty C1
lambda_S = 0.003594;
lambda_G = 0.001000;
lambda_N = 10;
R = 15;
sz = size(empty_amp1);
empty_amp1 = reshape(empty_amp1, [sz(1),sz(2),sz(3)*sz(4),sz(5)]);
[X,G,C1,relChgX_tmp]=TSL4CSI(empty_amp1, lambda_S,lambda_G,lambda_N,R); 

% intruded C2
lambda_S = 0.001292;
lambda_G = 0.001000;
lambda_N = 4.641589;
R = 15;
sz = size(intruded_amp1);
intruded_amp1 = reshape(intruded_amp1, [sz(1),sz(2),sz(3)*sz(4),sz(5)]);
[X,G,C2,relChgX_tmp]=TSL4CSI(intruded_amp1, lambda_S,lambda_G,lambda_N,R); 

%%convert
sz = size(empty_amp2);
empty_amp2 = reshape(empty_amp2, [sz(1),sz(2),sz(3)*sz(4),sz(5)]);
sz = size(intruded_amp2);
intruded_amp2 = reshape(intruded_amp2, [sz(1),sz(2),sz(3)*sz(4),sz(5)]);


train_empty_1 = empty_amp1-tmprod(tmprod(empty_amp1,C1',4),C1,4);
train_empty_2 = empty_amp1-tmprod(tmprod(empty_amp1,C2',4),C2,4);

test_empty_1 = empty_amp2-tmprod(tmprod(empty_amp2,C1',4),C1,4);
test_empty_2 = empty_amp2-tmprod(tmprod(empty_amp2,C2',4),C2,4);

train_intruded_1 = intruded_amp1-tmprod(tmprod(intruded_amp1,C1',4),C1,4);
train_intruded_2 = intruded_amp1-tmprod(tmprod(intruded_amp1,C2',4),C2,4);

test_intruded_1 = intruded_amp2-tmprod(tmprod(intruded_amp2,C1',4),C1,4);
test_intruded_2 = intruded_amp2-tmprod(tmprod(intruded_amp2,C2',4),C2,4);

save('A1A2dataAB.mat','train_empty_1','train_empty_2','test_empty_1','test_empty_2','train_intruded_1','train_intruded_2','test_intruded_1', 'test_intruded_2')

	
%% 制作提示词
load('A1A2amp.mat')

%
A = empty_amp1;
A = reshape(A,[],6*2*3*30);
min_vals = min(A, [], 2);
max_vals = max(A, [], 2);
mean_vals = mean(A, 2);
median_vals = median(A, 2);
var_vals = var(A, 0, 2);
e1_result = [min_vals, max_vals, mean_vals, median_vals, var_vals];


A = empty_amp2;
A = reshape(A,[],6*2*3*30);
min_vals = min(A, [], 2);
max_vals = max(A, [], 2);
mean_vals = mean(A, 2);
median_vals = median(A, 2);
var_vals = var(A, 0, 2);
e2_result = [min_vals, max_vals, mean_vals, median_vals, var_vals];

A = intruded_amp1;
A = reshape(A,[],6*2*3*30);
min_vals = min(A, [], 2);
max_vals = max(A, [], 2);
mean_vals = mean(A, 2);
median_vals = median(A, 2);
var_vals = var(A, 0, 2);
i1_result = [min_vals, max_vals, mean_vals, median_vals, var_vals];

A = intruded_amp2;
A = reshape(A,[],6*2*3*30);
min_vals = min(A, [], 2);
max_vals = max(A, [], 2);
mean_vals = mean(A, 2);
median_vals = median(A, 2);
var_vals = var(A, 0, 2);
i2_result = [min_vals, max_vals, mean_vals, median_vals, var_vals];


% 合并所有数据
data_all = [e1_result; e2_result; i1_result; i2_result;];

[e1_result, e2_result, i1_result,  i2_result] = normalize_columns(e1_result, e2_result, i1_result, i2_result);
function [e1_norm,  e2_norm, i1_norm, i2_norm] = normalize_columns(e1_result, e2_result, i1_result,  i2_result)

    combined_data = [e1_result;  e2_result; i1_result;  i2_result];

    col_max = max(combined_data);
    col_min = min(combined_data);
    normalize = @(x) (x - col_min) ./ (col_max - col_min);

    e1_norm = normalize(e1_result);
    e2_norm = normalize(e2_result);
    i1_norm = normalize(i1_result);
    i2_norm = normalize(i2_result);
end

save('A1A2_promptAB.mat','e1_result','e2_result','i1_result','i2_result');
