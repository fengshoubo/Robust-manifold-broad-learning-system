function [X_train_pre,X_test_pre] = bls_train_feng_timeseries(train_x,train_y,test_x,test_y,ps_n,max_min,C,N_feature,N_window,N_enhance,i_alpha,i_beta,iterNum_ADMM, Amplitude_Noise)
% Learning Process of the proposed broad learning system 增量式学习
%Input:
%---train_x,test_x : the training data and learning data
%---train_y,test_y : the label
%---We: the randomly generated coefficients of feature nodes
%---wh:the randomly generated coefficients of enhancement nodes
%----s: the shrinkage parameter for enhancement nodes
%----C: the regularization parameter for sparse regualarization
%----N1: the number of feature nodes  per window
%----N2: the number of windows of feature nodes
%----N3: 基本层的 enhancement nodes 的节点数
%----N4: additional enhancement nodes 的节点数
%----N5: the layer number of additional enhancement nodes
% ---m1:number of widow of feature nodes in the increment step
%----m2:number of enhancement nodes related to the incremental feature nodes per increment step
%%%%%%%%%%%%%%feature nodes%%%%%%%%%%%%%%
%    Compute W
tic;
% rand('seed',0);
H1 = train_x;        % 输入 24300*2049 往外扩0.1
[n_sam_train, n_input]=size(train_x);
T1_o=zeros(n_sam_train,N_window*N_feature);
for i_W=1:N_window
    [G_W{i_W}]=Random_Manifold_L21(train_x,N_feature,i_alpha,i_beta,iterNum_ADMM, Amplitude_Noise);
%     if n_input>=N_feature
%         G_W{i_W}=orth(2*rand(n_input,N_feature)-1);
%     else
%         G_W{i_W}=orth(2*rand(n_input,N_feature)'-1)';
%     end
%     we=2*rand(n_input,N_feature)-1;               %2049*100
%     A1 = H1 * we;
%     A1 = mapminmax(A1);
%     G_W{i_W} = sparse_bls(A1,H1,1e-4,50)';
    % training stage
    G_W{i_W}=G_W{i_W}/max(max(G_W{i_W}));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% mapped feature %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    T1=H1*G_W{i_W};
    [T1,ps_f{i_W}]=mapminmax(T1');
    T1_o(:,((i_W-1)*N_feature+1):i_W*N_feature)=T1';
    
end
clear H1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% mapped feature end %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% output:
% y: mapped feature 的表征
% beta11：mapped feature 的随机映射参数，测试时使用
% ps: 输出表征T1_o 的缩放参数
%%%%%%%%%%%%% enhancement nodes layer 1 H2 T2  start %%%%%%%%%%%%%%%%%%%%%%%%%%%%
H2 = [T1_o 0.1*ones(size(T1_o,1),1)];
if N_feature*N_window>=N_enhance
    wh2=orth(2*rand(N_window*N_feature+1,N_enhance)-1);
else
    wh2=orth(2*rand(N_window*N_feature+1,N_enhance)'-1)';
end
T2 = H2 *wh2;
[T2,ps_e]=mapminmax(T2');
T2=T2';
fprintf(1,'Enhancement nodes: Max Val of Output %f Min Val %f\n',max(max(T2)),min(T2(:)));
T2_o = tansig(T2 );  %
clear T2；clear H2;
%%%%%%%%%%%%% enhancement nodes layer 1 H2 T2  end %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%output:
%T2_o: enhancement feature layer 1 的表征
%wh2: mapped feature 到 enhancement nodes 的随机映射参数
%l2: enhancement nodes layer 1 的缩放系数

%%%%%%%%%%%%%%%%%%%%Compute the output weights%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T=[train_x T1_o T2_o];
clear T1_o; clear T2_o; clear T_addition_enhance;
% T_=gpuArray(T);
% C_=gpuArray(C);
% train_y_=gpuArray(train_y);
%%%%%%%%%%%%%%    cpu 计算 %%%%%%%%%%%%%%
beta = (T'  *  T+eye(size(T',1)) * (C)) \ ( T');
beta2=beta*train_y;
clear beta;
% beta2=gather(beta2_);
%%%%%%%%%%%%    gpu 计算 %%%%%%%%%%%%%%%%%%
% I_gpu=C*eye(size(T',1),'single','gpuArray') ;
% T_gpu=gpuArray(single(T));
% beta_gpu_part1=(T_gpu'  *  T_gpu+I_gpu);
% clear I_gpu;
% tic;
% wait(gpuDevice);
% beta_gpu = beta_gpu_part1 \ ( T_gpu');
% toc;
% beta=gather(beta_gpu);
% beta2=beta*train_y;
% clear beta ;
% clear T_gpu;
% clear beta_gpu;
% clear beta_gpu_part1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Training_time = toc;
disp('Training has been finished!');
disp(['The Total Training Time is : ', num2str(Training_time), ' seconds' ]);
%%%%%%%%%%%%%%%%%Training Accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%
X_train_pre = T * beta2;
X_train_pre=mapminmax('reverse',X_train_pre',ps_n)';
clear T;
%%%%%%%%%%%%%%%%%%%%%%Testing Process%%%%%%%%%%%%%%%%%%%
HH1 = test_x ;
%clear test_x;
%------------ mapped feature for testing---------
[n_sam_test,~]=size(test_x);
yy_o=zeros(n_sam_test,N_window*N_feature);
for i_W=1:N_window
    yy=HH1*G_W{i_W};
    yy=mapminmax('apply',yy',ps_f{i_W})';
    yy_o(:,((i_W-1)*N_feature+1):i_W*N_feature)=yy;
end
%     TT_random=tansig(HH1*wh_random);
clear TT1;clear HH1;
%------------enhancement nodes layer 1 for testing-------------------
HH2 = [yy_o  0.1 * ones(size(yy_o,1),1)];
TT2 = HH2 * wh2 ;
TT2 = mapminmax('apply',TT2',ps_e)';
TT2_o=tansig(TT2 );  %
clear TT2; clear HH2; clear wh2;
TT=[test_x  yy_o TT2_o ]; %隐含层串接矩阵
clear T_addition_enhance; clear HH; clear l; clear wh; clear yy1;clear TT2_o;
%%%%%%%%%%%%%%%%% testing accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%%
X_test_pre = TT * beta2;
clear TT;
X_test_pre=mapminmax('reverse',X_test_pre',ps_n)';