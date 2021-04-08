function [Q]=Random_Manifold_L21(X_i,N_feature,alpha,beta,iterNum_ADMM, Amplitude_Noise)
tic;
% rand('seed',0);
[mappedX, mapping] = laplacian_eigen(X_i(1:5000,:), N_feature);   % Need Laplacian eigen toolbox 
X=X_i(mapping.conn_comp,:);
L=mapping.L;
D=mapping.D;
[N_sam,N_dim]=size(X);
F=Amplitude_Noise*randn(N_sam,N_feature);
rou=1;
%% initialization
Lambda1=zeros(N_dim,N_feature);
Lambda2=zeros(N_sam,N_feature);
Q=rand(N_dim,N_feature);
P=rand(N_sam,N_feature);
P=orth(P);
W=rand(N_dim,N_feature);
D_12=sqrt(D);
I=eye(N_dim,N_dim);
W_part1=(2*X'*L*X+2*alpha*(X'*X)+rou*I+rou*(X'*D*X))^(-1);
W_part2=2*alpha*X'*F;
W_part3=rou*X'*D_12;
%% iteration
for i_ADMM=1:iterNum_ADMM
    %F=0.01*X*rand(N_dim,N_feature);
    %update W
    W=W_part1*(W_part2+rou*(Q+(Lambda1/rou))+W_part3*(P+(Lambda2/rou)));
    T=Row_Norm(W-(Lambda1/rou));
    %update Q with L21
    Q_part1 = max(T-(beta/rou),0);
    Q=(Q_part1./(Q_part1+(beta/rou))).*(W-(Lambda1/rou));
    % OR update Q with proximal gradient method
    A=D_12*X*W-(Lambda2/rou);
    %update P with SVD
%         [P_U,P_Sigma,P_V]=svd(A);
%         I_mn=eye(size(P_U,2),size(P_V,1));
%         P=P_U*I_mn*P_V';
    % OR update P with modified SVD 
%     [P_U,P_Sigma,~]=svd(A'*A);
%     P_Sigma_12=sqrt(P_Sigma);
%     P=A*P_U/P_Sigma_12*P_U';
    % OR update P with Caley feasible method
    opts.mxitr  = 30;
    opts.xtol = 1e-7;
    opts.gtol = 1e-7;
    opts.ftol = 1e-9;
    [P, out]= OptStiefelGBB(P, @fun,opts,A);
    %update Lambda1 Lambda2 rou
    Lambda1=Lambda1+rou*(Q-W);
    Lambda2=Lambda2+rou*(P-D_12*X*W);
    % record the information for debug
    norm_W(i_ADMM)=norm(W,'fro');
    norm_Q(i_ADMM)=norm(Q,'fro');
    norm_Lambda1(i_ADMM)=norm(Lambda1,'fro');
    norm_Lambda2(i_ADMM)=norm(Lambda2,'fro');
    norm_P_Delta(i_ADMM)=norm(P'*P-eye(N_feature),'fro');
    norm_obj_func(i_ADMM)= beta*sum(sqrt(sum(Q.^2,2)));%   alpha*norm(X*W-F,'fro') ;%trace(W'*X'*L*X*W)%+ 
         %+beta*sum(sqrt(sum(Q.^2,2)));  %+(rou/2)*norm(Q-W+(Lambda1/rou), 'fro')...
         %+(rou/2)*norm(P-D_12*X*W+(Lambda2/rou),'fro');
end
%% output
% figure();
% suptitle(['alpha=',num2str(alpha),'beta=',num2str(beta)]);
% subplot(3,3,1)
% plot(norm_W);
% title('norm W');
% subplot(3,3,2)
% plot(norm_Q);
% title('norm Q');
% subplot(3,3,3)
% plot(norm_Lambda1);
% title('norm Lambda1');
% subplot(3,3,4)
% plot(norm_Lambda2);
% title('norm Lambda2');
% subplot(3,3,5)
% plot(norm_P_Delta);
% title('norm P Delta');
% subplot(3,3,6)
% plot(norm_obj_func);
% title('norm obj func');
% disp(['the time is'])
% toc;
end

function [Function, Grandient] = fun(P, A)
Grandient = P - A;
Function = 0.5 * norm(P-A,'fro');
end

function [X_Row_norm]=Row_Norm(X)
[m,n]=size(X);
X_Row_norm=zeros(m,1);
for i_norm=1:m
    X_Row_norm(i_norm)=norm(X(i_norm,:),2);
end
end