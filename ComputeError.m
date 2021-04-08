function [RMSE,NRMSE,MAE,SMAPE,NJPE]=ComputeError(X,X_pre,max_min)
    Length=size(X,1);
    
    RMSE = sqrt((sum((X-X_pre).^2))/Length);
    
    NRMSE= RMSE./max_min;
    
    MAE=sum(abs(X-X_pre))/Length;
    
    SMAPE=2*sum(((abs(X-X_pre))./(abs(X)+abs(X_pre))))/Length;
    
    NJPE= sqrt(sum(sum(((X-X_pre)./max_min).^2))/Length);
    
end