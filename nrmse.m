function [NRMSE,RMSE]=nrmse(err,Max,Min)

MSE = mean(err.^2);
RMSE = sqrt(MSE);
% MSE=sum(err(:).^2)/length(err);
% RMSE=sqrt(MSE);
NRMSE=RMSE./(Max-Min);