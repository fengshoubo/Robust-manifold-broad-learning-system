function [X,Max, Min]=normalize(X)
Max=max(X);
Min=min(X);
X_min_reshape=repmat(min(X),length(X),1);
max_min=max(X)-min(X);
max_min_reshape=repmat(max_min,length(X),1);
X=(X-X_min_reshape)./max_min_reshape;



end

