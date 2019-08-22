function mdl = MMAD(X, Y, B, n_S)

% By Zahra Ghafoori V1 2017

% MaxiMin based Anomaly Detection (MMAD)
% Inputs:
%   X: n-by-d matrix of input data; n = #records and d = #dimensions
%       X should be normalised in range [0, 1] to choose a proper gamma
%   Y: n-by-1 label vector
%       The labels are only used if B~=0, when active learning is used
%       MMAD technique is unsupervised and does not need labels
%       If B = 0, pass an empty vector [] for Y
%   B: active learning budget
%   n_S: size of the random sample set S 
% Returns:
%   mdl: one-class classification model of MMAD


% Fix rng seed
rng(0)

% Uniform Random Sampling
[S,ind] = datasample(X,n_S,'Replace',false);
if B, y_S = Y(ind); end % Y is used only with active learning if B ~= 0

% Computing pairwise distances in S
D = pdist2(S,S);

% MM sampling
[~,idx] = min(pdist2(median(S,1),S));
[~,I_CRS(1)] = max(D(idx,:));
maximinD = D(I_CRS(1),:);
for i = 2:floor(0.4*n_S)
    [~,I_CRS(i)] = max(maximinD);
    maximinD = bsxfun(@min,maximinD,D(I_CRS(i),:));
    % Computing the Silhouette index 
    [~,U] = min(D(I_CRS,:),[],1);
    U = I_CRS(U);
    % Compute Silhouette index for current MMs
    silh(i-1) = mean(computeSilhouette(D,U));
end

% Reduce to RS set using the index that maximises silh
[~,maxI] = max(silh);
I_RS = I_CRS(1:maxI+1);

% Substitute each of RSs with closest point to their cluster mean
% % If belonging to the training set is not mandatory for RSs, 
% % this step can be modified so that RSs are substitute by their cluster mean 
[~,U] = min(D(I_RS,:),[],1);
for i = 1:length(I_RS)
    tmpRS(i,:) = mean(S(U == i,:),1);
end
[~,I_RS] = min(pdist2(S,tmpRS),[],1);

% Finding the dataset type (WS or NWS) to screen anomalies and set gamma
[~,U] = min(D(I_RS,:),[],1);
U = I_RS(U)';
[I_RS_sorted,idx] = sort(I_RS);
n_c = histc(U,I_RS_sorted);
n_c(idx) = n_c;
NWS = maxI+1 == length(I_CRS);
if ~NWS
    [n_c_sorted,idx_sorted] = sort(n_c);
    % Find index in which n_c is halfed or reduced more
    idx = find(n_c_sorted(2:end)./n_c_sorted(1:end-1) > 2);
    if idx, idx = idx(end); end
    I_RS(idx_sorted(1:idx)) = [];
    n_c(idx_sorted(1:idx)) = [];
    %Paul
    gammaV = 2.^(-6:6);
    idxx = find(triu(ones(size(S,1)),1));
    D2 = D.^2;
    rbfKernel = @(gamma,D2) exp(-gamma.*D2);
    for i = 1:length(gammaV)
        gamma = gammaV(i);
        K = rbfKernel(gamma,D2);
        o(i) = std(K(idxx))^2/(mean(K(idxx))+eps);
    end
    [~,i] = max(o);
    mdl.gamma = gammaV(i);
else
    tmp = D; tmp(tmp == 0) = []; tmp = tmp(:);
    mdl.gamma = -log(max(tmp)/min(tmp))/(min(tmp)^2-max(tmp)^2);
end
    
% Budget spending
if B == 0
    mdl.RS = S(I_RS,:);
    w = n_c./sum(n_c);
    w = w./sum(w);
    mdl.W = w';
else
    y = ones(length(I_RS),1);
    if B < length(I_RS)
        I_RS_tmp = I_RS(ismember(I_RS,I_RS));
        
        while and(B,~isempty(I_RS_tmp))
            I_toAsk = I_RS_tmp(1);
            I_RS_tmp(1) = [];
            y(I_toAsk ==I_RS) = y_S(I_toAsk);
            % Remove RSs close to anomalous RSs
            if(y_S(I_toAsk) == -1)
                th = 0.5;
                I_toRemove = find(exp(-mdl.gamma.*D(I_toAsk,I_RS).^2) > th);
                y(I_toRemove) = -1;
                I_RS_tmp(ismember(I_RS_tmp,I_toRemove)) = [];
            end
            B = B - 1;
        end
    else
        y = y_S(I_RS);
    end
    I_RS = I_RS(y == 1);
    n_c = n_c(y == 1);
    mdl.RS = S(I_RS,:);
    w = n_c./sum(n_c);
    w = w./sum(w);
    mdl.W = w';  
end

end


function silh = computeSilhouette(D, U)

% Function to compute Silhouette when a distance matrix is given
% Inputs:
%   D: squared distance matrix
%   U: cluster memberships

[idx,cnames] = grp2idx(U);
n = length(idx);
k = length(cnames);
count = histc(idx(:)',1:k);

% Create a list of members for each cluster
mbrs = (repmat(1:k,n,1) == repmat(idx,1,k));

myinf = zeros(1,1,class(D));
myinf(1) = Inf;
avgDWithin = repmat(myinf, n, 1);
avgDBetween = repmat(myinf, n, k);
for j = 1:n
    distj = D(j,:).^2;
    % Compute average distance by cluster number
    for i = 1:k
        if i == idx(j)
            avgDWithin(j) = sum(distj(mbrs(:,i))) ./ max(count(i)-1, 1);
        else
            avgDBetween(j,i) = sum(distj(mbrs(:,i))) ./ count(i);
        end
    end
end

% Calculate the silhouette values
minavgDBetween = min(avgDBetween, [], 2);
silh = (minavgDBetween - avgDWithin) ./ max(avgDWithin,minavgDBetween);
end









