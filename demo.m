clear
load toy

data = toy(:,1:2);
l = toy(:,3);

% Build the model with B = 0
mdl = MMAD(data, ones(size(data,1),1), 0, 200);

% Plot the decision boundary
x = min(data(:,1)):0.05:max(data(:,1));
y = min(data(:,2)):0.05:max(data(:,2));
[X,Y] = meshgrid(x,y);
Z = exp(-mdl.gamma*pdist2([X(:) Y(:)],mdl.RS).^2)*mdl.W';
[c1,h1] = contour(X,Y,reshape(Z,size(X)),'LevelStep',0.05);
clabel(c1,h1);
hold on
ix = l == 1;
scatter(toy(ix,1),toy(ix,2),200,'.')
scatter(toy(~ix,1),toy(~ix,2),100,'+')