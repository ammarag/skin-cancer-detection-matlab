function reduced_features_index=PCA(features)

features(isnan(features))=0;
[m,n] = size(features);

% Remove the mean
features = features - repmat(mean(features,2), 1, size(features,2));

% Compute the SVD
[U,S,V] =svd(features);

% Compute the number of eigenvectors representing
% the 95% of the variation
coverage = cumsum(diag(S));
coverage = coverage ./ max(coverage);
[~, nEig] = max(coverage > 0.95);

% Compute the norms of each vector in the new space
norms = zeros(n,1);
for i = 1:n
    norms(i) = norm(V(i,1:nEig))^2;
end
% Get the largest 2
[~, reduced_features_index] = sort(norms);

