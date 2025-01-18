function D = knn_classify( trainData, trainLabels, sample, k )
    % This function returns the most frequent digit label for a given
    % sample based on the given k-nearest neighbours model.
    %
    % Syntax:
    %   D = knn_classify(sample, model_path)
    %
    % Inputs:
    %   trainData - Reference data used in the classification
    %   trainLabels = Labes of the reference data
    %   sample - Numeric matrix of the same size as the references in
    %       the given model
    %   k - number of nearest neighbours to use in classification
    %
    % Outputs:
    %   D - Predicted digit label as double

    % Check compatible sizes between train and test
    dim_train = size( trainData, 2 );
    dim_test = size( sample, 2 );
    if dim_test < dim_train
        trainData = trainData( :, 1:dim_test );
    elseif dim_test > dim_train
        sample = sample( :, 1:dim__train );
    end
    
    % Compute distances to all reference samples
    num_references = size( trainData, 3 );
    distances = zeros( num_references, 1 );
    for i = 1:num_references
        reference_sample = trainData(:, :, i);
        % Calculate euclidean distance
        % size( sample )
        % size( reference_sample )
        distances(i) = sum(vecnorm(sample - reference_sample, 2, 2));
    end
    
    % Find the k nearest neighbors
    [~, sorted_indices] = sort(distances);
    nearest_labels = trainLabels( sorted_indices(1:k) );
    
    % Get the most frequent label
    D = mode(nearest_labels);
end