function C = digit_classify( testdata )
%DIGIT_CLASSIFY Classify 3D digit data.
%   C = DIGIT_CLASSIFY(TESTDATA) is the predicted label of given testdata.
%   TESTDATA is expected to be Nx3 matrix, although matrices with arbitrary
%   second dimensions are also supported.

    model_path = "data/final_knn_model.mat";
    load(model_path, 'knn_model');
    
    trainData = knn_model.data;
    trainLabels = knn_model.labels;
    k = knn_model.k;

    % Pre-processing
    testdata = preprocess_data( testdata );

    % Classifying
    C = knn_classify( trainData, trainLabels, testdata, k );
end
