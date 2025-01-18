function data_preprocessed = preprocess_data( sample, N_interp )
% PREPROCESS_DATA Preprocess a data sample 
%   sample - data sampe of size N x D (in our data, D = 3) 
%   N_interp - length to which interpolate each dimension

    % If no interpolation length given
    if nargin < 2
        N_interp = 128;
    end

    data_interpolated = interpolate_data( sample, N_interp );

    data_preprocessed = normalize( data_interpolated );
end

function data_preprocessed = interpolate_data( data, N_interp )
% INTERPOLATE_DATA Interpolate data sample to desired length

    NData = size( data, 1 );
    dim = size( data, 2 );
    
    % Treat data as parametrized w.r.t time
    tt = linspace( 0, 1, NData );
    tt_interp = linspace( 0, 1, N_interp );
    
    data_preprocessed = zeros( N_interp, dim );
    for ii = 1:dim
        interp_dim = interp1( tt, data( :, ii ), tt_interp );
        data_preprocessed( :, ii ) = interp_dim;
    end
end


