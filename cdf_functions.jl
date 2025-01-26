# function empcdf(data::AbstractArray; nx=100::Int, x=nothing)
#     if isnothing(x)
#         m = minimum(data)
#         M = maximum(data)
#         x = range(m, stop=M, length=nx)
#     end

#     ndata = length(data)
#     cdf_emp = zeros( Float64, (1, nx) )

#     for i in 1:nx
#         cdf_emp[i] = sum(data .<= x[i])
#     end

#     cdf_emp /= ndata
#     return cdf_emp, x
# end



function empcdf(data::AbstractArray; nx=100, x=nothing)

    if isnothing(x)
        m = minimum(data)
        M = maximum(data)
        x = range(m, stop=M, length=nx)
    end

    ndata = length(data)
    cdf_emp = zeros( Float64, (1, nx) )

    @turbo for ii in 1:nx, jj in 1:ndata
            cdf_emp[ii] += ( data[jj] <=  x[ii] )
    end


    cdf_emp /= ndata
    return cdf_emp, x
end



function invcdf(x, cdf, nr; cont=1)
    xi = x
    cdfi = cdf[:]
    rr = range(1.01 * minimum(cdf), stop=0.99 * maximum(cdf), length=nr)
    r = zeros( Float64, nr )

    for i in 1:nr
        ind = min(findfirst(>(rr[i]), cdfi) + 1, length(xi))
        if ind == 1 || cont == 2
            r[i] = xi[ind]
        else
            r[i] = ( xi[ind] - xi[ind-1] ) / ( cdfi[ind] - cdfi[ind-1] ) *
                   ( rr[i] - cdfi[ind-1] ) + xi[ind-1]
        end
    end

    return r
end
