# Functions for manipulating covariance matrices

function dropcol_(c::LinAlg.Cholesky, idx::Integer)
    n = size(c, 1)
    cU = c[:U][:, [1:(idx-1); (idx+1):end; idx]]
    for i in idx:(n-1)
        cU .= givens(cU, i, i+1, i)[1] * cU
    end
    LinAlg.Cholesky(cU[1:(n-1), 1:(n-1)], :U)
end

"""
    function dropcol(c::LinAlg.Cholesky, idx::Integer)

Drop the `idx` column from the Cholesky factorization `c` without
re-forming the original matrix.
"""
function dropcol(c::LinAlg.Cholesky, idx::Integer)
    idx ≤ size(c, 1) || error("Index out of bounds")
    dropcol_(c, idx)
end


"""
    dropcol(c::LinAlg.Cholesky, idx::Vector{T} where T<:Integer)

Drop the columns in `idx` from the Cholesky factorization `c` without
re-forming the original matrix.
"""
function dropcol(c::LinAlg.Cholesky, idx::Vector{T} where T<:Integer)
    # Get sizes
    n = size(c, 1)
    n_drop = length(idx)
    n_cond = n - n_drop
    all(idx .< n) || error("Index out of bounds")

    cU = c[:U][:, [setdiff(1:n, idx); idx]]
    for j in minimum(idx):n_cond # Iterate over columns
        for i in n_drop:-1:1 # Then rows, working upward
            cU .= givens(cU, i + j - 1, i + j, j)[1] * cU
        end
    end
    LinAlg.Cholesky(cU[1:n_cond, 1:n_cond], :U)
end
