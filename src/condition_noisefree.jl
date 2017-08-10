"""
    condition(Σ::PDMats.AbstractPDMat, train_idx::Vector{Idx}) where {Idx<:Integer}

Takes the covariance matrix from a multivariate normal distribution and
returns the covariance matrix of the distribution conditioned on values for the
elements in `train_idx`.
"""
function condition(Σ::PDMats.AbstractPDMat, train_idx::Vector{Idx}) where {Idx<:Integer}
    n = dim(Σ)
    test_idx = setdiff(1:n, train_idx)
    K_train = Σ.mat[train_idx, train_idx]
    K_cross = Σ.mat[test_idx, train_idx]
    K_test = Σ.mat[test_idx, test_idx]
    Σ_cond = K_test - K_cross * inv(K_train) * K_cross'
    PDMats.PDMat(cholfact(Hermitian(Σ_cond)))
end

"""
    condition(μ, Σ::PDMats.AbstractPDMat, train_idx::Vector{Idx}, train_val::Vector{T}) where {Idx<:Integer,T<:Real}

Returns a vector of the mean of the multivariate normal distribution
with `train_idx` conditioned on `train_val`.
"""
function condition(μ::Distributions.ZeroVector{T}, Σ::PDMats.AbstractPDMat, train_idx::Vector{Idx}, train_val::Vector{T}) where {Idx<:Integer,T<:Real}
    n = length(μ)
    test_idx = setdiff(1:n, train_idx)
    K_train = Σ.mat[train_idx, train_idx]
    K_cross = Σ.mat[test_idx, train_idx]
    K_cross * inv(K_train) * train_val
end

function condition(μ::Vector{T}, Σ::PDMats.AbstractPDMat, train_idx::Vector{Idx}, train_val::Vector{T}) where {Idx<:Integer,T<:Real}
    n = length(μ)
    test_idx = setdiff(1:n, train_idx)
    μ[test_idx] + condition(Distributions.ZeroVector{T}(n), Σ, train_idx, train_val)
end

"""
    condition(d::AbstractMvNormal, train_idx, train_val)

Returns the MvNormal distribution conditioned on the training values and the
indices of the non-conditioned entries.
"""
function condition(d::AbstractMvNormal, train_idx::Vector{Idx}, train_val::Vector{T}) where {Idx<:Integer,T<:Real}
    μ_cond = condition(d.μ, d.Σ, train_idx, train_val)
    Σ_cond = condition(d.Σ, train_idx)
    MvNormal(μ_cond, Σ_cond), setdiff(1:length(d), train_idx)
end
