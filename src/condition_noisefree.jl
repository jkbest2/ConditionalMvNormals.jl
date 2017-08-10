function condition(d::MvNormal{T, PDMats.PDMat{T, Array{T, 2}},
                               Union{Vector{T}, Distributions.ZeroVector{T}}},
                   obs_idx::Vector{<:Integer}, obs_val::Vector{T}) where {T<:Real}
    test_idx = setdiff(1:length(d), obs_idx)
    K_obs_chol = dropindex(d.Σ.chol, test_idx)
    K_obs_mat =  dropindex(d.Σ.mat, test_idx)
    K_cross_mat = d.Σ.mat[test_idx, obs_idx]
    K_test_mat = d.Σ.mat[test_idx, test_idx]
    K_test_chol = dropindex(d.Σ.chol, test_idx)
    μ_cond = K_cross_mat * inv(K_obs_chol) * obs_val
    downdate_fact = K_cross_mat \ K_obs_chol[:U]
    Σ_cond_mat = K_test_mat - downdate_fact * downdate_fact'
    for j in 1:length(obs_idx)
        K_test_chol .= downdate!(K_test_chol, downdate_fact[:, j])
    end
    Σ_cond = PDMats.PDMat(Σ_cond_mat, K_test_chol)
    ConditionalMvNormal(μ_cond, Σ_cond, obs_idx, obs_val)
end
