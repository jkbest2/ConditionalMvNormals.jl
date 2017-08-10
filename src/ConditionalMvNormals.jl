module ConditionalMvNormals

using Distributions
using PDMats

export
    AbstractMvNormal,
    ConditionalMvNormal

abstract type ConditionalMvNormal end

struct ConditionalMvNormal{C<:Integer,
                           V<:Real,
                           Cov<:AbstractPDMat{V},
                           Mean<:Union{Vector{V},Distributions.ZeroVector{V}}}
    μ::Mean
    Σ::Cov
    cond_idx::Vector{C}
    cond_val::Vector{V}
end

end # module
