# TODO: Compare with ChangesOfVariables.jl

abstract type AbstractTransformedMeasure <: AbstractMeasure end

abstract type AbstractPushforward <: AbstractTransformedMeasure end

abstract type AbstractPullback <: AbstractTransformedMeasure end

function gettransform(::AbstractTransformedMeasure) end

function params(::AbstractTransformedMeasure) end

function paramnames(::AbstractTransformedMeasure) end

function parent(::AbstractTransformedMeasure) end

export PushforwardMeasure

"""
    struct PushforwardMeasure{FF,IF,MU,VC<:TransformVolCorr} <: AbstractPushforward
        f :: FF
        inv_f :: IF
        origin :: MU
        volcorr :: VC
    end
"""
struct PushforwardMeasure{FF,IF,M,VC<:TransformVolCorr} <: AbstractPushforward
    f::FF
    inv_f::IF
    origin::M
    volcorr::VC
end


gettransform(ν::PushforwardMeasure) = ν.f
parent(ν::PushforwardMeasure) = ν.origin

function transport_def(ν::PushforwardMeasure{FF,IF,M}, μ::M, x) where {FF,IF,M}
    if μ == parent(ν)
        return ν.f(x)
    else
        invoke(transport_def, Tuple{Any, PushforwardMeasure, Any}, ν, μ, x)
    end
end

function transport_def(μ::M, ν::PushforwardMeasure{FF,IF,M}, y) where {FF,IF,M}
    if μ == parent(ν)
        return ν.inv_f(y)
    else
        invoke(transport_def, Tuple{Any, PushforwardMeasure, Any}, μ, ν, y)
    end
end

function Pretty.tile(ν::PushforwardMeasure)
    Pretty.list_layout(Pretty.tile.([ν.f, ν.inv_f, ν.origin]); prefix = :PushforwardMeasure)
end

# TODO: Reduce code duplication
@inline function logdensityof(
    ν::PushforwardMeasure{FF,IF,M,<:WithVolCorr},
    y,
) where {FF,IF,M}
    x_orig, inv_ladj = with_logabsdet_jacobian(ν.inv_f, y)
    μ = ν.origin
    logd_orig = unsafe_logdensityof(ν.origin, x_orig)
    logd = float(logd_orig + inv_ladj)
    neginf = oftype(logd, -Inf)
    insupport(μ, x_orig) || return oftype(logd, -Inf)
    return ifelse(
        # Zero density wins against infinite volume:
        (isnan(logd) && logd_orig == -Inf && inv_ladj == +Inf) ||
        # Maybe  also for (logd_orig == -Inf) && isfinite(inv_ladj) ?
        # Return constant -Inf to prevent problems with ForwardDiff:
        (isfinite(logd_orig) && (inv_ladj == -Inf)),
        neginf,
        logd,
    )
end

# TODO: THIS IS ALMOST CERTAINLY WRONG 
# @inline function logdensity_rel(
#     ν::PushforwardMeasure{FF1,IF1,M1,<:WithVolCorr},
#     β::PushforwardMeasure{FF2,IF2,M2,<:WithVolCorr},
#     y,
# ) where {FF1,IF1,M1,FF2,IF2,M2}
#     f = β.inv_f ∘ ν.f
#     inv_f = ν.inv_f ∘ β.f
#     logdensity_rel(pushfwd(f, inv_f, ν.origin, WithVolCorr()), β.origin, β.inv_f(y))
# end

@inline function logdensity_def(
    ν::PushforwardMeasure{FF,IF,M,<:WithVolCorr},
    y,
) where {FF,IF,M}
    x_orig, inv_ladj = with_logabsdet_jacobian(ν.inv_f, y)
    logd_orig = unsafe_logdensityof(ν.origin, x_orig)
    logd = float(logd_orig + inv_ladj)
    neginf = oftype(logd, -Inf)
    return ifelse(
        # Zero density wins against infinite volume:
        (isnan(logd) && logd_orig == -Inf && inv_ladj == +Inf) ||
        # Maybe  also for (logd_orig == -Inf) && isfinite(inv_ladj) ?
        # Return constant -Inf to prevent problems with ForwardDiff:
        (isfinite(logd_orig) && (inv_ladj == -Inf)),
        neginf,
        logd,
    )
end

@inline function logdensity_def(
    ν::PushforwardMeasure{FF,IF,M,<:NoVolCorr},
    y,
) where {FF,IF,M}
    x_orig = to_origin(ν, y)
    return unsafe_logdensityof(ν.origin, x_orig)
end

insupport(ν::PushforwardMeasure, y) = insupport(transport_origin(ν), to_origin(ν, y))

testvalue(::Type{T}, ν::PushforwardMeasure) where {T} = from_origin(ν, testvalue(T, transport_origin(ν)))

@inline function basemeasure(ν::PushforwardMeasure)
    PushforwardMeasure(ν.f, ν.inv_f, basemeasure(transport_origin(ν)), NoVolCorr())
end

_pushfwd_dof(::Type{MU}, ::Type, dof) where {MU} = NoDOF{MU}()
_pushfwd_dof(::Type{MU}, ::Type{<:Tuple{Any,Real}}, dof) where {MU} = dof

# Assume that DOF are preserved if with_logabsdet_jacobian is functional:
@inline function getdof(ν::MU) where {MU<:PushforwardMeasure}
    T = Core.Compiler.return_type(testvalue, Tuple{typeof(ν.origin)})
    R = Core.Compiler.return_type(with_logabsdet_jacobian, Tuple{typeof(ν.f),T})
    _pushfwd_dof(MU, R, getdof(ν.origin))
end

# Bypass `checked_arg`, would require potentially costly transformation:
@inline checked_arg(::PushforwardMeasure, x) = x

@inline transport_origin(ν::PushforwardMeasure) = ν.origin
@inline from_origin(ν::PushforwardMeasure, x) = ν.f(x)
@inline to_origin(ν::PushforwardMeasure, y) = ν.inv_f(y)

function Base.rand(rng::AbstractRNG, ::Type{T}, ν::PushforwardMeasure) where {T}
    return ν.f(rand(rng, T, parent(ν)))
end

export pushfwd

"""
    pushfwd(f, [f_inverse,] μ, volcorr = WithVolCorr())

Return the [pushforward
measure](https://en.wikipedia.org/wiki/Pushforward_measure) from `μ` the
[measurable function](https://en.wikipedia.org/wiki/Measurable_function) `f`.

If `f_inverse` is specified, it must be a valid inverse of the function given by
restricting `f` to the support of `μ`.
"""
pushfwd(f, μ, volcorr::TransformVolCorr = WithVolCorr()) = PushforwardMeasure(f, inverse(f), μ, volcorr)

pushfwd(f, finv, μ, volcorr::TransformVolCorr = WithVolCorr()) = PushforwardMeasure(f, finv, μ, volcorr)

@inline function pushfwd(f, μ::PushforwardMeasure, volcorr::TransformVolCorr = WithVolCorr())
    _pushfwd(f, inverse(f), μ, volcorr, μ.volcorr)
end


@inline function pushfwd(f, finv, μ::PushforwardMeasure, volcorr::TransformVolCorr = WithVolCorr())
    _pushfwd(f, finv, μ, volcorr, μ.volcorr)
end

@inline function _pushfwd(f, finv, μ::PushforwardMeasure, vf::V, vμ::V) where {V<:TransformVolCorr}
    pushfwd(f ∘ μ.f, μ.inv_f ∘ finv, μ, v)
end

@inline function _pushfwd(f, finv, μ::PushforwardMeasure, vf, vμ) 
    PushforwardMeasure(f, finv, μ, vf)
end