module StaticReals

import Static

using Static: known, dynamic

struct StaticReal{X} <: Real end
export StaticReal


Base.show(io::IO, @nospecialize(x::StaticReal)) = show(io, MIME"text/plain"(), x)

function Base.show(io::IO, ::MIME"text/plain", @nospecialize(x::StaticReal))
    print(io, "staticreal(" * repr(known(typeof(x))) * ")")
end


const StaticRealZero = StaticReal{0.0}
const StaticRealOne = StaticReal{1.0}


@inline staticreal(x::Float64) = StaticReal{x}()
@inline staticreal(x::Real) = StaticReal{Float64(x)}()
@inline staticreal(@nospecialize x::StaticReal) = x
@inline staticreal(::Static.StaticNumber{X}) where X = StaticReal{Float64(X)}()
export staticreal


@inline Static.known(::Type{StaticReal{X}}) where {X} = X

@inline Static.dynamic(@nospecialize x::StaticReal) = known(x)


#Base.eltype(::StaticReal) = Float64
Base.eltype(x::StaticReal{X}) where X = StaticReal{X}

Base.Float64(x::StaticReal) = known(x)

Base.promote_rule(::Type{<:StaticReal}, ::Type{T}) where {T<:Number} = T

Base.convert(::Type{StaticReal{X}}, ::StaticReal{X}) where X = StaticReal{X}()
Base.convert(T::Type{StaticReal{X}}, y::StaticReal{Y}) where {X,Y} = throw(InexactError(Symbol(T), T, y))
Base.convert(::Type{T}, ::StaticReal{X}) where {T<:Number,X} = T(X)
Base.convert(::Type{T}, ::StaticReal{X}) where {X,T<:Static.StaticNumber{X}} = T()

(::Type{T})(::StaticReal{X}) where {T<:Real,X} = T(X)


_sr_zero(::Val{true}, x) = x
_sr_zero(::Val{false}, x) = staticreal(zero(known(x)))
Base.zero(x::StaticReal{X}) where X = _sr_zero(Val(iszero(X)), x)

_sr_one(::Val{true}, x) = x
_sr_one(::Val{false}, x) = staticreal(one(known(x)))
Base.one(x::StaticReal{X}) where X = _sr_one(Val(isone(X)), x)


@inline Base.iszero(::StaticRealZero) = true
@inline Base.iszero(@nospecialize x::StaticReal) = false
@inline Base.isone(::StaticRealOne) = true
@inline Base.isone(@nospecialize x::StaticReal) = false


Base.:(*)(::Union{AbstractFloat, AbstractIrrational, Integer, Rational}, y::StaticRealZero) = y
Base.:(*)(x::StaticRealZero, ::Union{AbstractFloat, AbstractIrrational, Integer, Rational}) = x
Base.:(*)(::StaticReal{X}, ::StaticReal{Y}) where {X,Y} = staticreal(X * Y)
Base.:(/)(::StaticReal{X}, ::StaticReal{Y}) where {X,Y} = staticreal(X / Y)
Base.:(-)(::StaticReal{X}, ::StaticReal{Y}) where {X,Y} = staticreal(X - Y)
Base.:(+)(::StaticReal{X}, ::StaticReal{Y}) where {X,Y} = staticreal(X + Y)

@inline Base.:(+)(x::StaticReal) = x
@inline Base.:(-)(::StaticReal{X}) where X = staticreal(-X)

@inline Base.:(^)(x::Real, ::StaticRealZero) = staticreal(sign(x))
@inline Base.:(^)(x::Real, ::StaticRealOne) = x

Base.exp(::StaticReal{M}) where {M} = StaticReal{exp(M)}()
Base.exp2(::StaticReal{M}) where {M} = StaticReal{exp2(M)}()
Base.exp10(::StaticReal{M}) where {M} = StaticReal{exp10(M)}()
Base.expm1(::StaticReal{M}) where {M} = StaticReal{expm1(M)}()
Base.log(::StaticReal{M}) where {M} = StaticReal{log(M)}()
Base.log2(::StaticReal{M}) where {M} = StaticReal{log2(M)}()
Base.log10(::StaticReal{M}) where {M} = StaticReal{log10(M)}()
Base.log1p(::StaticReal{M}) where {M} = StaticReal{log1p(M)}()

end # module
