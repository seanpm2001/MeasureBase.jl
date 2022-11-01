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
const StaticRealPosInf = StaticReal{+Inf}
const StaticRealNegInf = StaticReal{-Inf}

const StaticRealInfinity = Union{StaticRealPosInf,StaticRealNegInf}

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

Base.promote_rule(::Type{<:StaticReal}, ::Type{T}) where {T<:Number} = float(T)

Base.convert(::Type{StaticReal{X}}, ::StaticReal{X}) where X = StaticReal{X}()
Base.convert(T::Type{StaticReal{X}}, y::StaticReal{Y}) where {X,Y} = throw(InexactError(Symbol(T), T, y))
Base.convert(::Type{T}, ::StaticReal{X}) where {T<:Number,X} = T(X)
Base.convert(::Type{T}, ::StaticReal{X}) where {X,T<:Static.StaticNumber{X}} = T()

(::Type{T})(::StaticReal{X}) where {T<:Real,X} = T(X)


Base.zero(::StaticReal) = StaticReal{0.0}()

Base.one(::StaticReal) = StaticReal{1.0}()


@inline Base.iszero(::StaticRealZero) = true
@inline Base.iszero(@nospecialize x::StaticReal) = false
@inline Base.isone(::StaticRealOne) = true
@inline Base.isone(@nospecialize x::StaticReal) = false

const RealSubType = Union{AbstractFloat, AbstractIrrational, Integer, Rational}

@inline Base.:(+)(x::StaticReal) = x
@inline Base.:(-)(::StaticReal{X}) where X = staticreal(-X)

Base.:(+)(x::RealSubType, ::StaticRealZero) = x
Base.:(+)(::StaticRealZero, y::RealSubType) = y
Base.:(+)(::StaticReal{X}, ::StaticReal{Y}) where {X,Y} = staticreal(X + Y)

Base.:(-)(x::RealSubType, ::StaticRealZero) = x
Base.:(-)(::StaticRealZero, y::RealSubType) = -y
Base.:(-)(::StaticReal{X}, ::StaticReal{Y}) where {X,Y} = staticreal(X - Y)

Base.:(*)(x::RealSubType, ::StaticRealOne) = x
Base.:(*)(::StaticRealOne, y::RealSubType) = y
Base.:(*)(::StaticReal{X}, ::StaticReal{Y}) where {X,Y} = staticreal(X * Y)

Base.:(/)(::StaticReal{X}, ::StaticReal{Y}) where {X,Y} = staticreal(X / Y)

@inline Base.:(^)(x::Real, ::StaticRealZero) = staticreal(sign(x))
@inline Base.:(^)(x::Real, ::StaticRealOne) = x

Base.exp(::StaticReal{M}) where {M} = StaticReal{exp(Float64(M))}()
Base.exp2(::StaticReal{M}) where {M} = StaticReal{exp2(Float64(M))}()
Base.exp10(::StaticReal{M}) where {M} = StaticReal{exp10(Float64(M))}()
Base.expm1(::StaticReal{M}) where {M} = StaticReal{expm1(Float64(M))}()
Base.log(::StaticReal{M}) where {M} = StaticReal{log(Float64(M))}()
Base.log2(::StaticReal{M}) where {M} = StaticReal{log2(Float64(M))}()
Base.log10(::StaticReal{M}) where {M} = StaticReal{log10(Float64(M))}()
Base.log1p(::StaticReal{M}) where {M} = StaticReal{log1p(Float64(M))}()

end # module
