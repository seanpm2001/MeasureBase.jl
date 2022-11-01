# Modified version of Static.jl

module StaticReals

import IfElse: ifelse

using Static: StaticFalse

import Static: dynamic, is_static, known, static_promote



abstract type StaticRealInteger{N} <: Integer end

"""
    StaticRealInt(N::Int) -> StaticRealInt{N}()

A statically sized `Int`.
Use `StaticRealInt(N)` instead of `Val(N)` when you want it to behave like a number.
"""
struct StaticRealInt{N} <: StaticRealInteger{N}
    StaticRealInt{N}() where {N} = new{N::Int}()
    StaticRealInt(N::Int) = new{N}()
    StaticRealInt(@nospecialize N::StaticRealInt) = N
    StaticRealInt(::Val{N}) where {N} = StaticRealInt(N)
end

Base.getindex(x::Tuple, ::StaticRealInt{N}) where {N} = getfield(x, N)

Base.zero(@nospecialize(::StaticRealInt)) = StaticRealInt{0}()

Base.to_index(x::StaticRealInt) = known(x)
function Base.checkindex(::Type{Bool}, inds::AbstractUnitRange, ::StaticReal{N}) where {N}
    checkindex(Bool, inds, N)
end

"""
    StaticFloat64{N}

A statically sized `Float64`.
Use `StaticRealInt(N)` instead of `Val(N)` when you want it to behave like a number.
"""
struct StaticFloat64{N} <: AbstractFloat
    StaticFloat64{N}() where {N} = new{N::Float64}()
    StaticFloat64(x::Float64) = new{x}()
    StaticFloat64(x::Int) = new{Base.sitofp(Float64, x)::Float64}()
    StaticFloat64(x::StaticRealInt{N}) where {N} = StaticFloat64(convert(Float64, N))
    StaticFloat64(x::Complex) = StaticFloat64(convert(Float64, x))
end

"""
    StaticBool(x::Bool) -> StaticTrue/StaticFalse

A statically typed `Bool`.
"""
abstract type StaticBool{bool} <: Integer end

struct StaticTrue <: StaticBool{true} end

struct StaticFalse <: StaticBool{false} end

StaticBool{true}() = StaticTrue()
StaticBool{false}() = StaticFalse()
StaticBool(x::StaticBool) = x
function StaticBool(x::Bool)
    if x
        return StaticTrue()
    else
        return StaticFalse()
    end
end

ifelse(::StaticTrue, @nospecialize(x), @nospecialize(y)) = x
ifelse(::StaticFalse, @nospecialize(x), @nospecialize(y)) = y

const Zero = StaticRealInt{0}
const One = StaticRealInt{1}
const FloatOne = StaticFloat64{one(Float64)}
const FloatZero = StaticFloat64{zero(Float64)}

const StaticType{T} = Union{StaticReal{T}, StaticSymbol{T}}

StaticRealInt(x::StaticFalse) = Zero()
StaticRealInt(x::StaticTrue) = One()
Base.Bool(::StaticTrue) = true
Base.Bool(::StaticFalse) = false

Base.eltype(@nospecialize(T::Type{<:StaticFloat64})) = Float64
Base.eltype(@nospecialize(T::Type{<:StaticRealInt})) = Int
Base.eltype(@nospecialize(T::Type{<:StaticBool})) = Bool



const StaticReal{N} = Union{StaticRealInteger{N},StaticRealFloat64{N},StaticBool{N}}

known(::Type{<:StaticReal{T}}) where {T} = T

function Base.show(io::IO, @nospecialize(x::StaticReal))
    show(io, MIME"text/plain"(), x)
end

function Base.show(io::IO, ::MIME"text/plain", @nospecialize(x::StaticReal))
    print(io, "staticreal(" * repr(known(typeof(x))) * ")")
end



"""
    staticreal(x)

Returns a static form of `x` that is a subtype of `Real`.
```
"""
staticreal(@nospecialize(x::StaticReal)) = x
@inline staticreal(::Static.StaticNumber{X}) where X = staticreal(X)
staticreal(x::Integer) = StaticRealInt(x)
function staticreal(x::Union{AbstractFloat, Complex, Rational, AbstractIrrational})
    StaticFloat64(Float64(x))
end
staticreal(x::Bool) = StaticBool(x)
staticreal(x::Tuple{Vararg{Any}}) = map(staticreal, x)
staticreal(::Val{V}) where {V} = staticreal(V)
function staticreal(x::X) where {X}
    Base.issingletontype(X) && return x
    error("There is no static real alternative for type $(typeof(x)).")
end


"""
    is_static(::Type{T}) -> StaticBool

Returns `StaticTrue` if `T` is a staticreal type.

See also: [`staticreal`](@ref), [`known`](@ref)
"""
is_static(@nospecialize(x)) = is_static(typeof(x))
is_static(@nospecialize(x::Type{<:StaticType})) = Static.True()
is_static(@nospecialize(x::Type{<:Val})) = StaticTrue()
_tuple_static(::Type{T}, i) where {T} = is_static(field_type(T, i))
@inline function is_static(@nospecialize(T::Type{<:Tuple}))
    if all(eachop(_tuple_static, nstatic(Val(fieldcount(T))), T))
        return StaticTrue()
    else
        return StaticFalse()
    end
end
is_static(T::DataType) = StaticFalse()

"""
    dynamic(x)

Returns the "dynamic" or non-staticreal form of `x`.
"""
@inline dynamic(@nospecialize x::StaticType) = known(x)
@inline dynamic(@nospecialize x::Tuple) = map(dynamic, x)
dynamic(@nospecialize(x::NDIndex)) = CartesianIndex(dynamic(Tuple(x)))
dynamic(@nospecialize x) = x



function Base.promote_rule(@nospecialize(T1::Type{<:StaticReal}),
                           @nospecialize(T2::Type{<:StaticReal}))
    promote_rule(eltype(T1), eltype(T2))
end
function Base.promote_rule(::Type{<:Base.TwicePrecision{R}},
                           @nospecialize(T::Type{<:StaticReal})) where {R <: Number}
    promote_rule(Base.TwicePrecision{R}, eltype(T))
end
function Base.promote_rule(@nospecialize(T1::Type{<:StaticReal}),
                           T2::Type{<:Union{Rational, AbstractFloat, Signed}})
    promote_rule(T2, eltype(T1))
end

Base.:(~)(::StaticReal{N}) where {N} = staticreal(~N)

Base.inv(x::StaticReal{N}) where {N} = one(x) / x

@inline Base.one(@nospecialize T::Type{<:StaticReal}) = staticreal(one(eltype(T)))
@inline Base.zero(@nospecialize T::Type{<:StaticReal}) = staticreal(zero(eltype(T)))
@inline Base.iszero(::Union{StaticReal{0}, StaticReal{0.0}, StaticReal{false}}) = true
@inline Base.iszero(@nospecialize x::StaticReal) = false
@inline Base.isone(::Union{StaticReal{1}, StaticReal{1.0}, StaticReal{true}}) = true
@inline Base.isone(@nospecialize x::StaticReal) = false
@inline Base.iseven(@nospecialize x::StaticReal) = iseven(known(x))
@inline Base.isodd(@nospecialize x::StaticReal) = isodd(known(x))

Base.AbstractFloat(::StaticReal{N}) where {N} = StaticFloat64{X}()

Base.abs(::StaticReal{N}) where {N} = staticreal(abs(N))
Base.abs2(::StaticReal{N}) where {N} = staticreal(abs2(N))
Base.sign(::StaticReal{N}) where {N} = staticreal(sign(N))

Base.widen(@nospecialize(x::StaticReal)) = widen(known(x))

Base.convert(::Type{T}, @nospecialize(x::StaticReal)) where {T <: Number} = convert(T, known(x))

Base.Integer(@nospecialize(x::StaticRealInt)) = x
(::Type{T})(x::StaticReal) where {T <: Real} = T(known(x))
function (@nospecialize(T::Type{<:StaticReal}))(x::Union{AbstractFloat,
                                                           AbstractIrrational, Integer,
                                                           Rational})
    staticreal(convert(eltype(T), x))
end

@inline Base.:(-)(::StaticReal{N}) where {N} = staticreal(-N)
Base.:(*)(::Union{AbstractFloat, AbstractIrrational, Integer, Rational}, y::Zero) = y
Base.:(*)(x::Zero, ::Union{AbstractFloat, AbstractIrrational, Integer, Rational}) = x
Base.:(*)(::StaticReal{X}, ::StaticReal{Y}) where {X, Y} = staticreal(X * Y)
Base.:(/)(::StaticReal{X}, ::StaticReal{Y}) where {X, Y} = staticreal(X / Y)
Base.:(-)(::StaticReal{X}, ::StaticReal{Y}) where {X, Y} = staticreal(X - Y)
Base.:(+)(::StaticReal{X}, ::StaticReal{Y}) where {X, Y} = staticreal(X + Y)
Base.:(-)(x::Ptr, ::StaticRealInt{N}) where {N} = x - N
Base.:(-)(::StaticRealInt{N}, y::Ptr) where {N} = y - N
Base.:(+)(x::Ptr, ::StaticRealInt{N}) where {N} = x + N
Base.:(+)(::StaticRealInt{N}, y::Ptr) where {N} = y + N

@generated Base.sqrt(::StaticReal{N}) where {N} = :($(staticreal(sqrt(N))))

function Base.div(::StaticReal{X}, ::StaticReal{Y}, m::RoundingMode) where {X, Y}
    staticreal(div(X, Y, m))
end
Base.div(x::Real, ::StaticReal{Y}, m::RoundingMode) where {Y} = div(x, Y, m)
Base.div(::StaticReal{X}, y::Real, m::RoundingMode) where {X} = div(X, y, m)
Base.div(x::StaticBool, y::StaticFalse) = throw(DivideError())
Base.div(x::StaticBool, y::StaticTrue) = x

Base.rem(@nospecialize(x::StaticReal), T::Type{<:Integer}) = rem(known(x), T)
Base.rem(::StaticReal{X}, ::StaticReal{Y}) where {X, Y} = staticreal(rem(X, Y))
Base.rem(x::Real, ::StaticReal{Y}) where {Y} = rem(x, Y)
Base.rem(::StaticReal{X}, y::Real) where {X} = rem(X, y)

Base.mod(::StaticReal{X}, ::StaticReal{Y}) where {X, Y} = staticreal(mod(X, Y))

Base.round(::StaticFloat64{M}) where {M} = StaticFloat64(round(M))
roundtostaticint(::StaticFloat64{M}) where {M} = StaticRealInt(round(Int, M))
roundtostaticint(x::AbstractFloat) = round(Int, x)
floortostaticint(::StaticFloat64{M}) where {M} = StaticRealInt(Base.fptosi(Int, M))
floortostaticint(x::AbstractFloat) = Base.fptosi(Int, x)

Base.:(==)(::StaticReal{X}, ::StaticReal{Y}) where {X, Y} = ==(X, Y)

Base.:(<)(::StaticReal{X}, ::StaticReal{Y}) where {X, Y} = <(X, Y)

Base.isless(::StaticReal{X}, ::StaticReal{Y}) where {X, Y} = isless(X, Y)
Base.isless(::StaticReal{X}, y::Real) where {X} = isless(X, y)
Base.isless(x::Real, ::StaticRealInteger{Y}) where {Y} = isless(x, Y)

Base.min(::StaticReal{X}, ::StaticReal{Y}) where {X, Y} = staticreal(min(X, Y))
Base.min(::StaticReal{X}, y::Number) where {X} = min(X, y)
Base.min(x::Number, ::StaticReal{Y}) where {Y} = min(x, Y)

Base.max(::StaticReal{X}, ::StaticReal{Y}) where {X, Y} = staticreal(max(X, Y))
Base.max(::StaticReal{X}, y::Number) where {X} = max(X, y)
Base.max(x::Number, ::StaticReal{Y}) where {Y} = max(x, Y)

Base.minmax(::StaticReal{X}, ::StaticReal{Y}) where {X, Y} = staticreal(minmax(X, Y))

Base.:(<<)(::StaticRealInteger{X}, ::StaticRealInteger{Y}) where {X, Y} = staticreal(<<(X, Y))
Base.:(<<)(::StaticRealInteger{X}, n::Integer) where {X} = <<(X, n)
Base.:(<<)(x::Integer, ::StaticRealInteger{N}) where {N} = <<(x, N)

Base.:(>>)(::StaticRealInteger{X}, ::StaticRealInteger{Y}) where {X, Y} = staticreal(>>(X, Y))
Base.:(>>)(::StaticRealInteger{X}, n::Integer) where {X} = >>(X, n)
Base.:(>>)(x::Integer, ::StaticRealInteger{N}) where {N} = >>(x, N)

Base.:(>>>)(::StaticRealInteger{X}, ::StaticRealInteger{Y}) where {X, Y} = staticreal(>>>(X, Y))
Base.:(>>>)(::StaticRealInteger{X}, n::Integer) where {X} = >>>(X, n)
Base.:(>>>)(x::Integer, ::StaticRealInteger{N}) where {N} = >>>(x, N)

Base.:(&)(::StaticRealInteger{X}, ::StaticRealInteger{Y}) where {X, Y} = staticreal(X & Y)
Base.:(&)(::StaticRealInteger{X}, y::Union{Integer, Missing}) where {X} = X & y
Base.:(&)(x::Union{Integer, Missing}, ::StaticRealInteger{Y}) where {Y} = x & Y
Base.:(&)(x::Bool, y::StaticTrue) = x
Base.:(&)(x::Bool, y::StaticFalse) = y
Base.:(&)(x::StaticTrue, y::Bool) = y
Base.:(&)(x::StaticFalse, y::Bool) = x

Base.:(|)(::StaticRealInteger{X}, ::StaticRealInteger{Y}) where {X, Y} = staticreal(|(X, Y))
Base.:(|)(::StaticRealInteger{X}, y::Union{Integer, Missing}) where {X} = X | y
Base.:(|)(x::Union{Integer, Missing}, ::StaticRealInteger{Y}) where {Y} = x | Y
Base.:(|)(x::Bool, y::StaticTrue) = y
Base.:(|)(x::Bool, y::StaticFalse) = x
Base.:(|)(x::StaticTrue, y::Bool) = x
Base.:(|)(x::StaticFalse, y::Bool) = y

Base.xor(::StaticRealInteger{X}, ::StaticRealInteger{Y}) where {X, Y} = staticreal(xor(X, Y))
Base.xor(::StaticRealInteger{X}, y::Union{Integer, Missing}) where {X} = xor(X, y)
Base.xor(x::Union{Integer, Missing}, ::StaticRealInteger{Y}) where {Y} = xor(x, Y)

Base.:(!)(::StaticTrue) = StaticFalse()
Base.:(!)(::StaticFalse) = StaticTrue()

Base.all(::Tuple{Vararg{StaticTrue}}) = true
Base.all(::Tuple{Vararg{Union{StaticTrue, StaticFalse}}}) = false
Base.all(::Tuple{Vararg{StaticFalse}}) = false

Base.any(::Tuple{Vararg{StaticTrue}}) = true
Base.any(::Tuple{Vararg{Union{StaticTrue, StaticFalse}}}) = true
Base.any(::Tuple{Vararg{StaticFalse}}) = false

Base.real(@nospecialize(x::StaticReal)) = x
Base.real(@nospecialize(T::Type{<:StaticReal})) = eltype(T)
Base.imag(@nospecialize(x::StaticReal)) = zero(x)

"""
    field_type(::Type{T}, f)

Functionally equivalent to `fieldtype(T, f)` except `f` may be a staticreal type.
"""
@inline field_type(T::Type, f::Union{Int, Symbol}) = fieldtype(T, f)
@inline field_type(::Type{T}, ::StaticRealInt{N}) where {T, N} = fieldtype(T, N)
@inline field_type(::Type{T}, ::StaticSymbol{S}) where {T, S} = fieldtype(T, S)

Base.rad2deg(::StaticFloat64{M}) where {M} = StaticFloat64(rad2deg(M))
Base.deg2rad(::StaticFloat64{M}) where {M} = StaticFloat64(deg2rad(M))
@generated Base.cbrt(::StaticFloat64{M}) where {M} = StaticFloat64(cbrt(M))
Base.mod2pi(::StaticFloat64{M}) where {M} = StaticFloat64(mod2pi(M))
@generated Base.sinpi(::StaticFloat64{M}) where {M} = StaticFloat64(sinpi(M))
@generated Base.cospi(::StaticFloat64{M}) where {M} = StaticFloat64(cospi(M))
Base.exp(::StaticFloat64{M}) where {M} = StaticFloat64(exp(M))
Base.exp2(::StaticFloat64{M}) where {M} = StaticFloat64(exp2(M))
Base.exp10(::StaticFloat64{M}) where {M} = StaticFloat64(exp10(M))
@generated Base.expm1(::StaticFloat64{M}) where {M} = StaticFloat64(expm1(M))
@generated Base.log(::StaticFloat64{M}) where {M} = StaticFloat64(log(M))
@generated Base.log2(::StaticFloat64{M}) where {M} = StaticFloat64(log2(M))
@generated Base.log10(::StaticFloat64{M}) where {M} = StaticFloat64(log10(M))
@generated Base.log1p(::StaticFloat64{M}) where {M} = StaticFloat64(log1p(M))
@generated Base.sin(::StaticFloat64{M}) where {M} = StaticFloat64(sin(M))
@generated Base.cos(::StaticFloat64{M}) where {M} = StaticFloat64(cos(M))
@generated Base.tan(::StaticFloat64{M}) where {M} = StaticFloat64(tan(M))
Base.sec(x::StaticFloat64{M}) where {M} = inv(cos(x))
Base.csc(x::StaticFloat64{M}) where {M} = inv(sin(x))
Base.cot(x::StaticFloat64{M}) where {M} = inv(tan(x))
@generated Base.asin(::StaticFloat64{M}) where {M} = StaticFloat64(asin(M))
@generated Base.acos(::StaticFloat64{M}) where {M} = StaticFloat64(acos(M))
@generated Base.atan(::StaticFloat64{M}) where {M} = StaticFloat64(atan(M))
@generated Base.sind(::StaticFloat64{M}) where {M} = StaticFloat64(sind(M))
@generated Base.cosd(::StaticFloat64{M}) where {M} = StaticFloat64(cosd(M))
Base.tand(x::StaticFloat64{M}) where {M} = sind(x) / cosd(x)
Base.secd(x::StaticFloat64{M}) where {M} = inv(cosd(x))
Base.cscd(x::StaticFloat64{M}) where {M} = inv(sind(x))
Base.cotd(x::StaticFloat64{M}) where {M} = inv(tand(x))
Base.asind(x::StaticFloat64{M}) where {M} = rad2deg(asin(x))
Base.acosd(x::StaticFloat64{M}) where {M} = rad2deg(acos(x))
Base.asecd(x::StaticFloat64{M}) where {M} = rad2deg(asec(x))
Base.acscd(x::StaticFloat64{M}) where {M} = rad2deg(acsc(x))
Base.acotd(x::StaticFloat64{M}) where {M} = rad2deg(acot(x))
Base.atand(x::StaticFloat64{M}) where {M} = rad2deg(atan(x))
@generated Base.sinh(::StaticFloat64{M}) where {M} = StaticFloat64(sinh(M))
Base.cosh(::StaticFloat64{M}) where {M} = StaticFloat64(cosh(M))
Base.tanh(x::StaticFloat64{M}) where {M} = StaticFloat64(tanh(M))
Base.sech(x::StaticFloat64{M}) where {M} = inv(cosh(x))
Base.csch(x::StaticFloat64{M}) where {M} = inv(sinh(x))
Base.coth(x::StaticFloat64{M}) where {M} = inv(tanh(x))
@generated Base.asinh(::StaticFloat64{M}) where {M} = StaticFloat64(asinh(M))
@generated Base.acosh(::StaticFloat64{M}) where {M} = StaticFloat64(acosh(M))
@generated Base.atanh(::StaticFloat64{M}) where {M} = StaticFloat64(atanh(M))
Base.asech(x::StaticFloat64{M}) where {M} = acosh(inv(x))
Base.acsch(x::StaticFloat64{M}) where {M} = asinh(inv(x))
Base.acoth(x::StaticFloat64{M}) where {M} = atanh(inv(x))
Base.asec(x::StaticFloat64{M}) where {M} = acos(inv(x))
Base.acsc(x::StaticFloat64{M}) where {M} = asin(inv(x))
Base.acot(x::StaticFloat64{M}) where {M} = atan(inv(x))

@inline Base.exponent(::StaticReal{M}) where {M} = staticreal(exponent(M))

Base.:(^)(::StaticFloat64{x}, y::Float64) where {x} = exp2(log2(x) * y)

Base.:(+)(x::StaticTrue) = One()
Base.:(+)(x::StaticFalse) = Zero()

# from `^(x::Bool, y::Bool) = x | !y`
Base.:(^)(x::StaticBool, y::StaticFalse) = StaticTrue()
Base.:(^)(x::StaticBool, y::StaticTrue) = x
Base.:(^)(x::Integer, y::StaticFalse) = one(x)
Base.:(^)(x::Integer, y::StaticTrue) = x
Base.:(^)(x::BigInt, y::StaticFalse) = one(x)
Base.:(^)(x::BigInt, y::StaticTrue) = x

@inline function Base.ntuple(f::F, ::StaticRealInt{N}) where {F, N}
    (N >= 0) || throw(ArgumentError(string("tuple length should be ≥ 0, got ", N)))
    if @generated
        quote
            Base.Cartesian.@ntuple $N i->f(i)
        end
    else
        Tuple(f(i) for i in 1:N)
    end
end

@inline function invariant_permutation(@nospecialize(x::Tuple), @nospecialize(y::Tuple))
    if y === x === ntuple(staticreal, StaticRealInt(nfields(x)))
        return StaticTrue()
    else
        return StaticFalse()
    end
end

@inline nstatic(::Val{N}) where {N} = ntuple(StaticRealInt, Val(N))

permute(@nospecialize(x::Tuple), @nospecialize(perm::Val)) = permute(x, staticreal(perm))
@inline function permute(@nospecialize(x::Tuple), @nospecialize(perm::Tuple))
    if invariant_permutation(nstatic(Val(nfields(x))), perm) === StaticFalse()
        return eachop(getindex, perm, x)
    else
        return x
    end
end

"""
    eachop(op, args...; iterator::Tuple{Vararg{StaticRealInt}}) -> Tuple

Produces a tuple of `(op(args..., iterator[1]), op(args..., iterator[2]),...)`.
"""
@inline function eachop(op::F, itr::Tuple{T, Vararg{Any}}, args::Vararg{Any}) where {F, T}
    (op(args..., first(itr)), eachop(op, Base.tail(itr), args...)...)
end
eachop(::F, ::Tuple{}, args::Vararg{Any}) where {F} = ()

"""
    eachop_tuple(op, arg, args...; iterator::Tuple{Vararg{StaticRealInt}}) -> Type{Tuple}

Produces a tuple type of `Tuple{op(arg, args..., iterator[1]), op(arg, args..., iterator[2]),...}`.
Note that if one of the arguments passed to `op` is a `Tuple` type then it should be the first argument
instead of one of the trailing arguments, ensuring type inference of each element of the tuple.
"""
eachop_tuple(op, itr, arg, args...) = _eachop_tuple(op, itr, arg, args)
@generated function _eachop_tuple(op, ::I, arg, args::A) where {A, I}
    t = Expr(:curly, Tuple)
    narg = length(A.parameters)
    for p in I.parameters
        call_expr = Expr(:call, :op, :arg)
        if narg > 0
            for i in 1:narg
                push!(call_expr.args, :(getfield(args, $i)))
            end
        end
        push!(call_expr.args, :(StaticRealInt{$(p.parameters[1])}()))
        push!(t.args, call_expr)
    end
    Expr(:block, Expr(:meta, :inline), t)
end

#=
    find_first_eq(x, collection::Tuple)

Finds the position in the tuple `collection` that is exactly equal (i.e. `===`) to `x`.
If `x` and `collection` are staticreal (`is_static`) and `x` is in `collection` then the return
value is a `StaticRealInt`.
=#
@generated function find_first_eq(x::X, itr::I) where {X, N, I <: Tuple{Vararg{Any, N}}}
    # we avoid incidental code gen when evaluated a tuple of known values by iterating
    #  through `I.parameters` instead of `known(I)`.
    index = known(X) === nothing ? nothing : findfirst(==(X), I.parameters)
    if index === nothing
        :(Base.Cartesian.@nif $(N + 1) d->(dynamic(x) == dynamic(getfield(itr, d))) d->(d) d->(nothing))
    else
        :($(staticreal(index)))
    end
end

function Base.invperm(p::Tuple{StaticRealInt, Vararg{StaticRealInt, N}}) where {N}
    map(Base.Fix2(find_first_eq, p), ntuple(staticreal, StaticRealInt(N + 1)))
end



