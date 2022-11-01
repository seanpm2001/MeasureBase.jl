const IntegerLike = Union{Integer, Static.StaticInteger}
const RealLike = Union{Real, Static.StaticNumber}



struct StaticUnitRange{A,B} <: AbstractUnitRange{Int} end

@inline function Base.length(r::StaticUnitRange{A,B}) where {A,B}
    c = static(B - A + one(B))
    ifelse(B >= A, c, zero(c))
end

@inline Base.axes(r::StaticUnitRange) = (oneto_range(length(r)),)

@inline _sur_getindex(r::StaticUnitRange, i::Static.StaticInteger, ::StaticBool{false}) = Base.throw_boundserror(r, i)
@inline _sur_getindex(r::StaticUnitRange{A,B}, i::Static.StaticInteger{C}, ::StaticBool{true}) where {A,B,C} = static(A + C - one(C))

@inline function Base.getindex(r::StaticUnitRange{A,B}, i::Static.StaticInteger) where {A,B}
    _sur_getindex(r, i, static(one(i) <= i <= length(r)))
end

@inline Base.getindex(r::StaticUnitRange, i::Integer) = r[static(i)]

Base.first(::StaticUnitRange{A,B}) where {A,B} = A
Base.last(::StaticUnitRange{A,B}) where {A,B} = B
Base.step(::StaticUnitRange{A,B}) where {A,B} = static(1)

Base.show(io::IO, ::StaticUnitRange{A,B}) where {A,B} = print(io, "StaticUnitRange{$A,$B}()")

unit_range(a::IntegerLike, b::IntegerLike) = dynamic(a):dynamic(b)
unit_range(::Static.StaticInteger{A}, ::Static.StaticInteger{B}) where {A,B} = StaticUnitRange{A,B}()


const StaticOneTo{N} = StaticUnitRange{1,N}
Base.show(io::IO, ::StaticOneTo{N}) where N = print(io, "StaticOneTo{$N}()")

oneto_range(n::Integer) = Base.OneTo(n)
oneto_range(::Static.StaticInteger{N}) where N = StaticOneTo{N}()

const OneToLike = Union{Base.OneTo,StaticOneTo}
