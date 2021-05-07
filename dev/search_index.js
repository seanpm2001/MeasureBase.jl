var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = MeasureBase","category":"page"},{"location":"#MeasureBase","page":"Home","title":"MeasureBase","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for MeasureBase.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [MeasureBase]","category":"page"},{"location":"#MeasureBase.Density","page":"Home","title":"MeasureBase.Density","text":"struct Density{M,B}\n    μ::M\n    base::B\nend\n\nFor measures μ and ν with μ≪ν, the density of μ with respect to ν (also called the Radon-Nikodym derivative dμ/dν) is a function f defined on the support of ν with the property that for any measurable a ⊂ supp(ν), μ(a) = ∫ₐ f dν.\n\nBecause this function is often difficult to express in closed form, there are many different ways of computing it. We therefore provide a formal representation to allow comptuational flexibilty.\n\n\n\n\n\n","category":"type"},{"location":"#MeasureBase.DensityMeasure","page":"Home","title":"MeasureBase.DensityMeasure","text":"struct DensityMeasure{F,B} <: AbstractMeasure\n    density :: F\n    base    :: B\nend\n\nA DensityMeasure is a measure defined by a density with respect to some other \"base\" measure \n\n\n\n\n\n","category":"type"},{"location":"#MeasureBase.SuperpositionMeasure","page":"Home","title":"MeasureBase.SuperpositionMeasure","text":"struct SuperpositionMeasure{X,NT} <: AbstractMeasure\n    components :: NT\nend\n\nSuperposition of measures is analogous to mixture distributions, but (because measures need not be normalized) requires no scaling.\n\nThe superposition of two measures μ and ν can be more concisely written as μ + ν.\n\n\n\n\n\n","category":"type"},{"location":"#MeasureBase.WeightedMeasure","page":"Home","title":"MeasureBase.WeightedMeasure","text":"struct WeightedMeasure{R,M} <: AbstractMeasure\n    logweight :: R\n    base :: M\nend\n\n\n\n\n\n","category":"type"},{"location":"#MeasureBase.:≃-Tuple{Any, Any}","page":"Home","title":"MeasureBase.:≃","text":"≃(μ,ν)\n\nEquivalence of Measure\n\nMeasures μ and ν on the same space X are equivalent, written μ ≃ ν, if μ ≪ ν and ν ≪ μ. Note that this is often written ~ in the literature, but this is overloaded in probabilistic programming, so we use this alternate notation. \n\nAlso note that equivalence is very different from equality. For two equivalent measures, the sets of non-zero measure will be identical, but what that measure is in each case can be very different. \n\n\n\n\n\n","category":"method"},{"location":"#MeasureBase.:≪","page":"Home","title":"MeasureBase.:≪","text":"≪(μ,ν)\n\nAbsolute continuity\n\nA measure μ is absolutely continuous with respect to ν, written μ ≪ ν, if ν(A)==0 implies μ(A)==0 for every ν-measurable set A.\n\nLess formally, suppose we have a set A with ν(A)==0. If μ(A)≠0, then there can be no way to \"reweight\" ν to get to μ. We can't make something from nothing.\n\nThis \"reweighting\" is really a density function. If μ≪ν, then there is some function f that makes μ == ∫(f,ν) (see the help section for ∫).\n\nWe can get this f directly via the Radon-Nikodym derivative, f == 𝒹(μ,ν) (see the help section for 𝒹).\n\nNote that ≪ is not a partial order, because it is not antisymmetric. That is to say, it's possible (in fact, common) to have two different measures μ and ν with μ ≪ ν and ν ≪ μ. A simple example of this is \n\nμ = Normal()\nν = Lebesgue(ℝ)\n\nWhen ≪ holds in both directions, the measures μ and ν are equivalent, written μ ≃ ν. See the help section for ≃ for more information.\n\n\n\n\n\n","category":"function"},{"location":"#MeasureBase.For-Tuple{Any, Vararg{Any, N} where N}","page":"Home","title":"MeasureBase.For","text":"For(f, base...)\n\nFor provides a convenient way to construct a ProductMeasure. There are several options for the base. With Julia's do notation, this can look very similar to a standard for loop, while maintaining semantics structure that's easier to work with.\n\n\n\nFor(f, base::Int...)\n\nWhen one or several Int values are passed for base, the result is treated as depending on CartesianIndices(base). \n\njulia> For(3) do λ Exponential(λ) end |> marginals\n3-element mappedarray(MeasureTheory.var\"#17#18\"{var\"#15#16\"}(var\"#15#16\"()), ::CartesianIndices{1, Tuple{Base.OneTo{Int64}}}) with eltype Exponential{(:λ,), Tuple{Int64}}:\n Exponential(λ = 1,)\n Exponential(λ = 2,)\n Exponential(λ = 3,)\n\njulia> For(4,3) do μ,σ Normal(μ,σ) end |> marginals\n4×3 mappedarray(MeasureTheory.var\"#17#18\"{var\"#11#12\"}(var\"#11#12\"()), ::CartesianIndices{2, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}) with eltype Normal{(:μ, :σ), Tuple{Int64, Int64}}:\n Normal(μ = 1, σ = 1)  Normal(μ = 1, σ = 2)  Normal(μ = 1, σ = 3)\n Normal(μ = 2, σ = 1)  Normal(μ = 2, σ = 2)  Normal(μ = 2, σ = 3)\n Normal(μ = 3, σ = 1)  Normal(μ = 3, σ = 2)  Normal(μ = 3, σ = 3)\n Normal(μ = 4, σ = 1)  Normal(μ = 4, σ = 2)  Normal(μ = 4, σ = 3)\n\n\n\nFor(f, base::AbstractArray...)`\n\nIn this case, base behaves as if the arrays are zipped together before applying the map.\n\njulia> For(randn(3)) do x Exponential(x) end |> marginals\n3-element mappedarray(x->Main.Exponential(x), ::Vector{Float64}) with eltype Exponential{(:λ,), Tuple{Float64}}:\n Exponential(λ = -0.268256,)\n Exponential(λ = 1.53044,)\n Exponential(λ = -1.08839,)\n\njulia> For(1:3, 1:3) do μ,σ Normal(μ,σ) end |> marginals\n3-element mappedarray((:μ, :σ)->Main.Normal(μ, σ), ::UnitRange{Int64}, ::UnitRange{Int64}) with eltype Normal{(:μ, :σ), Tuple{Int64, Int64}}:\n Normal(μ = 1, σ = 1)\n Normal(μ = 2, σ = 2)\n Normal(μ = 3, σ = 3)\n\n\n\nFor(f, base::Base.Generator)\n\nFor Generators, the function maps over the values of the generator:\n\njulia> For(eachrow(rand(4,2))) do x Normal(x[1], x[2]) end |> marginals |> collect\n4-element Vector{Normal{(:μ, :σ), Tuple{Float64, Float64}}}:\n Normal(μ = 0.255024, σ = 0.570142)\n Normal(μ = 0.970706, σ = 0.0776745)\n Normal(μ = 0.731491, σ = 0.505837)\n Normal(μ = 0.563112, σ = 0.98307)\n\n\n\n\n\n","category":"method"},{"location":"#MeasureBase.asparams","page":"Home","title":"MeasureBase.asparams","text":"asparams build on TransformVariables.as to construct bijections to the parameter space of a given parameterized measure. Because this is only possible for continuous parameter spaces, we allow constraints to assign values to any subset of the parameters.\n\n\n\nasparams(::Type{<:ParameterizedMeasure}, ::Val{::Symbol})\n\nReturn a transformation for a particular parameter of a given parameterized measure. For example,\n\njulia> asparams(Normal, Val(:σ))\nasℝ₊\n\n\n\nasparams(::Type{<: ParameterizedMeasure{N}}, constraints::NamedTuple) where {N}\n\nReturn a transformation for a given parameterized measure subject to the named tuple constraints. For example,\n\njulia> asparams(Binomial{(:p,)}, (n=10,))\nTransformVariables.TransformTuple{NamedTuple{(:p,), Tuple{TransformVariables.ScaledShiftedLogistic{Float64}}}}((p = as𝕀,), 1)\n\n\n\naspararams(::ParameterizedMeasure)\n\nReturn a transformation with no constraints. For example,\n\njulia> asparams(Normal{(:μ,:σ)})\nTransformVariables.TransformTuple{NamedTuple{(:μ, :σ), Tuple{TransformVariables.Identity, TransformVariables.ShiftedExp{true, Float64}}}}((μ = asℝ, σ = asℝ₊), 2)\n\n\n\n\n\n","category":"function"},{"location":"#MeasureBase.basemeasure","page":"Home","title":"MeasureBase.basemeasure","text":"basemeasure(μ)\n\nMany measures are defined in terms of a logdensity relative to some base measure. This makes it important to be able to find that base measure.\n\nFor measures not defined in this way, we'll typically have basemeasure(μ) == μ.\n\n\n\n\n\n","category":"function"},{"location":"#MeasureBase.isprimitive-Tuple{Any}","page":"Home","title":"MeasureBase.isprimitive","text":"isprimitive(μ)\n\nMost measures are defined in terms of other measures, for example using a density or a pushforward. Those that are not are considered (in this library, it's not a general measure theory thing) to be primitive. The canonical example of a primitive measure is Lebesgue(X) for some X.\n\nThe default method is     isprimitive(μ) = false\n\nSo when adding a new primitive measure, it's necessary to add a method for its type that returns true.\n\n\n\n\n\n","category":"method"},{"location":"#MeasureBase.kernel","page":"Home","title":"MeasureBase.kernel","text":"kernel(f, M)\nkernel((f1, f2, ...), M)\n\nA kernel κ = kernel(f, m) returns a wrapper around a function f giving the parameters for a measure of type M, such that κ(x) = M(f(x)...) respective κ(x) = M(f1(x), f2(x), ...)\n\nIf the argument is a named tuple (;a=f1, b=f1), κ(x) is defined as M(;a=f(x),b=g(x)).\n\nReference\n\nhttps://en.wikipedia.org/wiki/Markov_kernel\n\n\n\n\n\n","category":"function"},{"location":"#MeasureBase.logdensity","page":"Home","title":"MeasureBase.logdensity","text":"logdensity(μ::AbstractMeasure{X}, x::X)\n\nCompute the logdensity of the measure μ at the point x. This is the standard way to define logdensity for a new measure. the base measure is implicit here, and is understood to be basemeasure(μ).\n\nMethods for computing density relative to other measures will be\n\n\n\n\n\n","category":"function"},{"location":"#MeasureBase.representative-Tuple{Any}","page":"Home","title":"MeasureBase.representative","text":"representative(μ::AbstractMeasure) -> AbstractMeasure\n\nWe need to be able to compute μ ≪ ν for each μ and ν. To do this directly would require a huge number of methods (quadratic in the number of defined measures). \n\nThis function is a way around that. When defining a new measure μ, you should also find some equivalent measure ρ that's \"as primitive as possible\". \n\nIf possible, ρ should be a PrimitiveMeasure, or a Product of these. If not, it should be a  transform (Pushforward or Pullback) of a PrimitiveMeasure (or Product of these). \n\n\n\n\n\n","category":"method"},{"location":"#MeasureBase.∫-Tuple{Any, AbstractMeasure}","page":"Home","title":"MeasureBase.∫","text":"∫(f, base::AbstractMeasure; log=true)\n\nDefine a new measure in terms of a density f over some measure base. If log=true (the default), f is considered as a log-density.\n\n\n\n\n\n","category":"method"},{"location":"#MeasureBase.𝒹-Tuple{AbstractMeasure, AbstractMeasure}","page":"Home","title":"MeasureBase.𝒹","text":"𝒹(μ::AbstractMeasure, base::AbstractMeasure; log=true)\n\nCompute the Radom-Nikodym derivative (or its log, if log=true) of μ with respect to base.\n\n\n\n\n\n","category":"method"},{"location":"#MeasureBase.@domain-Tuple{Any, Any}","page":"Home","title":"MeasureBase.@domain","text":"@domain(name, T)\n\nDefines a new singleton struct T, and a value name for building values of that type.\n\nFor example, @domain ℝ RealNumbers is equivalent to\n\nstruct RealNumbers <: AbstractDomain end\n\nexport ℝ\n\nℝ = RealNumbers()\n\nBase.show(io::IO, ::RealNumbers) = print(io, \"ℝ\")\n\n\n\n\n\n","category":"macro"},{"location":"#MeasureBase.@measure-Tuple{Any}","page":"Home","title":"MeasureBase.@measure","text":"@measure <declaration>\n\nThe <declaration> gives a measure and its default parameters, and specifies its relation to its base measure. For example,\n\n@measure Normal(μ,σ)\n\ndeclares the Normal is a measure with default parameters μ and σ. The result is equivalent to\n\nstruct Normal{N,T} <: ParameterizedMeasure{N}\n    par :: NamedTuple{N,T}\nend\n\nKeywordCalls.@kwstruct Normal(μ,σ)\n\nNormal(μ,σ) = Normal((μ=μ, σ=σ))\n\nSee KeywordCalls.jl for details on @kwstruct.\n\n\n\n\n\n","category":"macro"}]
}
