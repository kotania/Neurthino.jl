module Neurthino

using LinearAlgebra
using SparseArrays
using StaticArrays
using Polynomials
using DocStringExtensions
using LRUCache
using LightGraphs
using AxisArrays
using NLsolve

import Base

export oscprob, Pνν, Pνν, OscillationParameters, NonunitaryOscillationParameters, PMNSMatrix, Hamiltonian, MatterOscillationMatrices
export masssquareddiff!, setΔm²!, cpphase!, setδ!, mixingangle!, setθ!, setelem!, complete_to_unitary_matrix!, calc_residuals_for_unknown_elements
export cpphases, mixingangles

export NeutrinoFlavour, Electron, Muon, Tau

const N_A = 6.022e23 #[mol^-1]
const G_F = 8.961877245622253e-38 #[eV*cm^3]

# Julia 1.0 compatibility
isnothing(::Any) = false
isnothing(::Nothing) = true

include("Oscillation.jl")
include("Matter.jl")
include("PREM.jl")

end # module
