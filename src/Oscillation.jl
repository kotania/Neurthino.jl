@enum NeutrinoFlavour begin
  Electron = 1
  Muon = 2
  Tau = 3
end


struct OscillationParameters{T}
    mixing_angles::SparseMatrixCSC{T,<:Integer}
    mass_squared_diff::SparseMatrixCSC{T,<:Integer}
    cp_phases::SparseMatrixCSC{T,<:Integer}
    dim::Int64
    OscillationParameters(dim::Int64) = begin
        new{ComplexF64}(
                spzeros(dim, dim),
                spzeros(dim, dim),
                spzeros(dim, dim),
                dim)
    end
end


struct NonunitaryOscillationParameters{T}
"""
    This struct defines the matrix elements of the neutrino mixing matrix directly
    rather than parametrizing it via a product of 3 rotation matrices.

"""
    unitary_matrix_elements::Array{Float64, 2}
    alphas::Array{Float64, 2}
    mass_squared_diff::SparseMatrixCSC{T,<:Integer}
    cp_phases::SparseMatrixCSC{T,<:Integer}
    dim::Int64
    NonunitaryOscillationParameters(dim::Int64) = begin
        new{ComplexF64}(
                zeros.(Float64, dim, dim),
                zeros.(Float64, dim, dim),
                spzeros(dim, dim),
                spzeros(dim, dim),
                dim)
    end
end


function _generate_ordered_index_pairs(n::Integer)
    indices = Vector{Pair{Int64,Int64}}(undef, mixingangles(n))
    a = 1
    for i in 1:n 
        for j in 1:i-1
            indices[a] = Pair(j,i)
            a += 1
        end
    end
    indices
end

"""
$(SIGNATURES)

Set a mixing angle of an oscillation parameters struct

# Arguments
- `osc::OscillationParameters`: Oscillation parameters 
- `indices::Pair{<:Integer, <:Integer}`: The indices of the mixing angle
- `value<:Real` The value which should be applied to the oscillation parameters

"""
function mixingangle!(osc::OscillationParameters, indices::Pair{T, T}, value::S) where {T <: Integer, S <: Real}
    fromidx = first(indices)
    toidx = last(indices)
    if fromidx < toidx
        osc.mixing_angles[fromidx, toidx] = value
    else
        osc.mixing_angles[toidx, fromidx] = value
    end
end

"""
$(SIGNATURES)

Set a mixing angle of an oscillation parameters struct

# Arguments
- `osc::OscillationParameters`: Oscillation parameters 
- `args::Tuple{Pair{<:Integer, <:Integer}, <:Real}`: The indices of the mixing angle

"""
function mixingangle!(osc::OscillationParameters, (args::Tuple{Pair{T, T}, S})...) where {T <: Integer, S<: Real}
    for a in args
        mixingangle!(osc, first(a), last(a))
    end
end

const setθ! = mixingangle!

function _mass_matrix_fully_determined(osc::Union{OscillationParameters, NonunitaryOscillationParameters})
    I, J, _ = findnz(osc.mass_squared_diff)
    set_elements = collect(zip(I, J))
    indices = Set([first.(set_elements)..., last.(set_elements)...])
    (length(indices) >= osc.dim) & (length(set_elements) >= (osc.dim - 1))  
end

function _mass_matrix_overdetermined(osc::Union{OscillationParameters, NonunitaryOscillationParameters})
    I, J, _ = findnz(osc.mass_squared_diff)
    set_elements = collect(zip(I, J))
    indices = Set([first.(set_elements)..., last.(set_elements)...])
    length(set_elements) >= length(indices) 
end

function Base.isvalid(osc::Union{OscillationParameters, NonunitaryOscillationParameters})
    return _mass_matrix_fully_determined(osc)
end

function _completed_mass_matrix(osc::Union{OscillationParameters, NonunitaryOscillationParameters})
    tmp = Matrix(osc.mass_squared_diff)
    tmp = tmp - transpose(tmp)
    if _mass_matrix_fully_determined(osc)
        I, J, _ = findnz(osc.mass_squared_diff)
        given_idx = collect(zip(I, J))
        wanted_idx = filter(x->(x[1] < x[2]) & (x ∉ given_idx), collect(Iterators.product(1:osc.dim, 1:osc.dim)))
        graph = SimpleDiGraph(map(x->x!=0.0, tmp))
        for (from, to) in wanted_idx
            path = a_star(graph, from, to)
            for edge in path
                tmp[from, to] += tmp[edge.src, edge.dst]
            end
        end
    else
        error("Mass squared differences not fully determined!")
    end
    UpperTriangular(tmp)
end

"""
$(SIGNATURES)

Set a mass squared difference of an oscillation parameters struct

# Arguments
- `osc::OscillationParameters`: Oscillation parameters 
- `indices::Pair{<:Integer, <:Integer}`: The indices of the mass squared difference
- `value` The value which should be applied to the oscillation parameters

"""
function masssquareddiff!(osc::Union{OscillationParameters, NonunitaryOscillationParameters}, indices::Pair{T, T}, value::S) where {T <: Integer, S <: Number}
    fromidx = first(indices)
    toidx = last(indices)
    if fromidx < toidx
        osc.mass_squared_diff[fromidx, toidx] = value
    elseif fromidx == toidx
        error("Mass squared difference with equal index cannot be modified.")
    else
        osc.mass_squared_diff[toidx, fromidx] = -value
    end
    if _mass_matrix_overdetermined(osc)
        @warn "Mass squared difference fields (partially) overdetermined!"
    end
end

"""
$(SIGNATURES)

Set a mass squared difference of an oscillation parameters struct

# Arguments
- `osc::OscillationParameters`: Oscillation parameters 
- `args::Tuple{Pair{<:Integer, <:Integer}, <:Number}`: Indices and values of the mass squared difference

"""
function masssquareddiff!(osc::Union{OscillationParameters, NonunitaryOscillationParameters}, (args::Tuple{Pair{<:Integer, <:Integer}, <:Number})...)
    for a in args
        masssquareddiff!(osc, first(a), last(a))
    end
end

const setΔm²! = masssquareddiff!

"""
$(SIGNATURES)

Set a CP phase of an oscillation parameters struct

# Arguments
- `osc::OscillationParameters`: Oscillation parameters 
- `indices::Pair{<:Integer, <:Integer}`: The indices of the mass difference
- `value` The value which should be applied to the oscillation parameters

"""
function cpphase!(osc::OscillationParameters, indices::Pair{T, T}, value::S) where {T <: Integer, S <: Real}
    fromidx = first(indices)
    toidx = last(indices)
    if fromidx < toidx
        osc.cp_phases[fromidx, toidx] = value
    else
        osc.cp_phases[toidx, fromidx] = value
    end
end

"""
$(SIGNATURES)

Set a CP phase of an oscillation parameters struct

# Arguments
- `osc::OscillationParameters`: Oscillation parameters 
- `args::Tuple{Pair{<:Integer, <:Integer}, <:Number}`: Indices and values of the CP phase

"""
function cpphase!(osc::OscillationParameters, (args::Tuple{Pair{T, T}, S})...) where {T <: Integer, S <: Real}
    for a in args
        cpphase!(osc, first(a), last(a))
    end
end


function cpphase!(osc::NonunitaryOscillationParameters, indices::Tuple{T, T}, value::S) where {T <: Integer, S <: Real}
    osc.cp_phases[indices[1], indices[2]] = value
end

function alphamatrix!(osc::NonunitaryOscillationParameters, indices::Tuple{T, T}, value::S) where {T <: Integer, S <: Real}
    osc.alphas[indices[1], indices[2]] = value
end

const setδ! = cpphase!
const setα! = alphamatrix!

"""
$(SIGNATURES)

Set individual elements of the unitary mixing matrix

# Arguments
- `osc::NonunitaryOscillationParameters`: Oscillation parameters 
- `indices::Tuple{<:Integer, <:Integer}`: The indices of mixing matrix
- `value` The value which should be applied to the mixing matrix

"""
function elements!(osc::NonunitaryOscillationParameters, indices::Tuple{T, T}, value::S) where {T <: Integer, S <: Real}
    osc.unitary_matrix_elements[indices[1], indices[2]] = value
end

const setelem! = elements!



function calc_residuals_for_unknown_elements(x::Array{Float64, 1}, known_params::Array{Float64, 1})

    e1, mu3, tau3, phi_e3 = known_params
    e2, e3, mu1, mu2, tau1, tau2, phi_mu1, phi_mu2, phi_tau1, phi_tau2 = x

    if phi_e3 != 0.
        phi_mu1, phi_mu2, phi_tau1, phi_tau2 = mod2pi.([phi_mu1, phi_mu2, phi_tau1, phi_tau2])
    else 
        phi_mu1, phi_mu2, phi_tau1, phi_tau2 = zeros(4)
    end

    phi_e1, phi_e2, phi_mu3, phi_tau3 = zeros(4)

    # TODO: why doesn't this work using direct matrix multiplication?

    # U = Matrix{ComplexF64}([abs(e1)*exp(-1im*phi_e1) abs(e2)*exp(-1im*phi_e2) abs(e3)*exp(-1im*phi_e3); 
    #     abs(mu1)*exp(-1im*phi_mu1) abs(mu2)*exp(-1im*phi_mu2) abs(mu3)*exp(-1im*phi_mu3); 
    #     abs(tau1)*exp(-1im*phi_tau1) abs(tau2)*exp(-1im*phi_tau2) abs(tau3)*exp(-1im*phi_tau3)])

   
    # U_Udag = U * adjoint(U)

    # residuals = abs.(vec(U_Udag - I))

    expr1 = (e3^2 + mu3^2 + tau3^2) # = 1
    expr2 = (e2^2 + mu2^2 + tau2^2) # = 1
    expr3 = (e1^2 + mu1^2 + tau1^2) # = 1


    expr4 = (e1*exp(-1im*phi_e1)*conj(e2*exp(-1im * phi_e2)) + mu1*exp(-1im*phi_mu1)*conj(mu2*exp(-1im*phi_mu2)) + tau1*exp(-1im*phi_tau1)*conj(tau2*exp(-1im*phi_tau2))) # = 0
    expr5 = (e1*exp(-1im*phi_e1)*conj(e3*exp(-1im * phi_e3)) + mu1*exp(-1im*phi_mu1)*conj(mu3*exp(-1im*phi_mu3)) + tau1*exp(-1im*phi_tau1)*conj(tau3*exp(-1im*phi_tau3))) # = 0
    expr6 = (e2*exp(-1im*phi_e2)*conj(e3*exp(-1im * phi_e3)) + mu2*exp(-1im*phi_mu2)*conj(mu3*exp(-1im*phi_mu3)) + tau2*exp(-1im*phi_tau2)*conj(tau3*exp(-1im*phi_tau3))) # = 0


    residuals = [expr1 - 1., expr2 - 1., expr3 - 1., real(expr4), real(expr5), real(expr6), imag(expr4), imag(expr5), imag(expr6)]

    residuals

end


function complete_to_unitary_matrix!(osc::NonunitaryOscillationParameters)

"""
    We require that U_t3, U_e1, and U_mu3 elements are provided and complete the PMNS matrix
    to unitary based on these inputs.
"""
    dim = size(osc.unitary_matrix_elements)[1]
    for i in 1:dim
        for j in 1:dim 

            if (i == 1 && j == 1) || (i == 2 && j == 3) || (i == 3 && j == 3)
                @assert osc.unitary_matrix_elements[i, j] != 0. "A value for the element ($i, $j) was not set"
            else 
                @assert osc.unitary_matrix_elements[i, j] == 0. "A non-zero value for the matrix element other than U_e1, U_mu3, U_t3 was provided"
            end
        end
    end

    @assert (osc.unitary_matrix_elements[3, 3]^2 + osc.unitary_matrix_elements[2, 3]^3) < 1. "The normalization of the 3rd column exceeds 1"
    @assert (abs(osc.unitary_matrix_elements[1, 1]) < 1.) "The magnitude of the U_e1 element exceeds 1"

    e1, mu3, tau3 = osc.unitary_matrix_elements[1, 1], osc.unitary_matrix_elements[2, 3], osc.unitary_matrix_elements[3, 3]
    phi_e3 = Real(osc.cp_phases[1, 3])

    # Order of the unknown magnitudes: [e2, e3, mu1, mu2, tau1, tau2]
    order_of_element_magnitudes = [(1, 2), (1, 3), (2, 1), (2, 2), (3, 1), (3, 2)]

    # Order of the unknown phases: [phi_mu1, phi_mu2, phi_tau1, phi_tau2]
    order_of_element_phases = [(2, 1), (2, 2), (3, 1), (3, 2)]

    solver_func(x) = calc_residuals_for_unknown_elements(x, [e1, mu3, tau3, phi_e3])
    solution = nlsolve(solver_func, [ones(6); zeros(4)]).zero

    #display(solution)

    for n in 1:size(order_of_element_magnitudes)[1]
        setelem!(osc, order_of_element_magnitudes[n], solution[n])

    end

    for m in 1:size(order_of_element_phases)[1]
        a, b = order_of_element_phases[m]
        phase_value = solution[size(solution)[1]-3:size(solution)[1]][m]
        # println("Setting delta ($a, $b) to $phase_value")
        setδ!(osc, order_of_element_phases[m], phase_value)
    end

end

function PMNSMatrix(osc_params::OscillationParameters; anti=false)
"""
$(SIGNATURES)

Create rotation matrix (PMNS) based on the given oscillation parameters

# Arguments
- `osc_params::OscillationParameters`: Oscillation parameters
- `anti`: Is anti neutrino

"""
    dim = size(osc_params.mixing_angles)[1]
    pmns = Matrix{ComplexF64}(1.0I, dim, dim) 
    indices = _generate_ordered_index_pairs(dim)
    for (i, j) in indices
        rot = sparse((1.0+0im)I, dim, dim) 
        mixing_angle = osc_params.mixing_angles[i, j]
        c, s = cos(mixing_angle), sin(mixing_angle)
        rot[i, i] = c
        rot[j, j] = c
        rot[i, j] = s
        rot[j, i] = -s
        if CartesianIndex(i, j) in findall(!iszero, osc_params.cp_phases)
            cp_phase = osc_params.cp_phases[i, j]
            cp_term = exp(-1im * cp_phase)
            if anti
                cp_term = conj(cp_term)
            end
            rot[i, j] *= cp_term
            rot[j, i] *= conj(cp_term)
        end
        pmns = rot * pmns 
    end
    pmns
end

function PMNSMatrix(osc_params::NonunitaryOscillationParameters; anti=false)
"""
$(SIGNATURES)

Create rotation matrix (PMNS) based on the provided unitary matrix elements (real),
CP phases (up to 4 values applied to any 2x2 submatrix of PMNS), 
and small corrections to unitarity given in the alpha-matrix (generally complex).

# Arguments
- `osc_params::NonunitaryOscillationParameters`: Oscillation parameters
- `anti`: Is anti neutrino

"""
    dim = size(osc_params.unitary_matrix_elements)[1]

    pmns = convert(Matrix{ComplexF64}, (I - osc_params.alphas) * osc_params.unitary_matrix_elements)

    for i in 1:dim 
        for j in 1:dim 

            cp_phase = osc_params.cp_phases[i, j]
            cp_term = exp(-1im * cp_phase)
            if anti
                cp_term = conj(cp_term)
            end

            pmns[i, j] *= cp_term
        end
    end

    pmns


end

            

function Hamiltonian(osc_params::Union{OscillationParameters, NonunitaryOscillationParameters})
"""
$(SIGNATURES)

Create modified hamiltonian matrix consisting of the squared mass differences
based on the given oscillation parameters

# Arguments
- `osc_params::OscillationParameters`: Oscillation parameters

"""
    Hamiltonian(osc_params, zeros(Float64, osc_params.dim))
end 

function Hamiltonian(osc_params::Union{OscillationParameters, NonunitaryOscillationParameters}, lambda)
"""
$(SIGNATURES)

Create modified hamiltonian matrix consisting of the squared mass differences
based on the given oscillation parameters

# Arguments
- `osc_params::OscillationParameters`:  Oscillation parameters
- `lambda`:                             Decay parameters for each mass eigenstate

"""
    full_mass_squared_matrix = _completed_mass_matrix(osc_params)
    H = zeros(ComplexF64, osc_params.dim)
    for i in 1:osc_params.dim
        for j in 1:osc_params.dim
            if i < j
                H[i] += full_mass_squared_matrix[i,j]
            elseif j < i
                H[i] -= full_mass_squared_matrix[j,i]
            end
        end
        H[i] += 1im * lambda[i]
    end
    H /= osc_params.dim
    H
end


function _oscprobampl(U, H, energy, baseline)  
    H_diag = 2.5338653580781976 * Diagonal{ComplexF64}(H) * baseline / energy 
    U * exp(1im * H_diag) * adjoint(U)
end


function _nuoscprobampl(U, H, energy, baseline, first_layer, last_layer)  
   
    H_exp = 2.5338653580781976 * H * baseline / energy
    if first_layer && last_layer 
        U * exp(-1im * H_exp) * adjoint(U)
    elseif first_layer && (!last_layer)
        U * exp(-1im * H_exp)
    elseif last_layer && (!first_layer)
        exp(-1im * H_exp) * adjoint(U)
    elseif (!first_layer) && (!last_layer)
        exp(-1im * H_exp) 
    end
end

function _make_flavour_range(size::Integer)
    if size <= 3
        return NeutrinoFlavour.(1:size)
    else
        return [NeutrinoFlavour.(1:3)..., 4:size...]
    end
end


"""
$(SIGNATURES)

Calculate the transistion probabilities between the neutrino flavours

# Arguments
- `U`:          PMNS Matrix
- `H`:          Hamiltonian
- `energy`:     Energies [GeV]
- `baseline`:   Baselines [km]

"""
function oscprob(U, H, energy::Vector{T}, baseline::Vector{S}) where {T,S <: Real}
    s = (size(U)..., length(energy), length(baseline))
    combinations = collect(Iterators.product(energy, baseline))
    tmp = map(x->abs.(_oscprobampl(U, H, first(x), last(x))).^2, combinations)
    P = reshape(hcat(collect(Iterators.flatten(tmp))), s...)
    P = permutedims(P, (3,4,1,2))
    flavrange = _make_flavour_range(first(size(U)))


    AxisArray(P; Energy=energy, Baseline=baseline, InitFlav=flavrange, FinalFlav=flavrange)
end

const oscprob(U, H, energy::T, baseline::Vector{S}) where {S,T <: Real} = oscprob(U, H, [energy], baseline)
const oscprob(U, H, energy, baseline::T) where {T <: Real} = oscprob(U, H, energy, [baseline])

"""
$(SIGNATURES)

Calculate the transistion probabilities between the neutrino flavours

# Arguments
- `osc_params::OscillationParameters`:  Oscillation parameters
- `energy`:                             Energy [GeV]
- `baseline`:                           Baseline [km]
- `anti`:                               Is anti neutrino

"""
function oscprob(osc_params::OscillationParameters, energy, baseline; anti=false)  
    H = Hamiltonian(osc_params)
    U = PMNSMatrix(osc_params; anti=anti)
    Pνν(U, H, energy, baseline)
end


function oscprob(osc_params::NonunitaryOscillationParameters, energy, baseline; anti=false)  
    complete_to_unitary_matrix!(osc_params) 
    H = Hamiltonian(osc_params)
    U = PMNSMatrix(osc_params; anti=anti)
    Pνν(U, H, energy, baseline)
end


const Pνν = oscprob

"""
$(SIGNATURES)

Returns the number of CP violating phases at given number of neutrino types

# Arguments
- `n`: number of neutrino types in the supposed model

# Examples
```julia-repl
julia> cpphases(3)
1
```
"""
function cpphases(n)
    n < 1 && return 0
    div((n - 1) * (n - 2), 2 )
end

"""
$(SIGNATURES)

Returns the number of mixing angles at given number of neutrino types

# Arguments
- `n`: number of neutrino types in the supposed model

# Examples
```julia-repl
julia> mixingangles(3)
3
```
"""
function mixingangles(n)
    n < 1 && return 0
    div(n * (n - 1), 2)
end
