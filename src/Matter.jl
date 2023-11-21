struct Path
    density::Vector{Float64}
    baseline::Vector{Float64}
end

function extend_dims(A, which_dim)
    s = [size(A)...]
    insert!(s,which_dim,1)
    return reshape(A, s...)
end


Path(density::Number, baseline::Number) = Path([density],[baseline])

Base.iterate(p::Path, state=1) = state > length(p.density) ? nothing : ( (p.density[state], p.baseline[state]),  state+1)

Base.length(p::Path) = length(p.density)

"""
$(SIGNATURES)

Create modified oscillation parameters for neutrino propagation through matter

# Arguments
- `P`: Vacuum PMNS Matrix
- `H`: Vacuum Hamiltonian
- `density`: Matter density [g*cm^-3] 
- `energy`: Neutrino energy [GeV]
- `zoa`: Proton nucleon ratio (Z/A)
- `anti`: Is anti neutrino
"""
function MatterOscillationMatrices(U, H, energy, density; zoa=0.5, anti=false)
    # Convert from mass basis to flavour basis
    H_eff = convert(Array{ComplexF64}, U * Diagonal{Complex}(H) * adjoint(U))
    # Subtract the effective potential in the flavour basis by calling the function below
    MatterOscillationMatrices(H_eff, energy, density; zoa=zoa, anti=anti)
end


"""
$(SIGNATURES)

Create modified oscillation parameters for neutrino propagation through matter

# Arguments
- `H_eff`: Effective Matter Hamiltonian
- `density`: Matter density [g*cm^-3] 
- `energy`: Neutrino energy [GeV]
- `zoa`: Proton nucleon ratio (Z/A)
- `anti`: Is anti neutrino
"""
function MatterOscillationMatrices(H_eff, energy, density; zoa=0.5, anti=false)
    A = sqrt(2) * G_F * N_A * density
    if anti
        H_eff[1,1] -= A * zoa * 2 * energy * 1e9
    else
        H_eff[1,1] += A * zoa * 2 * energy * 1e9
    end
    # Subtract Vz (= - A*Nn*E*1e9) for any sterile flavours
    if size(H_eff)[1] > 3
        for i in 4:size(H_eff)[1]
            if anti
                H_eff[i,i] -= A * (1 - zoa) * energy * 1e9
            else
                H_eff[i,i] += A * (1 - zoa) * energy * 1e9
            end
        end
    end
    tmp = eigen(H_eff)
    return tmp.vectors, tmp.values
end


function NUMatterOscillationMatrices(U, H, energy, density; zoa=0.5, anti=false)

    if anti
        H_eff = conj.(U) * Diagonal{ComplexF64}(H) * adjoint(conj.(U))
    else
        H_eff = U * Diagonal{ComplexF64}(H) * adjoint(U)
    end

    H_eff = U * Diagonal{ComplexF64}(H) * adjoint(U)

    A = sqrt(2) * G_F * N_A * density
    # Neutron nucleon ratio: N/A = 1 - (Z/A)
    noa = 1 - zoa
    dim = size(H_eff)[1]
    A_eff = zeros(dim, dim)
    if anti
        A_eff[1,1] -= A * (2 * zoa - noa) * energy * 1e9 # CC + NC
        A_eff[2,2] += A * noa * energy * 1e9 #
        A_eff[3,3] += A * noa * energy * 1e9
    else
        A_eff[1,1] += A * (2 * zoa - noa) * energy * 1e9 
        A_eff[2,2] -= A * noa * energy * 1e9
        A_eff[3,3] -= A * noa * energy * 1e9
    end
    
    U_Udag = U * adjoint(U)

    # Non-unitary matter potential
    A_eff = (U_Udag) * A_eff * (U_Udag)
    H_eff -= A_eff

    tmp = eigen(H_eff)
    return tmp.vectors, tmp.values
end

"""
$(SIGNATURES)

Create modified oscillation parameters for neutrino propagation through matter

# Arguments
- `osc_vacuum::OscillationParameters`: Oscillation parameters in vacuum
- `energy`: Neutrino energy [GeV]
- `density`: Matter density in g*cm^-3 
- `zoa`: Proton nucleon ratio (Z/A)
- `anti`: Is anti neutrino

"""
function MatterOscillationMatrices(osc_vacuum::OscillationParameters, energy, density; zoa=0.5, anti=false)
    H_vacuum = Diagonal(Hamiltonian(osc_vacuum)) 
    U_vacuum = PMNSMatrix(osc_vacuum; anti=anti)
    H_eff = convert(Array{ComplexF64}, U_vacuum * Diagonal{ComplexF64}(H_vacuum) * adjoint(U_vacuum))
    return MatterOscillationMatrices(H_eff, energy, density; zoa=zoa, anti=anti)
end


"""
$(SIGNATURES)

# Arguments
- `U`: Vacuum PMNS Matrix
- `H`: Vacuum Hamiltonian
- `energy`: Neutrino energy [GeV]
- `path::Vector{Path}`: Neutrino path
- `zoa`: Proton nucleon ratio (Z/A)
- `anti`: Is anti neutrino
"""
function oscprob(U, H, energy::Vector{T}, path::Vector{Path}; zoa=0.5, anti=false) where {T <: Real}
    energy = convert.(Float64, energy)
    if anti
        H_eff = conj.(U) * Diagonal{ComplexF64}(H) * adjoint(conj.(U))
    else
        H_eff = U * Diagonal{ComplexF64}(H) * adjoint(U)
    end
    A = zeros(ComplexF64, length(energy), length(path), size(U)...)
    cache_size = length(energy) * sum(map(x->length(x.density), path)) 
    lru = LRU{Tuple{Float64, Float64},
              Tuple{Array{ComplexF64,2}, Vector{ComplexF64}}}(maxsize=cache_size)

    for k in 1:length(energy)
        @inbounds E = energy[k]

        for (l, p) in enumerate(path)

            tmp = Matrix{ComplexF64}(1I, size(U))
            for (m,b) in enumerate(p.baseline)
                @inbounds ρ = p.density[m]
                @inbounds z = zoa[l][m]
                U_mat, H_mat = get!(lru, (E, ρ)) do
                    MatterOscillationMatrices(copy(H_eff), E, ρ; zoa=z, anti=anti)
                end  
                tmp *= Neurthino._oscprobampl(U_mat, H_mat, E, b)
            end
            @inbounds A[k, l,  :, :] = tmp        
        end
    end
    P = map(x -> abs.(x) .^ 2, A)
    flavrange = _make_flavour_range(first(size(U)))
    AxisArray(P; Energy=energy, Path=path, InitFlav=flavrange, FinalFlav=flavrange)
end

const oscprob(U, H, energy::T, path::Vector{Path}; zoa=0.5, anti=false) where {T <: Real} = oscprob(U, H, [energy], path; zoa=zoa, anti=anti)

const oscprob(U, H, energy, path::Path; zoa=0.5, anti=false) = oscprob(U, H, energy, [path]; zoa=zoa, anti=anti)


function nu_oscprob(U, H, energy::Vector{T}, path::Vector{Path}; zoa=0.5, anti=false) where {T <: Real}
    energy = convert.(Float64, energy)
    
    A = zeros(ComplexF64, length(energy), length(path), size(U)...)
    cache_size = length(energy) * sum(map(x->length(x.density), path)) 
    lru = LRU{Tuple{Float64, Float64},
              Tuple{Array{ComplexF64,2}, Vector{ComplexF64}}}(maxsize=cache_size)
    for k in 1:length(energy)
        @inbounds E = energy[k]
        for (l, p) in enumerate(path)
            tmp = Matrix{ComplexF64}(1I, size(U))
            for (m,b) in enumerate(p.baseline)
                @inbounds ρ = p.density[m]
                U_mat, H_mat = get!(lru, (E, ρ)) do
                    NUMatterOscillationMatrices(copy(U), copy(H), E, ρ; zoa=zoa, anti=anti)
                end  
                tmp *= Neurthino._oscprobampl(U_mat, H_mat, E, b)
            end
            @inbounds A[k, l,  :, :] = tmp        
        end
    end
    P = map(x -> abs.(x) .^ 2, A)
    flavrange = _make_flavour_range(first(size(U)))

    U_Udag = abs.(U * adjoint(U))

    norms = [U_Udag[1, 1]^2 U_Udag[1, 1] * U_Udag[2, 2] U_Udag[1, 1]*U_Udag[3, 3];
            U_Udag[1, 1] * U_Udag[2, 2] U_Udag[2, 2]^2 U_Udag[2, 2]*U_Udag[3, 3];
            U_Udag[1, 1] * U_Udag[3, 3] U_Udag[2, 2] * U_Udag[3, 3] U_Udag[3, 3]^2]

    norms = extend_dims(extend_dims(norms, 1), 1)

    P = P ./ norms 

    AxisArray(P; Energy=energy, Path=path, InitFlav=flavrange, FinalFlav=flavrange)

end

const nu_oscprob(U, H, energy::T, path::Vector{Path}; zoa=0.5, anti=false) where {T <: Real} = nu_oscprob(U, H, [energy], path; zoa=zoa, anti=anti)

const nu_oscprob(U, H, energy, path::Path; zoa=0.5, anti=false) = nu_oscprob(U, H, energy, [path]; zoa=zoa, anti=anti)




"""
$(SIGNATURES)

# Arguments
- `osc_vacuum::OscillationParameters`: Vacuum oscillation parameters
- `energy`: Neutrino energy [GeV]
- `path`: Neutrino path
- `zoa`: Proton nucleon ratio (Z/A)
- `anti`: Is anti neutrino
"""
function oscprob(osc_vacuum::OscillationParameters, energy, path::Union{Path, Vector{Path}}; zoa=0.5, anti=false)
    # TODO: attach U_vac and H_vac to the oscillation parameters, so that it's
    # only calculated once and invalidated when any of the oscillation parameters
    # are changed
    U_vac = PMNSMatrix(osc_vacuum; anti=anti)
    H_vac = Hamiltonian(osc_vacuum)
    oscprob(U_vac, H_vac, energy, path; zoa=zoa, anti=anti)
end
