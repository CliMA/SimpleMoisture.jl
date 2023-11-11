# A simulation of forced-dissipative two-dimensional turbulence. We solve the
# two-dimensional vorticity equation with stochastic excitation and dissipation in
# the form of linear drag and hyperviscosity.
include("passive_tracer.jl")

using CairoMakie
using CUDA
using GeophysicalFlows
using HDF5
using Printf
using Random 
using StatsBase
using Plots
using ProgressBars

### Device
dev = CPU()

### RNG
id = 12
if dev == CPU()
  Random.seed!(id)
else
  CUDA.seed!(id)
end
random_uniform = dev == CPU() ? rand : CUDA.rand

### Parameters to drive things from Gaussian to Non-Gaussian
# strongly non-Gaussian: τc = 0.01
# medium: τc = 0.1
# weakly: τc = 1.0
τc = 0.01

### Numerical, domain, and simulation parameters
n = 32                            # number of grid points
L = 2π                            # domain size
stepper = "ETDRK4"                # timestepper
ν, nν = 1e-6, 4                   # hyperviscosity coefficient and hyperviscosity order, 512: 1e-16; 256: 1e-14; 128: 1e-12: 64: 1e-8; 32: 1e-6
νc, nνc = ν, nν                   # hyperviscosity coefficient and hyperviscosity order for tracer
μ, nμ = 1e-2, 0                   # linear drag coefficient
forcing_wavenumber = 3.0 * 2π / L # the forcing wavenumber, `k_f`, for a spectrum that is a ring in wavenumber space
forcing_bandwidth = 2.0 * 2π / L  # the width of the forcing spectrum, `δ_f`
ε = 0.1                           # energy input rate by the forcing
γ₀ = 1.0                          # saturation specific humidity gradient
e = 1.0                           # evaporation rate           
τc = τc                           # condensation time scale
dt = 0.5e-1                       # timestep
dt_save = 10.0                    # when to store data
tend = dt_save*30000              # end time
nsteps = Int(tend ÷ dt)           # number of simulation steps
nsubs = Int(dt_save ÷ dt)         # number of steps between saves
c_init = τc * e                   # guess a good initial condition for the tracer to reduce spinup 

### Grid
grid = TwoDGrid(dev; nx=n, Lx=L)

### Vorticity forcing
# We force the vorticity equation with stochastic excitation that is delta-correlated in time 
# and while spatially homogeneously and isotropically correlated. The forcing has a spectrum 
# with power in a ring in wavenumber space of radius ``k_f`` (`forcing_wavenumber`) and width 
# ``δ_f`` (`forcing_bandwidth`), and it injects energy per unit area and per unit time 
# equal to ``\varepsilon``. That is, the forcing covariance spectrum is proportional to 
# ``\exp{[-(|\bm{k}| - k_f)^2 / (2 δ_f^2)]}``.
K = @. sqrt(grid.Krsq)               # a 2D array with the total wavenumber
forcing_spectrum = @. exp(-(K - forcing_wavenumber)^2 / (2 * forcing_bandwidth^2))
CUDA.@allowscalar forcing_spectrum[grid.Krsq.==0] .= 0 # ensure forcing has zero domain-average
ε0 = FourierFlows.parsevalsum(forcing_spectrum .* grid.invKrsq / 2, grid) / (grid.Lx * grid.Ly)
@. forcing_spectrum *= ε / ε0        # normalize forcing to inject energy at rate ε

# Next we construct function `calcF!` that computes a forcing realization every timestep.
function calcF!(Fh, sol, t, clock, vars, params, grid)
  Fh .= sqrt.(forcing_spectrum) .* exp.(2π * im * random_uniform(eltype(grid), size(sol))) ./ sqrt(clock.dt)
  return nothing
end

### Saturation specific humidity gradients
function warp_none(grid)
  γx = device_array(dev)(zeros(grid.nx, grid.ny))
  γy = γ₀ * device_array(dev)(ones(grid.nx, grid.ny))
  return γx, γy
end

### Problem setup
γx, γy = warp_none(grid)
NSprob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ν, nν, μ, nμ, dt, stepper=stepper, calcF=calcF!, stochastic=true)
TwoDNavierStokes.set_ζ!(NSprob, device_array(dev)(zeros(grid.nx, grid.ny)))
ADprob = TracerAdvection.Problem(NSprob; νc=νc, nνc=nνc, e=e, τc=τc, γx=γx, γy=γy, stepper)

# Some shortcuts for the advection-diffusion problem:
sol, clock, vars, params, grid = ADprob.sol, ADprob.clock, ADprob.vars, ADprob.params, ADprob.grid
x, y = grid.x, grid.y

# Set tracer initial conditions
profile(x, y, σ) = c_init
amplitude, spread = 1.0, 0.3
c₀ = device_array(dev)([amplitude * profile(x[i], y[j], spread) for j = 1:grid.ny, i = 1:grid.nx])
TracerAdvection.set_c!(ADprob, c₀)

### Diagnostics
E = Diagnostic(TwoDNavierStokes.energy, params.base_prob; nsteps) # energy
Z = Diagnostic(TwoDNavierStokes.enstrophy, params.base_prob; nsteps) # enstrophy
diags = [E, Z] # a list of Diagnostics passed to `stepforward!` will  be updated every timestep.

# Makie
c⁻ = Observable(Array(vars.c))
ζ = Observable(Array(params.base_prob.vars.ζ))
title_ζ = Observable("vorticity, μ t=" * @sprintf("%.2f", μ * clock.t))
energy = Observable(Point2f[(μ * E.t[1], E.data[1])])
enstrophy = Observable(Point2f[(μ * Z.t[1], Z.data[1] / forcing_wavenumber^2)])

fig = Figure(resolution=(3200, 1440))
axζ = Axis(fig[1, 1];
  ylabel="y",
  title=title_ζ,
  aspect=1,
  limits=((-L / 2, L / 2), (-L / 2, L / 2)))
axc = Axis(fig[1, 2];
  xlabel="x",
  ylabel="y",
  title="saturation deficit",
  aspect=1,
  limits=((-L / 2, L / 2), (-L / 2, L / 2)))
CairoMakie.heatmap!(axζ, x, y, ζ;
  colormap=:balance, colorrange=(-40, 40))
CairoMakie.heatmap!(axc, x, y, c⁻;
  colormap=:balance, colorrange=(0, 20))

# Solution!
startwalltime = time()
frames = ProgressBar(0:(nsteps ÷ nsubs))

# Storage array
solution = zeros(Float32, n, n, 2, length(frames))
CairoMakie.record(fig, "two_dimensional_turbulence_with_condensation.mp4", frames, framerate=25) do j
  # terminal update
  cfl = clock.dt * maximum([maximum(params.base_prob.vars.u) / grid.dx, maximum(params.base_prob.vars.v) / grid.dy])
  log = @sprintf("t: %d, cfl: %.2f, E: %.1f, Z: %.1f", clock.t, cfl, E.data[E.i], Z.data[Z.i])
  set_description(frames, log)

  # Store data
  solution[:, :, :, j+1] = cat(Float32.(Array(vars.c))[:, :, :], Float32.(Array(params.base_prob.vars.ζ))[:, :, :], dims=3)

  # Diags
  c⁻[] = vars.c
  ζ[] = params.base_prob.vars.ζ
  energy[] = push!(energy[], Point2f(μ * E.t[E.i], E.data[E.i]))
  enstrophy[] = push!(enstrophy[], Point2f(μ * Z.t[E.i], Z.data[Z.i] / forcing_wavenumber^2))
  title_ζ[] = "vorticity, μ t=" * @sprintf("%.2f", μ * clock.t)

  # Step!
  for _ in 1:nsubs
    stepforward!(ADprob, 1)
    TracerAdvection.updatevars!(ADprob)
    stepforward!(params.base_prob, diags, 1)
    TwoDNavierStokes.updatevars!(params.base_prob)
  end
end

# Further analysis
for ch in 1:2
  # Autocorrelation time analysis
  # We'll look at the center row  
  vals = zeros(size(solution)[end],n)
  for i in 1:size(solution)[end]
      vals[i,:] .= solution[n ÷ 2,:,ch,i]
  end
  lags = [0:100...]
  ac = StatsBase.autocor(vals, lags; demean = true)
  mean_ac = mean(ac, dims = 2)[:]
  Plots.plot(lags * dt_save, mean(ac, dims = 2)[:], label = "", ylabel = "Autocorrelation Coeff", xlabel = "Lag (time)", xlim = [0,100])
  Plots.savefig(string("ac_channel_$(ch)_","$(n)_$(ν)_$(nν)_$(μ)_$(nμ)_$(ε)_$(γ₀)_$(e)_$(τc)_$(dt)_$(dt_save)",".png"))

  # Single pixel
  p528 = [solution[n ÷ 2,n ÷ 2,ch, k] for k in 1:size(solution)[end]]
  Plots.histogram(p528, label = "", ylabel = "Frequency", xlabel = "Value")
  Plots.savefig(string("single_pixel_channel_$(ch)_","$(n)_$(ν)_$(nν)_$(μ)_$(nμ)_$(ε)_$(γ₀)_$(e)_$(τc)_$(dt)_$(dt_save)",".png"))  
end

# Append to HDF5
fname = "./two_dimensional_turbulence_with_condensation.hdf5"
fid = h5open(fname, "cw")
fid["$(n)_$(ν)_$(nν)_$(μ)_$(nμ)_$(ε)_$(γ₀)_$(e)_$(τc)_$(dt)_$(dt_save)"] = solution
close(fid)
