using Random: Random, rand
using StaticArrays: StaticArrays
using Profile: Profile, @profile
using BenchmarkTools: @benchmark, @btime
# using PProf: PProf, pprof
using PyPrint: pprint, @pprint
using NPZ: npzread
using KernelFunctions: KernelFunctions as Kernels, ColVecs
using Sensors: Sensors

rng = Random.seed!(1)

# kernel = Kernels.MaternKernel(ν=2.5)
kernel = Kernels.Matern52Kernel()
s = 75

# read problem and solution from Python code
X = ColVecs(npzread("data/entropy_X.npy")')
ans = npzread("data/entropy_indexes.npy")

indexes = Sensors.entropynaive(X, kernel, s)
# pprint(indexes', ans')
@assert indexes .- 1 == ans "entropy naive wrong"
indexes = Sensors.entropy(X, kernel, s)
@assert indexes .- 1 == ans "entropy chol  wrong"

X = ColVecs(npzread("data/mi_X.npy")')
ans = npzread("data/mi_indexes.npy")

indexes = Sensors.miprec(X, kernel, s)
@assert indexes .- 1 == ans "mi prec wrong"
indexes = Sensors.mi(X, kernel, s)
@assert indexes .- 1 == ans "mi chol wrong"

# exit()

# benchmarking

X = ColVecs(rand(rng, 3, 10_000))
s = 200

Profile.clear_malloc_data()
Sensors.entropynaive(X, kernel, s)
# Profile.print()
@time Sensors.entropynaive(X, kernel, s)
# @profile Sensors.entropynaive(X, kernel, s)

Profile.clear_malloc_data()
Sensors.entropy(X, kernel, s)
@time Sensors.entropy(X, kernel, s)

X = ColVecs(rand(rng, 3, 1000))
s = 100

# Profile.clear_malloc_data()
# Sensors.miprec(X, kernel, s)
# @time Sensors.miprec(X, kernel, s)

Profile.clear_malloc_data()
Sensors.mi(X, kernel, s)
@time Sensors.mi(X, kernel, s)
