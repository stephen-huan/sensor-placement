using Test: @testset, @test
using Random: Random, rand

using KernelFunctions: KernelFunctions as Kernels, ColVecs

using Sensors: Sensors

Random.seed!(1)

@testset "entropy" begin
    X = rand(3, 100)
    kernel = Kernels.MaternKernel(ν=2.5)
    s = 75

    @test Sensors.naiveentropy(X, kernel, s) == Sensors.entropy(X, kernel, s)
end
