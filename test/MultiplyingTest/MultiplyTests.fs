module MultiplyingTest

open System
open Brahma.OpenCL
open MatrixMultiplying
open NUnit.Framework
open Brahma.FSharp.OpenCL.Extensions

let CheckEqArrays a b =
    Assert.DoesNotThrow(fun _ -> CheckEq a b)

[<Test>]
let SimpleTest() =
    let provider, _, commandQueue, wgSize = GPU.Initialise None None
    let _, kernelPrepare, kernelRun, ndRange =
        GPU.GetCompiledMultiplyCommand provider wgSize 4 4 4 4

    let m = [|
        2f; 1f; 0f; 0f
        3f; 2f; 0f; 0f
        1f; 1f; 3f; 4f
        2f; -1f; 2f; 3f|]
    let revM = [|
        2f; -1f; 0f; 0f
        -3f; 2f; 0f; 0f
        31f; -19f; 3f; -4f
        -23f; 14f; -2f; 3f
    |]
    let expected = [|
        1f; 0f; 0f; 0f
        0f; 1f; 0f; 0f
        0f; 0f; 1f; 0f
        0f; 0f; 0f; 1f
    |]
    let actual = Array.zeroCreate(4 * 4)
    kernelPrepare ndRange m revM actual
    commandQueue.Add(kernelRun()).Finish()
                .Add(actual.ToHost provider).Finish() |> ignore
    CheckEqArrays expected actual
    GPU.Release provider commandQueue

[<Test>]
let StressTest() =
    let provider, _, commandQueue, wgSize = GPU.Initialise None None

    let iterations = 10
    let random = Random()
    for _ in 1..iterations do
        let a = random.Next(30, 70)
        let b = random.Next(30, 70)
        let c = random.Next(30, 70)
        let m1 = GenerateRandomMatrix random a b
        let m2 = GenerateRandomMatrix random b c
        let _, kernelPrepare, kernelRun, ndRange =
            GPU.GetCompiledMultiplyCommand provider wgSize a b b c
        
        let expected = Array.zeroCreate(a * c)
        CPU.Multiply m1 a b m2 b c expected
        
        let actual = Array.zeroCreate(a * c)
        kernelPrepare ndRange m1 m2 actual
        commandQueue.Add(kernelRun()).Finish()
                    .Add(actual.ToHost provider).Finish() |> ignore
        
        CheckEqArrays expected actual
    GPU.Release provider commandQueue
