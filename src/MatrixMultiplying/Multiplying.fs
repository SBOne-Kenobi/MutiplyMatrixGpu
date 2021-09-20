module MatrixMultiplying

open System
open OpenCL.Net
open Brahma.OpenCL
open Brahma.FSharp.OpenCL.Core
open Brahma.FSharp.OpenCL.Extensions

let random = Random()

let GenerateRandomMatrix rows cols =
    Array.init (rows * cols) (fun _ -> float32 <| random.NextDouble())

let GetResultMatrixDims aRows aCols bRows bCols =
    if aCols <> bRows then
        failwith "Matrix dims not match for multiplying"
    aRows, bCols

let GetDeviceName (device: Device) =
    let mutable err = ErrorCode()
    Cl.GetDeviceInfo(device, DeviceInfo.Name, &err).ToString()

let IterateWithAvgTime iterations body =
    let start = DateTime.Now
    for i in 1..iterations do
        body()
    (DateTime.Now - start).TotalMilliseconds / float iterations

let RoundUp value div = (value + div - 1) / div * div

let CheckEq (a:array<float32>) (b:array<float32>) =
    let aSize = a.GetLength 0
    let bSize = b.GetLength 0
    if aSize <> bSize then
        failwith $"CheckEq: sizes not match {aSize} <> {bSize}"
    for i in 0..aSize - 1 do
        let error = Math.Abs(a.[i] - b.[i])
        if error > 0.01f then
            failwith $"CheckEq: too match error ${error}"

module CPU =
    let Multiply (a:array<_>) aRows aCols (b:array<_>) bRows bCols (c:array<_>) =
        let cRows, cCols = GetResultMatrixDims aRows aCols bRows bCols
        for i in 0..cRows - 1 do
            for j in 0..cCols - 1 do
                let mutable sum = 0.f
                for k in 0..aCols - 1 do
                    sum <- sum + a.[i * aCols + k] * b.[k * bCols + j]
                c.[i * cCols + j] <- sum
                
    let Run iterations (a:array<_>) aRows aCols (b:array<_>) bRows bCols =
        let cRows, cCols = GetResultMatrixDims aRows aCols bRows bCols
        let c = Array.zeroCreate(cRows * cCols)
        let time = IterateWithAvgTime iterations (fun _ -> Multiply a aRows aCols b bRows bCols c)
        c, time

module GPU =
    
    let Initialise (platformName: String option) (deviceType: DeviceType option) =
        let provider =
            try
                ComputeProvider.Create(platformName |> Option.defaultWith (fun _ -> "*"),
                                       deviceType |> Option.defaultWith (fun _ -> DeviceType.Default))
            with
            | e -> failwith e.Message
        
        let device = Seq.head provider.Devices
        let mutable err = ErrorCode()
        let wgSize = Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkGroupSize, &err).CastTo<int>()
                     |> float32 |> sqrt |> floor |> int32 |> max 1
        
        let commandQueue = new CommandQueue(provider, device)
        
        provider, device, commandQueue, wgSize
    
    let GetMultiplyCommand wgSize aRows aCols bRows bCols cRows cCols=
        let localArrMultiplySize = wgSize * wgSize
        <@
             fun (r:_2D) (a:array<float32>) (b:array<float32>) (res:array<float32>) -> 
                let lid0 = r.LocalID0
                let lid1 = r.LocalID1
                let gid0 = r.GlobalID0
                let gid1 = r.GlobalID1

                let localA = local(Array.zeroCreate(localArrMultiplySize))
                let localB = local(Array.zeroCreate(localArrMultiplySize))
                let localRes = local(Array.zeroCreate(localArrMultiplySize))
                localRes.[lid1 * wgSize + lid0] <- 0.f
                
                let mutable off = 0
                while off < aCols do
                    if lid0 + off < aCols && gid1 < aRows then
                        localA.[lid1 * wgSize + lid0] <- a.[gid1 * aCols + lid0 + off]
                    else
                        localA.[lid1 * wgSize + lid0] <- 0.f
                    if lid1 + off < bRows && gid0 < bCols then
                        localB.[lid1 * wgSize + lid0] <- b.[(lid1 + off) * bCols + gid0]
                    else
                        localB.[lid1 * wgSize + lid0] <- 0.f

                    barrier()
                    
                    let mutable sum = localRes.[lid1 * wgSize + lid0]
                    for i in 0..wgSize - 1 do
                        sum <- sum + localA.[lid1 * wgSize + i] * localB.[i * wgSize + lid0]
                    localRes.[lid1 * wgSize + lid0] <- sum                                        
                    off <- off + wgSize
                    
                    barrier()
                
                if gid0 < cCols && gid1 < cRows then
                    res.[gid1 * cCols + gid0] <- localRes.[lid1 * wgSize + lid0]
        @>
        
    let GetCompiledMultiplyCommand (provider: ComputeProvider) wgSize aRows aCols bRows bCols =
        let cRows, cCols = GetResultMatrixDims aRows aCols bRows bCols
        let command = GetMultiplyCommand wgSize aRows aCols bRows bCols cRows cCols
        let kernel, kernelPrepare, kernelRun = provider.Compile command
        let ndRange = _2D(RoundUp cRows wgSize, RoundUp cCols wgSize, wgSize, wgSize)
        kernel, kernelPrepare, kernelRun, ndRange

    let Run iterations (a:array<_>) aRows aCols (b:array<_>) bRows bCols =
        let cRows, cCols = GetResultMatrixDims aRows aCols bRows bCols
        let c = Array.zeroCreate(cRows * cCols)

        let provider, _, commandQueue, wgSize = Initialise None None
        let _, kernelPrepare, kernelRun, ndRange =
            GetCompiledMultiplyCommand provider wgSize aRows aCols bRows bCols

        kernelPrepare ndRange a b c
        let time = IterateWithAvgTime iterations (fun _ -> commandQueue.Add(kernelRun()).Finish() |> ignore)
        commandQueue.Add(c.ToHost provider).Finish() |> ignore
        
        c, time

    let Release (provider: ComputeProvider) (commandQueue: CommandQueue) =
        commandQueue.Dispose()
        provider.CloseAllBuffers()
        provider.Dispose()

let main =
    let iterations = 100
    let size = 100
    let m1 = GenerateRandomMatrix size size
    let m2 = GenerateRandomMatrix size size
    
    printfn $"Multiplying matrices {size}x{size} {iterations} times on CPU"
    let cpuRes, cpuTime = CPU.Run iterations m1 size size m2 size size    
    printfn "Done"
    
    printfn $"Multiplying matrices {size}x{size} {iterations} times on GPU"
    let gpuRes, gpuTime = GPU.Run iterations m1 size size m2 size size
    printfn "Done"
    
    printfn $"CPU avg time: %A{cpuTime}ms"
    printfn $"GPU avg time: %A{gpuTime}ms"
    
    printfn "Verifying"
    CheckEq cpuRes gpuRes
    printfn "Done"
    
    printfn "--------------"
    
    let iterations = 100
    let size = 1000
    let m1 = GenerateRandomMatrix size size
    let m2 = GenerateRandomMatrix size size
    
    printfn $"Multiplying matrices {size}x{size} {iterations} times on GPU"
    let _, gpuTime = GPU.Run iterations m1 size size m2 size size
    printfn "Done"
    
    printfn $"GPU avg time: %A{gpuTime}ms"
