module CheckDevices

open System
open OpenCL.Net

let random = Random()

let GenerateMatrix rows cols =
    Array.init (rows * cols) (fun _ -> float32 <| random.NextDouble())

let GetResultMatrixDims aRows aCols bRows bCols =
    if aCols <> bRows then
        failwith "Dimensions not match for multiplying"
    aRows, bCols

let Multiply (a:array<_>) aRows aCols (b:array<_>) bRows bCols (c:array<_>) =
    let cRows, cCols = GetResultMatrixDims aRows aCols bRows bCols
    for i in 0..cRows - 1 do
        for j in 0..cCols - 1 do
            let mutable sum = 0.f
            for k in 0..aCols - 1 do
                sum <- sum + a.[i * aCols + k] * b.[k * bCols + j]
            c.[i * cCols + j] <- sum

let GetDeviceName (device: Device) =
    let mutable error = ErrorCode()
    Cl.GetDeviceInfo(device, DeviceInfo.Name, &error).ToString()

let GetAllDevices =
    let mutable error = ErrorCode()
    seq {
        for platform in Cl.GetPlatformIDs(&error) do
            yield! Cl.GetDeviceIDs(platform, DeviceType.All, &error)
    }

let main =
    printfn $"All devices: %A{GetAllDevices |> Seq.map GetDeviceName}"
