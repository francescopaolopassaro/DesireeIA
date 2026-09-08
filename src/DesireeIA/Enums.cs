namespace DesireeIA;

public enum ModelFormat
{
    Unknown = 0,
    Gguf = 1,
    Safetensors = 2
}

public enum InferenceBackend
{
    Cpu = 1,
    Cuda = 2,
    Metal = 3,
    Vulkan = 4,
    Intel = 5,
    Axelera = 6
}

public enum Quantization
{
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q8_0 = 4,
    Q4K = 5,
    Q5K = 6,
    Q6K = 7
}
