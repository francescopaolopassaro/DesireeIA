using DesireeIA;
using Xunit;

namespace DesireeIA.Tests;

public class LocalModelInteropTest
{
    private static bool NativeAvailable()
    {
        try
        {
            _ = DesireeIAEngine.DetectHardware();
            return true;
        }
        catch (DllNotFoundException)
        {
            return false;
        }
        catch (InvalidOperationException)
        {
            return false;
        }
    }

    private static string? ModelPath() =>
        Environment.GetEnvironmentVariable("DESIREEIA_TEST_MODEL_PATH");

    [Fact]
    public void Predict_And_NextToken_RunOnRealModel()
    {
        if (!NativeAvailable()) return;
        var path = ModelPath();
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;

        var plan = DesireeIAEngine.BuildPlan(path);
        using var model = LocalModel.Load(path, plan);

        var first = model.Predict(new[] { 2 }); // BOS
        Assert.InRange(first, 0, int.MaxValue);

        var second = model.NextToken();
        Assert.InRange(second, 0, int.MaxValue);

        Assert.True(model.ContextSize() > 0);
    }
}
