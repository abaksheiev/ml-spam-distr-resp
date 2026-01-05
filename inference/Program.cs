using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

var settigns = Settings.Instance;


// Create ASP.NET Core minimal API
var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

// Load your ONNX spam model
var session = new InferenceSession(settigns.ModelPath);

// Shared spam check function
Func<string, object> checkSpam = (string emailText) =>
{
    // Replace "input" below with actual ONNX model's input name
    var inputTensor = new DenseTensor<string>(new[] { emailText }, new[] { 1, 1 });

    var inputs = new List<NamedOnnxValue>
    {
        NamedOnnxValue.CreateFromTensor("input", inputTensor)
    };

    using var results = session.Run(inputs);
    var score = results.First().AsEnumerable<long>().First();

    return new
    {
        spamProbability = score,
        isSpam = score > 0.5f
    };
};

// POST route — accepts JSON body: { "emailText": "..." }
app.MapPost("/predict", ([FromBody] EmailRequest req) =>
{
    return Results.Ok(checkSpam(req.EmailText));
});

app.Run();


public record EmailRequest(string EmailText);