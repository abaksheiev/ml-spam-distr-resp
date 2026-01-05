
using System.Text.Json;
using System.Text.Json.Serialization;

public class Settings
{
    [JsonPropertyName("csv_model_path")]
    public string ModelPath { get; set; }
    public static Settings Instance
    {
        get
        {
            string json = File.ReadAllText("settings.json");
            return JsonSerializer.Deserialize<Settings>(json);
        }
    }
}
