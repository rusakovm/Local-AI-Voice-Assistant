using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Pipes;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;
using System.Threading.Tasks;

internal static class Program
{
    private const string PipeName = "LLMPipe";

    private static readonly Uri OllamaBase = new("http://127.0.0.1:11434");
    private static readonly HttpClient Http = new() { BaseAddress = OllamaBase };

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = false
    };

    public static async Task Main()
    {
        Console.OutputEncoding = Encoding.UTF8;

        while (true)
        {
            try
            {
                using var pipe = new NamedPipeClientStream(".", PipeName, PipeDirection.InOut, PipeOptions.Asynchronous);
                Console.Write("LLMWorker: подключаюсь... ");
                await pipe.ConnectAsync(5000);
                Console.WriteLine("OK ✅");

                using var reader = new StreamReader(pipe, Encoding.UTF8, false, 4096, leaveOpen: true);
                using var writer = new StreamWriter(pipe, new UTF8Encoding(false), 4096, leaveOpen: true) { AutoFlush = true };

                var outChan = Channel.CreateUnbounded<string>(new UnboundedChannelOptions
                {
                    SingleReader = true,
                    SingleWriter = false
                });

                var writerTask = Task.Run(async () =>
                {
                    await foreach (var s in outChan.Reader.ReadAllAsync())
                        await writer.WriteLineAsync(s);
                });

                while (pipe.IsConnected)
                {
                    var line = await reader.ReadLineAsync();
                    if (line == null) break;

                    line = line.Trim();
                    if (line.Length == 0) continue;

                    Msg msg;
                    try
                    {
                        msg = JsonSerializer.Deserialize<Msg>(line, JsonOpts) ?? throw new Exception("bad json");
                    }
                    catch
                    {
                        await outChan.Writer.WriteAsync(JsonSerializer.Serialize(new Msg
                        {
                            Type = "llm.chunk",
                            Id = "unknown",
                            Done = true,
                            Error = "Bad request JSON"
                        }, JsonOpts));
                        continue;
                    }

                    if (msg.Type != "llm.request")
                        continue;

                    _ = Task.Run(() => HandleRequest(msg, outChan.Writer));
                }

                outChan.Writer.TryComplete();
                try { await writerTask; } catch { }
            }
            catch (Exception ex)
            {
                Console.WriteLine("LLMWorker error: " + ex.Message);
                await Task.Delay(1000);
            }
        }
    }

    private sealed class ChatMsg
    {
        public string Role { get; set; } = "";
        public string Content { get; set; } = "";
    }

    private sealed class Msg
    {
        public string Type { get; set; } = "";
        public string Id { get; set; } = "";

        public string Model { get; set; } = "";
        public string System { get; set; } = "";
        public string Prompt { get; set; } = "";

        public List<ChatMsg>? History { get; set; }

        public string Delta { get; set; } = "";
        public bool Done { get; set; }
        public string Error { get; set; } = "";
    }

    private static async Task HandleRequest(Msg req, ChannelWriter<string> outWriter)
    {
        var id = string.IsNullOrWhiteSpace(req.Id) ? Guid.NewGuid().ToString("N") : req.Id;
        var model = string.IsNullOrWhiteSpace(req.Model) ? "llama3.1" : req.Model;

        var messages = new List<Dictionary<string, string>>();

        if (!string.IsNullOrWhiteSpace(req.System))
            messages.Add(new Dictionary<string, string> { ["role"] = "system", ["content"] = req.System });

        if (req.History != null)
        {
            foreach (var h in req.History)
            {
                if (h == null) continue;
                var role = (h.Role ?? "").Trim();
                var content = (h.Content ?? "").Trim();
                if (content.Length == 0) continue;

                if (role == "user" || role == "assistant")
                    messages.Add(new Dictionary<string, string> { ["role"] = role, ["content"] = content });
            }
        }

        messages.Add(new Dictionary<string, string> { ["role"] = "user", ["content"] = req.Prompt ?? "" });

        var payload = new
        {
            model,
            stream = true,
            keep_alive = "30m",
            options = new
            {
                num_ctx = 2048,
                num_predict = 300
            },
            messages
        };

        using var httpReq = new HttpRequestMessage(HttpMethod.Post, "/api/chat");
        httpReq.Content = new StringContent(JsonSerializer.Serialize(payload, JsonOpts), Encoding.UTF8, "application/json");
        httpReq.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));

        try
        {
            using var resp = await Http.SendAsync(httpReq, HttpCompletionOption.ResponseHeadersRead);
            resp.EnsureSuccessStatusCode();

            await using var stream = await resp.Content.ReadAsStreamAsync();
            using var sr = new StreamReader(stream, Encoding.UTF8);

            while (!sr.EndOfStream)
            {
                var line = await sr.ReadLineAsync();
                if (string.IsNullOrWhiteSpace(line)) continue;

                try
                {
                    using var doc = JsonDocument.Parse(line);
                    var root = doc.RootElement;

                    var done = root.TryGetProperty("done", out var doneEl) && doneEl.GetBoolean();

                    var delta = "";
                    if (root.TryGetProperty("message", out var msg) &&
                        msg.TryGetProperty("content", out var contentEl))
                        delta = contentEl.GetString() ?? "";

                    await outWriter.WriteAsync(JsonSerializer.Serialize(new Msg
                    {
                        Type = "llm.chunk",
                        Id = id,
                        Delta = delta,
                        Done = done
                    }, JsonOpts));

                    if (done) break;
                }
                catch
                {
                    // ignore
                }
            }
        }
        catch (Exception ex)
        {
            await outWriter.WriteAsync(JsonSerializer.Serialize(new Msg
            {
                Type = "llm.chunk",
                Id = id,
                Done = true,
                Error = ex.Message
            }, JsonOpts));
        }
    }
}
