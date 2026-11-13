using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.IO.Pipes;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

internal static class Program
{
    private const string PipeName = "TTSPipe";

    // ПУТИ ПОД СЕБЯ:
    private static readonly string PiperExe = @"M:\piper_windows_amd64\piper\piper.exe";
    private static readonly string ModelOnnx = @"M:\piper_windows_amd64\piper\models\ru_RU-dmitri-medium.onnx";
    private static readonly string ModelJson = @"M:\piper_windows_amd64\piper\models\ru_RU-dmitri-medium.onnx.json";
    private static readonly string OutDir = @"G:\neyrosiza2\tts\out";

    // ===== JSON =====
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = false
    };

    // ===== Queue (single consumer) =====
    private static readonly BlockingCollection<TtsReq> Queue = new(new ConcurrentQueue<TtsReq>());
    private static long _seq = 0;

    // active synth
    private static readonly object ActiveLock = new();
    private static Process? _activeProc = null;
    private static string? _activeId = null;
    private static CancellationTokenSource? _activeCts = null;

    // cache (очень простой, по тексту)
    private static readonly ConcurrentDictionary<string, string> Cache = new(); // text -> wavPath
    private const int CacheMax = 200;

    private static StreamWriter? _writer; // shared writer (pipe)
    private static readonly object WriterLock = new();

    private static async Task Main()
    {
        Console.OutputEncoding = Encoding.UTF8;
        Directory.CreateDirectory(OutDir);

        Console.WriteLine("🔊 TTSWorker starting...");
        ValidateFiles();

        // single consumer task
        _ = Task.Run(ConsumerLoop);

        while (true)
        {
            try
            {
                using var pipe = new NamedPipeClientStream(".", PipeName, PipeDirection.InOut, PipeOptions.Asynchronous);
                Console.Write("🔌 Connecting to MainHost TTS pipe... ");
                await pipe.ConnectAsync(5000);
                Console.WriteLine("✅ OK");

                using var reader = new StreamReader(pipe, Encoding.UTF8, false, 4096, leaveOpen: true);
                using var writer = new StreamWriter(pipe, new UTF8Encoding(false), 4096, leaveOpen: true)
                {
                    AutoFlush = true
                };

                lock (WriterLock) _writer = writer;

                while (pipe.IsConnected)
                {
                    var line = await reader.ReadLineAsync();
                    if (line == null) break;

                    line = line.Trim();
                    if (line.Length == 0) continue;

                    TtsReq? req;
                    try
                    {
                        req = JsonSerializer.Deserialize<TtsReq>(line, JsonOpts);
                    }
                    catch (Exception ex)
                    {
                        await SendResp(new TtsResp { id = "unknown", ok = false, error = "Bad JSON: " + ex.Message });
                        continue;
                    }

                    if (req == null || string.IsNullOrWhiteSpace(req.id) || string.IsNullOrWhiteSpace(req.text))
                    {
                        await SendResp(new TtsResp { id = req?.id ?? "unknown", ok = false, error = "Bad request" });
                        continue;
                    }

                    req.text = Normalize(req.text);
                    req.seq = Interlocked.Increment(ref _seq);

                    // BARGE-IN: новый запрос отменяет предыдущий активный синтез
                    CancelActive("barge-in");

                    Console.WriteLine($"📥 [{req.seq}] TTS req id={req.id} chars={req.text.Length}");

                    Queue.Add(req);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("⚠️ Pipe error: " + ex.Message);
                await Task.Delay(600);
            }
            finally
            {
                lock (WriterLock) _writer = null;
            }
        }
    }

    private static void ValidateFiles()
    {
        if (!File.Exists(PiperExe)) Console.WriteLine("⚠️ PiperExe not found: " + PiperExe);
        if (!File.Exists(ModelOnnx)) Console.WriteLine("⚠️ ModelOnnx not found: " + ModelOnnx);
        if (!File.Exists(ModelJson)) Console.WriteLine("⚠️ ModelJson not found: " + ModelJson);
    }

    private static async Task ConsumerLoop()
    {
        foreach (var req in Queue.GetConsumingEnumerable())
        {
            // если за время ожидания пришёл новый запрос и старый уже не актуален —
            // мы всё равно синтезим только текущий, потому что barge-in отменяет активный
            // но очередь может содержать "устаревшие". Их можно отбрасывать, если нужно.
            // Пока оставим: выполняем в порядке.
            var swTotal = Stopwatch.StartNew();

            // cache hit
            if (TryGetCache(req.text, out var cachedPath) && File.Exists(cachedPath))
            {
                Console.WriteLine($"⚡ [{req.seq}] cache hit ({Path.GetFileName(cachedPath)})");
                await SendResp(new TtsResp { id = req.id, ok = true, wavPath = cachedPath, ms = (int)swTotal.ElapsedMilliseconds, cached = true });
                continue;
            }

            var cts = new CancellationTokenSource();
            lock (ActiveLock)
            {
                _activeCts = cts;
                _activeId = req.id;
            }

            var resp = await Synthesize(req, cts.Token);

            swTotal.Stop();

            // если запрос отменили — не шлём "успех" (чтобы MainHost не играл старый wav)
            if (resp.cancelled)
            {
                Console.WriteLine($"🛑 [{req.seq}] cancelled id={req.id} ({swTotal.ElapsedMilliseconds} ms)");
                // можно отправить ok=false cancelled, но лучше вообще не шуметь
                await SendResp(new TtsResp { id = req.id, ok = false, cancelled = true, ms = (int)swTotal.ElapsedMilliseconds, error = "Cancelled" });
                continue;
            }

            resp.ms = (int)swTotal.ElapsedMilliseconds;
            await SendResp(resp);

            if (resp.ok && !string.IsNullOrWhiteSpace(resp.wavPath))
                PutCache(req.text, resp.wavPath);

            lock (ActiveLock)
            {
                if (_activeId == req.id)
                {
                    _activeId = null;
                    _activeCts = null;
                    _activeProc = null;
                }
            }
        }
    }

    private static void CancelActive(string reason)
    {
        lock (ActiveLock)
        {
            try
            {
                _activeCts?.Cancel();
            }
            catch { }

            var p = _activeProc;
            if (p != null && !p.HasExited)
            {
                try
                {
                    Console.WriteLine($"🛑 Cancel active synth ({reason}) id={_activeId}");
                    p.Kill(entireProcessTree: true);
                }
                catch { }
            }
        }
    }

    private static async Task<TtsResp> Synthesize(TtsReq req, CancellationToken ct)
    {
        var started = DateTime.UtcNow;
        var sw = Stopwatch.StartNew();

        try
        {
            var wavPath = Path.Combine(OutDir, $"{DateTime.Now:yyyyMMdd_HHmmssfff}_{req.id}.wav");

            // Piper читает текст из stdin
            var psi = new ProcessStartInfo
            {
                FileName = PiperExe,
                Arguments = $"--model \"{ModelOnnx}\" --config \"{ModelJson}\" --output_file \"{wavPath}\"",
                RedirectStandardInput = true,
                RedirectStandardError = true,
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var proc = new Process { StartInfo = psi, EnableRaisingEvents = true };

            lock (ActiveLock)
                _activeProc = proc;

            Console.WriteLine($"▶️ [{req.seq}] synth start id={req.id} chars={req.text.Length}");

            proc.Start();

            // stdin UTF-8 без BOM
            await using (var swIn = new StreamWriter(proc.StandardInput.BaseStream, new UTF8Encoding(false)) { AutoFlush = true })
            {
                await swIn.WriteLineAsync(req.text);
            }

            // ждём завершение или отмену
            while (!proc.HasExited)
            {
                if (ct.IsCancellationRequested)
                {
                    TryKill(proc);
                    return new TtsResp { id = req.id, ok = false, cancelled = true };
                }

                await Task.Delay(20);
            }

            sw.Stop();

            var err = await proc.StandardError.ReadToEndAsync();
            if (ct.IsCancellationRequested)
                return new TtsResp { id = req.id, ok = false, cancelled = true };

            if (proc.ExitCode != 0 || !File.Exists(wavPath))
            {
                Console.WriteLine($"❌ [{req.seq}] piper failed code={proc.ExitCode} ms={sw.ElapsedMilliseconds}");
                return new TtsResp { id = req.id, ok = false, error = $"Piper failed ({proc.ExitCode}): {TrimErr(err)}" };
            }

            // (опционально) оценка длительности wav и RTF
            double? wavSec = TryGetWavSeconds(wavPath);
            double rtf = (wavSec.HasValue && wavSec.Value > 0.001) ? (sw.Elapsed.TotalSeconds / wavSec.Value) : -1;

            Console.WriteLine($"✅ [{req.seq}] synth done id={req.id} ms={sw.ElapsedMilliseconds}" +
                              (wavSec.HasValue ? $" wav={wavSec.Value:F2}s rtf={rtf:F2}" : ""));

            return new TtsResp
            {
                id = req.id,
                ok = true,
                wavPath = wavPath,
                genMs = (int)sw.ElapsedMilliseconds,
                wavSec = wavSec,
                rtf = (rtf >= 0 ? rtf : null)
            };
        }
        catch (Exception ex)
        {
            if (ct.IsCancellationRequested)
                return new TtsResp { id = req.id, ok = false, cancelled = true };

            Console.WriteLine($"❌ [{req.seq}] synth exception: {ex.Message}");
            return new TtsResp { id = req.id, ok = false, error = ex.Message };
        }
        finally
        {
            sw.Stop();
        }
    }

    private static void TryKill(Process p)
    {
        try
        {
            if (!p.HasExited)
                p.Kill(entireProcessTree: true);
        }
        catch { }
    }

    private static async Task SendResp(TtsResp resp)
    {
        StreamWriter? w;
        lock (WriterLock) w = _writer;
        if (w == null) return;

        var json = JsonSerializer.Serialize(resp, JsonOpts);
        try
        {
            lock (WriterLock)
                _writer?.WriteLine(json);
        }
        catch { await Task.CompletedTask; }
    }

    private static string Normalize(string s)
    {
        s = (s ?? "").Trim();
        s = s.Replace("\r", " ").Replace("\n", " ");
        while (s.Contains("  ", StringComparison.Ordinal)) s = s.Replace("  ", " ");
        return s;
    }

    private static string TrimErr(string err)
    {
        err = (err ?? "").Trim();
        if (err.Length > 600) err = err.Substring(0, 600) + "...";
        return err;
    }

    private static bool TryGetCache(string text, out string path)
    {
        path = "";
        if (string.IsNullOrWhiteSpace(text)) return false;
        return Cache.TryGetValue(text, out path!);
    }

    private static void PutCache(string text, string wavPath)
    {
        if (string.IsNullOrWhiteSpace(text) || string.IsNullOrWhiteSpace(wavPath)) return;

        if (Cache.Count >= CacheMax)
        {
            // простейший сброс, чтобы не усложнять LRU
            Cache.Clear();
        }

        Cache[text] = wavPath;
    }

    private static double? TryGetWavSeconds(string wavPath)
    {
        try
        {
            // очень простой WAV parser: читаем RIFF header и fmt/data chunks
            using var fs = File.OpenRead(wavPath);
            using var br = new BinaryReader(fs);

            // "RIFF" + size + "WAVE"
            if (new string(br.ReadChars(4)) != "RIFF") return null;
            br.ReadInt32();
            if (new string(br.ReadChars(4)) != "WAVE") return null;

            int sampleRate = 0;
            short channels = 0;
            short bitsPerSample = 0;
            int dataSize = 0;

            while (fs.Position + 8 <= fs.Length)
            {
                var id = new string(br.ReadChars(4));
                int size = br.ReadInt32();

                if (id == "fmt ")
                {
                    short audioFormat = br.ReadInt16();
                    channels = br.ReadInt16();
                    sampleRate = br.ReadInt32();
                    br.ReadInt32(); // byteRate
                    br.ReadInt16(); // blockAlign
                    bitsPerSample = br.ReadInt16();

                    // skip rest
                    var remain = size - 16;
                    if (remain > 0) br.ReadBytes(remain);

                    if (audioFormat != 1) { /* PCM expected */ }
                }
                else if (id == "data")
                {
                    dataSize = size;
                    br.ReadBytes(size);
                }
                else
                {
                    br.ReadBytes(size);
                }

                if (sampleRate > 0 && channels > 0 && bitsPerSample > 0 && dataSize > 0)
                    break;
            }

            if (sampleRate <= 0 || channels <= 0 || bitsPerSample <= 0 || dataSize <= 0) return null;

            int bytesPerSampleFrame = channels * (bitsPerSample / 8);
            if (bytesPerSampleFrame <= 0) return null;

            double frames = (double)dataSize / bytesPerSampleFrame;
            return frames / sampleRate;
        }
        catch
        {
            return null;
        }
    }

    private sealed class TtsReq
    {
        public string id { get; set; } = "";
        public string text { get; set; } = "";
        public long seq { get; set; } = 0;
    }

    private sealed class TtsResp
    {
        public string id { get; set; } = "";
        public bool ok { get; set; }
        public string? wavPath { get; set; }
        public string? error { get; set; }

        // extra logging/telemetry
        public bool cancelled { get; set; }
        public bool cached { get; set; }
        public int ms { get; set; }          // общий (очередь+генерация)
        public int genMs { get; set; }       // чисто piper
        public double? wavSec { get; set; }  // длительность wav
        public double? rtf { get; set; }     // genSec / wavSec
    }
}
