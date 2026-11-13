using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Pipes;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using NAudio.Wave;

namespace MainHost;

internal static class Program
{
    private const string LlmPipeName = "LLMPipe";
    private const string VoicePipeName = "VoicePipe";
    private const string TtsPipeName = "TTSPipe";

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = false
    };

    // ===== LLM pipe =====
    private static NamedPipeServerStream? _llmPipe;
    private static StreamReader? _llmReader;
    private static StreamWriter? _llmWriter;
    private static readonly object _llmSendLock = new();
    private static volatile bool _llmConnected;

    // ===== Voice pipe =====
    private static volatile bool _voiceConnected;

    // ===== TTS pipe =====
    private static NamedPipeServerStream? _ttsPipe;
    private static StreamReader? _ttsReader;
    private static StreamWriter? _ttsWriter;
    private static readonly object _ttsSendLock = new();
    private static volatile bool _ttsConnected;

    // ===== State (barge-in) =====
    private static readonly object _stateLock = new();
    private static string? _activeChatId;
    private static bool _llmBusy;

    // ===== metrics =====
    private static DateTime _reqStartUtc;
    private static DateTime? _firstDeltaUtc;
    private static int _answerChars;

    // voice metrics
    private static double? _lastAsrSec;
    private static DateTime? _asrFinalUtc;

    // ===== history =====
    private sealed class ChatMsg
    {
        public string Role { get; set; } = "";
        public string Content { get; set; } = "";
    }

    private static readonly object _histLock = new();
    private static readonly List<ChatMsg> _history = new();
    private static int _ctxKeepMessages = 12;

    private static readonly object _respBufLock = new();
    private static readonly StringBuilder _respBuf = new();

    // ===== warmup ignore =====
    private static readonly object _ignoreLock = new();
    private static readonly HashSet<string> _ignoredIds = new();

    // ===== TTS playback control (NAudio) =====
    private static readonly object _playLock = new();
    private static WaveOutEvent? _waveOut;
    private static AudioFileReader? _audioReader;
    private static string? _activeTtsId;

    private const string DefaultSystemPrompt =
        "Ты голосовой ассистент. Пользователь говорит по-русски. " +
        "Отвечай по-русски, без лишних приветствий и без воды. " +
        "Не задавай лишних вопросов. Если запрос непонятен — задай ОДИН уточняющий вопрос. " +
        "Не более 500 символов.";

    public static async Task Main()
    {
        Console.OutputEncoding = Encoding.UTF8;

        _ = Task.Run(AcceptLlmWorkerLoop);
        _ = Task.Run(AcceptVoiceLoop);
        _ = Task.Run(AcceptTtsWorkerLoop);

        while (true)
        {
            Console.WriteLine();
            Console.WriteLine("=== MainHost ===");
            Console.WriteLine("1) Чат с LLaMA (Ollama)");
            Console.WriteLine("2) Статус модулей");
            Console.WriteLine("3) Выход");
            Console.Write("Выбор: ");

            var choice = (Console.ReadLine() ?? "").Trim();

            if (choice == "1")
                await ChatMode();
            else if (choice == "2")
                ShowStatus();
            else if (choice == "3")
                return;
        }
    }

    private static void ShowStatus()
    {
        Console.WriteLine();
        Console.WriteLine("=== Статус модулей ===");
        Console.WriteLine($"LLMWorker: {(_llmConnected ? "✅ подключен" : "⛔ не подключен")}");
        Console.WriteLine($"MicroWorker: {(_voiceConnected ? "✅ подключен" : "⛔ не подключен")}");
        Console.WriteLine($"TTSWorker: {(_ttsConnected ? "✅ подключен" : "⛔ не подключен")}");
        lock (_histLock)
            Console.WriteLine($"История: {_history.Count} сообщений, ctx keep: {_ctxKeepMessages}");
    }

    private static async Task ChatMode()
    {
        Console.WriteLine();
        Console.WriteLine("=== Чат ===");
        Console.WriteLine("Команды: /back, /clear, /sys <текст>, /reset, /ctx N");
        Console.WriteLine();

        var systemPrompt = DefaultSystemPrompt;

        while (true)
        {
            Console.Write("\nТы: ");
            var prompt = Console.ReadLine();
            if (prompt == null) return;

            prompt = prompt.Trim();
            if (prompt.Length == 0) continue;

            if (prompt.Equals("/back", StringComparison.OrdinalIgnoreCase))
                return;

            if (prompt.Equals("/clear", StringComparison.OrdinalIgnoreCase))
            {
                Console.Clear();
                continue;
            }

            if (prompt.StartsWith("/sys ", StringComparison.OrdinalIgnoreCase))
            {
                systemPrompt = prompt.Substring(5).Trim();
                Console.WriteLine("✅ System prompt обновлён.");
                continue;
            }

            if (prompt.Equals("/reset", StringComparison.OrdinalIgnoreCase))
            {
                lock (_histLock) _history.Clear();
                Console.WriteLine("✅ История очищена.");
                continue;
            }

            if (prompt.StartsWith("/ctx ", StringComparison.OrdinalIgnoreCase))
            {
                var s = prompt.Substring(5).Trim();
                if (int.TryParse(s, out var n) && n >= 0 && n <= 200)
                {
                    _ctxKeepMessages = n;
                    Console.WriteLine($"✅ Контекст: последние {_ctxKeepMessages} сообщений.");
                }
                else
                {
                    Console.WriteLine("⛔ Пример: /ctx 12 (0..200)");
                }
                continue;
            }

            if (!_llmConnected || _llmWriter == null)
            {
                Console.WriteLine("⛔ LLMWorker не подключен. Запусти LLMWorker.exe.");
                continue;
            }

            var id = Guid.NewGuid().ToString("N");

            lock (_stateLock)
            {
                _activeChatId = id;
                _llmBusy = true;
            }

            _reqStartUtc = DateTime.UtcNow;
            _firstDeltaUtc = null;
            _answerChars = 0;
            lock (_respBufLock) _respBuf.Clear();

            var req = new Msg
            {
                Type = "llm.request",
                Id = id,
                Model = "llama3.1",
                System = systemPrompt,
                Prompt = prompt,
                History = GetHistorySlice()
            };

            lock (_histLock)
                _history.Add(new ChatMsg { Role = "user", Content = prompt });

            var json = JsonSerializer.Serialize(req, JsonOpts);
            lock (_llmSendLock)
                _llmWriter.WriteLine(json);

            await WaitForDone(id);
        }
    }

    private static List<ChatMsg> GetHistorySlice()
    {
        lock (_histLock)
        {
            if (_ctxKeepMessages <= 0 || _history.Count == 0) return new List<ChatMsg>();
            var start = Math.Max(0, _history.Count - _ctxKeepMessages);
            var count = Math.Min(_ctxKeepMessages, _history.Count);
            return _history.GetRange(start, count);
        }
    }

    private static async Task WaitForDone(string id)
    {
        while (true)
        {
            string? cur;
            lock (_stateLock) cur = _activeChatId;
            if (cur != id) break;
            await Task.Delay(20);
        }
    }

    // ===== LLM accept/read =====

    private static async Task AcceptLlmWorkerLoop()
    {
        while (true)
        {
            try
            {
                Console.WriteLine("🧠 Ожидаю подключение LLMWorker...");

                _llmPipe = new NamedPipeServerStream(
                    LlmPipeName,
                    PipeDirection.InOut,
                    1,
                    PipeTransmissionMode.Byte,
                    PipeOptions.Asynchronous);

                await _llmPipe.WaitForConnectionAsync();

                _llmReader = new StreamReader(_llmPipe, Encoding.UTF8, false, 4096, leaveOpen: true);
                _llmWriter = new StreamWriter(_llmPipe, new UTF8Encoding(false), 4096, leaveOpen: true)
                {
                    AutoFlush = true
                };

                _llmConnected = true;
                Console.WriteLine("🧠 LLMWorker подключен ✅");

                SendWarmup();

                await ReadLlmLoop(_llmReader);
            }
            catch (Exception ex)
            {
                Console.WriteLine("❌ LLM pipe error: " + ex.Message);
            }
            finally
            {
                _llmConnected = false;
                try { _llmReader?.Dispose(); } catch { }
                try { _llmWriter?.Dispose(); } catch { }
                try { _llmPipe?.Dispose(); } catch { }
                _llmReader = null;
                _llmWriter = null;
                _llmPipe = null;

                lock (_stateLock)
                {
                    _activeChatId = null;
                    _llmBusy = false;
                }

                await Task.Delay(500);
            }
        }
    }

    private static void SendWarmup()
    {
        if (_llmWriter == null) return;

        var warmId = "warmup_" + Guid.NewGuid().ToString("N");
        lock (_ignoreLock) _ignoredIds.Add(warmId);

        var warmReq = new Msg
        {
            Type = "llm.request",
            Id = warmId,
            Model = "llama3.1",
            System = "Ответь одним словом: ОК",
            Prompt = "ОК",
            History = new List<ChatMsg>()
        };

        var json = JsonSerializer.Serialize(warmReq, JsonOpts);
        lock (_llmSendLock) _llmWriter.WriteLine(json);

        Console.WriteLine("🔥 Прогрев модели отправлен (скрыто).");
    }

    private static async Task ReadLlmLoop(StreamReader reader)
    {
        while (true)
        {
            var line = await reader.ReadLineAsync();
            if (line == null) break;

            line = line.Trim();
            if (line.Length == 0) continue;

            try
            {
                var msg = JsonSerializer.Deserialize<Msg>(line, JsonOpts);
                if (msg == null || msg.Type != "llm.chunk") continue;

                // ignore warmup
                lock (_ignoreLock)
                {
                    if (_ignoredIds.Contains(msg.Id))
                    {
                        if (msg.Done) _ignoredIds.Remove(msg.Id);
                        continue;
                    }
                }

                // печатаем только активный id
                string? activeId;
                lock (_stateLock) activeId = _activeChatId;

                if (string.IsNullOrEmpty(activeId)) continue;
                if (!string.Equals(msg.Id, activeId, StringComparison.Ordinal)) continue;

                if (!string.IsNullOrWhiteSpace(msg.Error))
                {
                    Console.WriteLine($"\n[LLM ERR] {msg.Error}");
                    lock (_stateLock)
                    {
                        _activeChatId = null;
                        _llmBusy = false;
                    }
                    continue;
                }

                if (!string.IsNullOrEmpty(msg.Delta))
                {
                    if (_firstDeltaUtc == null)
                        _firstDeltaUtc = DateTime.UtcNow;

                    _answerChars += msg.Delta.Length;
                    lock (_respBufLock) _respBuf.Append(msg.Delta);
                    Console.Write(msg.Delta);
                }

                if (msg.Done)
                {
                    var now = DateTime.UtcNow;

                    var firstTokenSec = _firstDeltaUtc.HasValue
                        ? (_firstDeltaUtc.Value - _reqStartUtc).TotalSeconds
                        : (now - _reqStartUtc).TotalSeconds;

                    var totalSec = (now - _reqStartUtc).TotalSeconds;
                    var cps = totalSec > 0.0001 ? (_answerChars / totalSec) : 0.0;

                    string assistantText;
                    lock (_respBufLock)
                    {
                        assistantText = _respBuf.ToString().Trim();
                        _respBuf.Clear();
                    }

                    if (!string.IsNullOrWhiteSpace(assistantText))
                    {
                        lock (_histLock)
                            _history.Add(new ChatMsg { Role = "assistant", Content = assistantText });
                    }

                    // E2E: от финала ASR до done
                    double? e2eSec = null;
                    if (_asrFinalUtc.HasValue)
                        e2eSec = (now - _asrFinalUtc.Value).TotalSeconds;

                    Console.WriteLine();
                    var asrPart = _lastAsrSec.HasValue ? $"ASR: {_lastAsrSec.Value:F2}с, " : "";
                    var e2ePart = e2eSec.HasValue ? $", E2E: {e2eSec.Value:F2}с" : "";
                    Console.WriteLine($"({asrPart}первый токен: {firstTokenSec:F2} сек, всего: {totalSec:F2} сек, скорость: {cps:F0} симв/сек{e2ePart})");

                    _lastAsrSec = null;
                    _asrFinalUtc = null;

                    // ===== отправляем в TTS =====
                    if (!string.IsNullOrWhiteSpace(assistantText))
                        SendToTts(assistantText);

                    lock (_stateLock)
                    {
                        if (_activeChatId == msg.Id)
                        {
                            _activeChatId = null;
                            _llmBusy = false;
                        }
                    }
                }
            }
            catch
            {
                // ignore
            }
        }
    }

    // ===== Voice accept/read =====

    private static async Task AcceptVoiceLoop()
    {
        while (true)
        {
            NamedPipeServerStream? pipe = null;
            StreamReader? reader = null;

            try
            {
                Console.WriteLine("🎙️ Ожидаю подключение MicroWorker...");

                pipe = new NamedPipeServerStream(
                    VoicePipeName,
                    PipeDirection.In,
                    1,
                    PipeTransmissionMode.Byte,
                    PipeOptions.Asynchronous);

                await pipe.WaitForConnectionAsync();
                _voiceConnected = true;
                Console.WriteLine("🎙️ MicroWorker подключен ✅");

                reader = new StreamReader(pipe, Encoding.UTF8, false, 4096, leaveOpen: true);

                while (pipe.IsConnected)
                {
                    var line = await reader.ReadLineAsync();
                    if (line == null) break;
                    line = line.Trim();
                    if (line.Length == 0) continue;

                    HandleAsrLine(line);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("❌ VoicePipe error: " + ex.Message);
            }
            finally
            {
                _voiceConnected = false;
                try { reader?.Dispose(); } catch { }
                try { pipe?.Dispose(); } catch { }
                await Task.Delay(500);
            }
        }
    }

    private static void HandleAsrLine(string line)
    {
        try
        {
            if (line.Length > 0 && line[0] == '{')
            {
                using var doc = JsonDocument.Parse(line);
                var root = doc.RootElement;

                var type = root.TryGetProperty("type", out var t) ? (t.GetString() ?? "") : "";
                var text = root.TryGetProperty("text", out var tx) ? (tx.GetString() ?? "") : "";
                text = (text ?? "").Trim();

                if (type == "asr.partial")
                    return;

                if (type == "asr.final")
                {
                    int asrMs = 0;
                    if (root.TryGetProperty("asrMs", out var msEl) && msEl.TryGetInt32(out var v))
                        asrMs = v;

                    string source = "";
                    if (root.TryGetProperty("source", out var srcEl))
                        source = srcEl.GetString() ?? "";

                    if (text.Length == 0) return;

                    _lastAsrSec = asrMs > 0 ? asrMs / 1000.0 : (double?)null;
                    _asrFinalUtc = DateTime.UtcNow;

                    var asrInfo = asrMs > 0 ? $"ASR {asrMs} ms" : "ASR ?";
                    if (!string.IsNullOrWhiteSpace(source)) asrInfo += $", src={source}";
                    Console.WriteLine($"\n🎙️ ASR: {text}  ({asrInfo})");

                    // ===== barge-in: остановить воспроизведение и отменить текущие “активные” id =====
                    StopPlayback();

                    lock (_stateLock)
                    {
                        _activeChatId = null;
                        _llmBusy = false;
                    }

                    SendPromptFromVoice(text);
                }

                return;
            }

            // legacy P:/F:
            if (line.StartsWith("P:", StringComparison.Ordinal)) return;

            var finalText = line.StartsWith("F:", StringComparison.Ordinal) ? line.Substring(2).Trim() : line.Trim();
            if (finalText.Length == 0) return;

            Console.WriteLine($"\n🎙️ ASR: {finalText}");

            StopPlayback();
            lock (_stateLock)
            {
                _activeChatId = null;
                _llmBusy = false;
            }

            SendPromptFromVoice(finalText);
        }
        catch
        {
            // ignore
        }
    }

    private static void SendPromptFromVoice(string prompt)
    {
        if (!_llmConnected || _llmWriter == null)
        {
            Console.WriteLine("⛔ LLMWorker не подключен. Голосовой запрос не отправлен.");
            return;
        }

        prompt = (prompt ?? "").Trim();
        if (prompt.Length == 0) return;

        var id = Guid.NewGuid().ToString("N");

        lock (_stateLock)
        {
            _activeChatId = id;
            _llmBusy = true;
        }

        _reqStartUtc = DateTime.UtcNow;
        _firstDeltaUtc = null;
        _answerChars = 0;
        lock (_respBufLock) _respBuf.Clear();

        lock (_histLock)
            _history.Add(new ChatMsg { Role = "user", Content = prompt });

        var req = new Msg
        {
            Type = "llm.request",
            Id = id,
            Model = "llama3.1",
            System = DefaultSystemPrompt,
            Prompt = prompt,
            History = GetHistorySlice()
        };

        var json = JsonSerializer.Serialize(req, JsonOpts);
        lock (_llmSendLock)
            _llmWriter.WriteLine(json);
    }

    // ===== TTS pipe server + response handling =====

    private static async Task AcceptTtsWorkerLoop()
    {
        while (true)
        {
            try
            {
                Console.WriteLine("🔊 Ожидаю подключение TTSWorker...");

                _ttsPipe = new NamedPipeServerStream(
                    TtsPipeName,
                    PipeDirection.InOut,
                    1,
                    PipeTransmissionMode.Byte,
                    PipeOptions.Asynchronous);

                await _ttsPipe.WaitForConnectionAsync();

                _ttsReader = new StreamReader(_ttsPipe, Encoding.UTF8, false, 4096, leaveOpen: true);
                _ttsWriter = new StreamWriter(_ttsPipe, new UTF8Encoding(false), 4096, leaveOpen: true)
                {
                    AutoFlush = true
                };

                _ttsConnected = true;
                Console.WriteLine("🔊 TTSWorker подключен ✅");

                while (_ttsPipe.IsConnected)
                {
                    var line = await _ttsReader.ReadLineAsync();
                    if (line == null) break;

                    line = line.Trim();
                    if (line.Length == 0) continue;

                    HandleTtsResp(line);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("❌ TTSPipe error: " + ex.Message);
            }
            finally
            {
                _ttsConnected = false;
                try { _ttsReader?.Dispose(); } catch { }
                try { _ttsWriter?.Dispose(); } catch { }
                try { _ttsPipe?.Dispose(); } catch { }
                _ttsReader = null;
                _ttsWriter = null;
                _ttsPipe = null;
                await Task.Delay(500);
            }
        }
    }

    private static void SendToTts(string text)
    {
        if (!_ttsConnected || _ttsWriter == null)
        {
            Console.WriteLine("⛔ TTSWorker не подключен.");
            return;
        }

        text = (text ?? "").Trim();
        if (text.Length == 0) return;

        var id = Guid.NewGuid().ToString("N");
        lock (_playLock) _activeTtsId = id;

        var req = new TtsReq { id = id, text = text };
        var json = JsonSerializer.Serialize(req, JsonOpts);

        Console.WriteLine($"🔊 TTS → req id={id} chars={text.Length}");

        lock (_ttsSendLock)
            _ttsWriter.WriteLine(json);
    }

    private static void HandleTtsResp(string jsonLine)
    {
        try
        {
            var resp = JsonSerializer.Deserialize<TtsResp>(jsonLine, JsonOpts);
            if (resp == null) return;

            string? active;
            lock (_playLock) active = _activeTtsId;

            if (active != null && resp.id != active)
            {
                // устаревшее
                return;
            }

            if (!resp.ok)
            {
                Console.WriteLine($"🔊 TTS ERR id={resp.id} cancelled={resp.cancelled} err={resp.error}");
                return;
            }

            if (string.IsNullOrWhiteSpace(resp.wavPath) || !File.Exists(resp.wavPath))
            {
                Console.WriteLine($"🔊 TTS ERR: wav not found: {resp.wavPath}");
                return;
            }

            Console.WriteLine($"🔊 TTS ✓ id={resp.id} ms={resp.ms} genMs={resp.genMs} cached={resp.cached} rtf={resp.rtf}");

            PlayWav(resp.wavPath);
        }
        catch (Exception ex)
        {
            Console.WriteLine("🔊 TTS JSON error: " + ex.Message);
        }
    }

    private static void PlayWav(string wavPath)
    {
        StopPlayback();

        lock (_playLock)
        {
            try
            {
                _audioReader = new AudioFileReader(wavPath);
                _waveOut = new WaveOutEvent();
                _waveOut.Init(_audioReader);
                _waveOut.Play();
            }
            catch (Exception ex)
            {
                Console.WriteLine("🔊 Play error: " + ex.Message);
                try { _waveOut?.Dispose(); } catch { }
                try { _audioReader?.Dispose(); } catch { }
                _waveOut = null;
                _audioReader = null;
            }
        }
    }

    private static void StopPlayback()
    {
        lock (_playLock)
        {
            try { _waveOut?.Stop(); } catch { }
            try { _waveOut?.Dispose(); } catch { }
            try { _audioReader?.Dispose(); } catch { }

            _waveOut = null;
            _audioReader = null;

            // сбрасываем active id, чтобы поздний ответ не сыграл
            _activeTtsId = null;
        }
    }

    // ===== models =====

    private sealed class Msg
    {
        public string Type { get; set; } = "";
        public string Id { get; set; } = "";

        // request
        public string Model { get; set; } = "";
        public string System { get; set; } = "";
        public string Prompt { get; set; } = "";
        public List<ChatMsg>? History { get; set; }

        // chunk
        public string Delta { get; set; } = "";
        public bool Done { get; set; }
        public string Error { get; set; } = "";
    }

    private sealed class TtsReq
    {
        public string id { get; set; } = "";
        public string text { get; set; } = "";
    }

    private sealed class TtsResp
    {
        public string id { get; set; } = "";
        public bool ok { get; set; }
        public string? wavPath { get; set; }
        public string? error { get; set; }

        public bool cancelled { get; set; }
        public bool cached { get; set; }
        public int ms { get; set; }
        public int genMs { get; set; }
        public double? wavSec { get; set; }
        public double? rtf { get; set; }
    }
}
