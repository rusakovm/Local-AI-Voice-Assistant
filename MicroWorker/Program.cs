using System;
using System.IO;
using System.IO.Pipes;
using System.Text;
using System.Threading;
using NAudio.Wave;
using Newtonsoft.Json.Linq;
using Vosk;

internal static class Program
{
    static NamedPipeClientStream? pipeClient;
    static StreamWriter? pipeWriter;

    static DateTime _lastPartialSent = DateTime.MinValue;
    static DateTime? _utterStartUtc = null;

    // ===== VAD state =====
    static bool _inSpeech = false;
    static DateTime _lastVoiceUtc = DateTime.MinValue;
    static DateTime _speechHoldUntilUtc = DateTime.MinValue;

    // ===== endpointing / dedupe =====
    static string _lastPartial = "";
    static DateTime _lastFinalSentUtc = DateTime.MinValue;
    static string _lastFinalText = "";

    // ===== tuning =====
    const int SampleRate = 16000;
    const int PartialSendEveryMs = 250;

    const int EndSilenceMs = 320;
    const int FinalCooldownMs = 900;
    const int MinPartialForFinal = 6;

    const int SpeechHangoverMs = 250;

    // ===== Auto-threshold =====
    static int _thrEnter = 200;            // будет перекалибровано
    static int _thrExit = 130;             // будет вычислено от thrEnter

    static readonly DateTime _thrStartUtc = DateTime.UtcNow;
    static long _noiseSum = 0;
    static int _noiseCount = 0;
    static bool _thrCalibrated = false;

    // Защита от щелчков
    const int SilenceHoldMs = 80;
    static int _silenceAccumMs = 0;

    // Гистерезис (выход ниже входа)
    const double ExitRatio = 0.65;

    static readonly string ModelPath = @"Q:\ConsoleApp1\model\vosk-model-ru-0.42";

    static void Main()
    {
        Console.OutputEncoding = Encoding.UTF8;
        Vosk.Vosk.SetLogLevel(0);

        var model = new Model(ModelPath);
        var recognizer = new VoskRecognizer(model, SampleRate);

        EnsurePipeConnected();

        var waveIn = new WaveInEvent
        {
            DeviceNumber = 0,
            WaveFormat = new WaveFormat(SampleRate, 1),
            BufferMilliseconds = 100
        };

        waveIn.DataAvailable += (s, a) =>
        {
            try
            {
                int meanAbs = ComputeMeanAbs16(a.Buffer, a.BytesRecorded);
                var now = DateTime.UtcNow;

                // ==== калибровка (2 сек тишины) ====
                if (!_thrCalibrated)
                {
                    _noiseSum += meanAbs;
                    _noiseCount++;

                    if ((now - _thrStartUtc).TotalMilliseconds >= 2000)
                    {
                        var avg = _noiseCount > 0 ? (int)(_noiseSum / _noiseCount) : 0;

                        // вход в речь — выше шума
                        _thrEnter = avg * 6 + 120;           // БОЛЕЕ строгий вход, чем раньше
                        _thrEnter = Math.Clamp(_thrEnter, 140, 2000);

                        _thrExit = (int)(_thrEnter * ExitRatio);
                        _thrExit = Math.Clamp(_thrExit, 100, _thrEnter - 20);

                        _thrCalibrated = true;
                        Console.WriteLine($"🎚️ VAD calibrated: noise_avg={avg}, thrEnter={_thrEnter}, thrExit={_thrExit}");
                    }
                }

                // ===== VAD hysteresis =====
                bool voiceNow = _inSpeech
                    ? (meanAbs >= _thrExit)   // когда уже в речи — держим до более низкого порога
                    : (meanAbs >= _thrEnter); // вход в речь — по более высокому порогу

                if (voiceNow)
                {
                    _lastVoiceUtc = now;
                    _silenceAccumMs = 0;
                    _speechHoldUntilUtc = now.AddMilliseconds(SpeechHangoverMs);

                    if (!_inSpeech)
                    {
                        _inSpeech = true;
                        _utterStartUtc ??= now;
                    }
                }
                else
                {
                    // hangover
                    if (now <= _speechHoldUntilUtc)
                    {
                        _silenceAccumMs = 0;
                    }
                    else
                    {
                        _silenceAccumMs += waveIn.BufferMilliseconds;

                        if (_silenceAccumMs >= SilenceHoldMs && _inSpeech)
                        {
                            var silenceMs = (now - _lastVoiceUtc).TotalMilliseconds;
                            if (silenceMs >= EndSilenceMs)
                            {
                                TrySendEndpointFinal("vad");
                                ResetUtteranceState();
                            }
                        }
                    }
                }

                // ===== feed Vosk =====
                if (recognizer.AcceptWaveform(a.Buffer, a.BytesRecorded))
                {
                    HandleVoskFinal(recognizer.Result());
                }
                else
                {
                    var pjson = recognizer.PartialResult();
                    var pobj = JObject.Parse(pjson);
                    var p = (pobj["partial"]?.ToString() ?? "").Trim();

                    if (!string.IsNullOrWhiteSpace(p))
                    {
                        _utterStartUtc ??= now;
                        _lastPartial = p;

                        if ((now - _lastPartialSent).TotalMilliseconds >= PartialSendEveryMs)
                        {
                            _lastPartialSent = now;
                            SendJson(new JObject { ["type"] = "asr.partial", ["text"] = p });
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine("❌ ASR error: " + ex.Message);
            }
        };

        waveIn.StartRecording();
        Console.WriteLine("🎙️ Говорите... (ESC выход)");
        Console.WriteLine("🟡 Для калибровки VAD первые 2 сек лучше помолчать.");

        while (!(Console.KeyAvailable && Console.ReadKey(true).Key == ConsoleKey.Escape))
            Thread.Sleep(50);

        waveIn.StopRecording();
        pipeWriter?.Dispose();
        pipeClient?.Dispose();
    }

    static void ResetUtteranceState()
    {
        _inSpeech = false;
        _utterStartUtc = null;
        _lastPartial = "";
        _silenceAccumMs = 0;
        _lastPartialSent = DateTime.MinValue;
        _speechHoldUntilUtc = DateTime.MinValue;
    }

    static int ComputeMeanAbs16(byte[] buffer, int bytesRecorded)
    {
        long sum = 0;
        int count = 0;

        for (int i = 0; i + 1 < bytesRecorded; i += 2)
        {
            short s = (short)(buffer[i] | (buffer[i + 1] << 8));
            sum += Math.Abs((int)s);
            count++;
        }

        if (count == 0) return 0;
        return (int)(sum / count);
    }

    static void TrySendEndpointFinal(string source)
    {
        if ((DateTime.UtcNow - _lastFinalSentUtc).TotalMilliseconds < FinalCooldownMs)
            return;

        var text = (_lastPartial ?? "").Trim();
        if (text.Length < MinPartialForFinal)
            return;

        SendFinalText(text, source);
    }

    static void HandleVoskFinal(string resultJson)
    {
        if ((DateTime.UtcNow - _lastFinalSentUtc).TotalMilliseconds < FinalCooldownMs)
        {
            ResetUtteranceState();
            return;
        }

        try
        {
            var obj = JObject.Parse(resultJson);
            var text = (obj["text"]?.ToString() ?? "").Trim();
            if (string.IsNullOrWhiteSpace(text)) { ResetUtteranceState(); return; }

            if (IsNearDuplicate(text, _lastFinalText)) { ResetUtteranceState(); return; }

            _utterStartUtc ??= DateTime.UtcNow;
            SendFinalText(text, "vosk");
        }
        finally
        {
            ResetUtteranceState();
        }
    }

    static void SendFinalText(string text, string source)
    {
        text = (text ?? "").Trim();
        if (text.Length == 0) return;

        if (IsNearDuplicate(text, _lastFinalText))
            return;

        var now = DateTime.UtcNow;
        var asrMs = _utterStartUtc.HasValue ? (int)(now - _utterStartUtc.Value).TotalMilliseconds : 0;

        _lastFinalSentUtc = now;
        _lastFinalText = text;

        Console.WriteLine($"📤 {text}  (ASR {asrMs} ms, src={source}, thrEnter={_thrEnter}, thrExit={_thrExit})");

        var msg = new JObject
        {
            ["type"] = "asr.final",
            ["text"] = text,
            ["asrMs"] = asrMs,
            ["source"] = source
        };

        SendJson(msg);
    }

    static bool IsNearDuplicate(string a, string b)
    {
        a = (a ?? "").Trim();
        b = (b ?? "").Trim();
        if (a.Length == 0 || b.Length == 0) return false;
        if (string.Equals(a, b, StringComparison.OrdinalIgnoreCase)) return true;

        var al = a.ToLowerInvariant();
        var bl = b.ToLowerInvariant();
        if (al.Contains(bl) || bl.Contains(al))
        {
            var min = Math.Min(a.Length, b.Length);
            var max = Math.Max(a.Length, b.Length);
            return (max - min) <= 8;
        }

        return false;
    }

    static void SendJson(JObject obj)
        => SendLine(obj.ToString(Newtonsoft.Json.Formatting.None));

    static void SendLine(string line)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(line)) return;
            if (pipeClient == null || !pipeClient.IsConnected) EnsurePipeConnected();
            pipeWriter!.WriteLine(line);
        }
        catch
        {
            EnsurePipeConnected();
            pipeWriter!.WriteLine(line);
        }
    }

    static void EnsurePipeConnected()
    {
        while (true)
        {
            try
            {
                pipeClient?.Dispose();
                pipeClient = new NamedPipeClientStream(".", "VoicePipe", PipeDirection.Out);
                Console.Write("🔌 Connecting VoicePipe... ");
                pipeClient.Connect(5000);
                pipeWriter = new StreamWriter(pipeClient, new UTF8Encoding(false)) { AutoFlush = true };
                Console.WriteLine("✅ OK");
                break;
            }
            catch (Exception ex)
            {
                Console.WriteLine("⚠️ Connection failed: " + ex.Message + ". Retrying in 2s...");
                Thread.Sleep(2000);
            }
        }
    }
}
