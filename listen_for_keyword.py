import sounddevice as sd
import queue
import json
import os
import sys
import tempfile
from vosk import Model, KaldiRecognizer

class AudioRecorder:
    def __init__(self):
        self.audio_buffer = bytearray()
    def push_and_render(self, matched_text, raw_data):
        self.audio_buffer.extend(raw_data)
    def save_wav(self, filename, samplerate=16000):
        import wave
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(samplerate)
            wf.writeframes(self.audio_buffer)


class VoskModel:
    def __init__(self, model_path, sample_rate=16000):
        if not os.path.exists(model_path):
            print("Please download the model from https://alphacephei.com/vosk/models and unpack as 'vosk' in your CG cache folder.")
            sys.exit(1)
        self.model = Model(model_path)
        self.sample_rate = sample_rate
    
    def listen_for_keywords(self, 
        keywords, 
        foreach_loop_callback=[lambda iteration, data : None], 
        keyword_miss_callback=[lambda text, data: None], 
        keyword_hit_callback=[lambda text, data : None],
        record=False
    ):
        KEYWORDS = keywords

        if record:
            rec = AudioRecorder()
            foreach_loop_callback.append(rec.push_and_render)

        q = queue.Queue()
        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            q.put(bytes(indata))

        # Grammar: only listen for these words
        grammar = json.dumps(KEYWORDS)
        recognizer = KaldiRecognizer(self.model, self.sample_rate, grammar)

        print("Listening for keywords:", KEYWORDS)

        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        ):
            iteration = 0
            while True:
                #print("tick")
                data = q.get()
                for callback in foreach_loop_callback:
                    callback(iteration, data)
                iteration += 1
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "")
                    for callback in keyword_hit_callback:
                        callback(text, data)
                    if text:
                        print("🔥 Keyword detected:", text)
                        if record:
                            with tempfile.NamedTemporaryFile(dir="/tmp", suffix=".wav", delete=False) as tmp:
                                rec.save_wav(tmp.name)
                            return text, tmp.name
                        else:
                            return text
                else:
                    # Optional: partial results
                    partial = json.loads(recognizer.PartialResult()).get("partial", "")
                    if partial:
                        print("…", partial)
                    for callback in keyword_miss_callback:
                        callback(partial, data)
