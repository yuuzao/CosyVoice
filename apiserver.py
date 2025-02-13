from fastapi import FastAPI
from pydantic import BaseModel
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch

app = FastAPI()

# 初始化模型
prompt_speech_16k = load_wav('./output/eng.wav', 16000)
cosyvoice2 = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

class TTSRequest(BaseModel):
    text: str
    prompt: str = "Lionel Wallace told me this story of the Door in the Wall. And at the time I thought that so far as he was concerned it was a true story."

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        # 执行TTS推理
        audio_segments = []
        for segment in cosyvoice2.inference_zero_shot(request.text, request.prompt, prompt_speech_16k, stream=False):
            audio_segments.append(segment['tts_speech'])

        # 保存音频文件
        output_path = "output/generated.wav"
        torchaudio.save(output_path, audio_segments[0], cosyvoice2.sample_rate)

        return {"status": "success", "message": "Audio generated successfully", "file_path": output_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
