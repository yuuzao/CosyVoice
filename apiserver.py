from pydantic import BaseModel
import sys
import io
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch


# zero_shot
prompt_speech_16k = load_wav('./asset/eng.wav', 16000)
cosyvoice2 = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)
ref_prompt = "And at the time I thought that so far as he was concerned it was a true story. He told it me with such a direct simplicity of conviction that I could not do otherwise than believe in him."

import torch
import random
import numpy as np
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
app = FastAPI()

class TextToSpeechRequest(BaseModel):
    text: str

# 请求示例
"""
import requests
import json

url = "http://localhost:8000/zero_shot"
payload = {
    "text": "你好,这是一段测试文本。"
}
headers = {
    'Content-Type': 'application/json'
}

response = requests.post(url, json=payload)
print(response.json())
"""

@app.post("/zero_shot")
def text_to_speech(request: TextToSpeechRequest):
    text = request.text
    output = io.BytesIO()
    try:
        # 返回音频文件
        model_output = cosyvoice2.inference_zero_shot(text, ref_prompt, prompt_speech_16k)
        audio_data = torch.concat([i['tts_speech'] for i in model_output], dim=1)
        torchaudio.save(output, audio_data, cosyvoice2.sample_rate, format="wav")
        output.seek(0)
        return StreamingResponse(output, media_type="audio/wav")

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=58000)
