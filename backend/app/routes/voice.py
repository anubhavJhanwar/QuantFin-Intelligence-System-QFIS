"""
/api/voice — speech-to-text endpoint.
Accepts audio file, transcribes using transformers Whisper, returns text + answer.
"""

import io
import time
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException
from loguru import logger

from app.services.inference import answer_query
from app.services.database import log_query

router = APIRouter(prefix="/api/voice", tags=["Voice"])

_whisper_model = None
_whisper_processor = None


def _load_whisper():
    global _whisper_model, _whisper_processor
    if _whisper_model is not None:
        return
    try:
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        logger.info("Loading Whisper tiny for STT...")
        _whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        _whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        _whisper_model.eval()
        logger.info("Whisper loaded")
    except Exception as e:
        logger.error(f"Failed to load Whisper: {e}")
        raise


@router.post("/transcribe")
async def transcribe_and_answer(audio: UploadFile = File(...)):
    """
    Accept audio file (wav/mp3/webm), transcribe with Whisper,
    then answer using the QA pipeline.
    """
    _load_whisper()

    import torch
    import librosa
    import numpy as np

    try:
        audio_bytes = await audio.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Load audio
        waveform, sr = librosa.load(tmp_path, sr=16000, mono=True)
        Path(tmp_path).unlink(missing_ok=True)

        # Transcribe
        inputs = _whisper_processor(
            waveform, sampling_rate=16000, return_tensors="pt"
        )
        with torch.no_grad():
            predicted_ids = _whisper_model.generate(inputs["input_features"])
        transcription = _whisper_processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        logger.info(f"Transcribed: {transcription}")

        if not transcription:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")

        # Answer
        start = time.time()
        result = answer_query(transcription, use_rag=True)
        latency = (time.time() - start) * 1000

        await log_query(
            query=transcription,
            answer=result["answer"],
            sources=result["sources"],
            model_type=result["model_type"],
            rag_used=result["rag_used"],
            latency_ms=latency,
            voice_input=True,
        )

        return {
            "transcription": transcription,
            "answer": result["answer"],
            "sources": result["sources"],
            "model_type": result["model_type"],
            "latency_ms": round(latency, 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
