from dataclasses import dataclass
from typing import Optional

from app.models.audio_model import AudioDownloadResult
from app.models.transcriber_model import TranscriptResult


@dataclass
class NoteResult:
    markdown: str                  # GPT 总结的 Markdown 内容
    transcript: TranscriptResult                # Whisper 转写结果
    audio_meta: AudioDownloadResult  # 音频下载的元信息（title、duration、封面等）
    model_name: Optional[str] = None   # 生成时使用的模型名称
    provider_id: Optional[str] = None  # 生成时使用的供应商 ID
    style: Optional[str] = None        # 生成时的笔记风格