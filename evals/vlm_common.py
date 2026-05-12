"""Shared infrastructure for VLM evaluation (backends, parsing, I/O).

Public release variant — supports the released phyjudge LoRA served via vLLM
(OpenAIBackend), and three cloud judges so users can reproduce the closed-
source baselines without a GPU:

  qwen9b / qwen27b   OpenAI-compatible vLLM endpoint
  gemini             Gemini (AI Studio API key OR Vertex AI)
  gpt                OpenAI direct API (chat/completions with image frames)
  claude             Claude on Vertex AI (rawPredict, gcloud auth)
"""

from __future__ import annotations

import abc
import argparse
import base64
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from evals.eval_types import SCORED
from evals.physics_criteria import ALL_LAWS
from evals.prompts import GENERAL_KEYS

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Response parsing
# ──────────────────────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_JSON_RE_DOTALL = re.compile(r"\{.*\}", re.DOTALL)

try:
    import json_repair
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False


def _extract_balanced_json(text: str) -> str | None:
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]
            if depth < 0:
                depth = 0
    return None


def _try_parse_json(json_str: str) -> dict | None:
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass
    if _HAS_JSON_REPAIR:
        try:
            result = json_repair.loads(json_str)
            if isinstance(result, dict):
                logger.info("json_repair recovered truncated JSON")
                return result
        except Exception:
            pass
    return None


def parse_cot_response(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    rationale = ""
    m = _THINK_RE.search(text)
    if m:
        rationale = m.group(1).strip()
    after_think = text[m.end():] if m else text
    json_str = _extract_balanced_json(after_think)
    if not json_str and rationale:
        json_str = _extract_balanced_json(rationale)
    if not json_str:
        json_str = _extract_balanced_json(text)
    if not json_str:
        jm = _JSON_RE_DOTALL.search(text)
        json_str = jm.group() if jm else ""
    return rationale, json_str


def parse_single_general_score(text: str, metric: str) -> tuple[int | None, str]:
    rationale, json_str = parse_cot_response(text)
    if not json_str:
        return None, rationale
    data = _try_parse_json(json_str)
    if data is None:
        return None, rationale
    if not rationale and isinstance(data.get("reasoning"), str):
        rationale = data["reasoning"]
    v = data.get(metric)
    if v is not None and isinstance(v, (int, float)):
        return v, rationale
    numeric_vals = [val for val in data.values() if isinstance(val, (int, float))]
    if len(numeric_vals) == 1:
        return numeric_vals[0], rationale
    return None, rationale


# ──────────────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────────────


def save_eval_json(path: str, results: list[dict], *, meta: dict | None = None, **extra_fields) -> None:
    out: dict = {}
    if meta is not None:
        out["meta"] = meta
    out.update({
        "num_videos": len(results),
        "general_dimensions": GENERAL_KEYS,
        "results": results,
        **extra_fields,
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d results to %s", len(results), path)


# ──────────────────────────────────────────────────────────────────────────────
# Backend ABC
# ──────────────────────────────────────────────────────────────────────────────


class VLMBackend(abc.ABC):

    @abc.abstractmethod
    def call(self, video_path: str, prompt_text: str, system_prompt: str | None = None) -> str:
        ...

    def release_video(self, video_path: str) -> None:
        pass


class VLMResponse(str):
    """String response with optional provider usage metadata."""

    usage: dict[str, Any] | None

    def __new__(cls, content: str, usage: dict[str, Any] | None = None):
        obj = str.__new__(cls, content)
        obj.usage = usage
        return obj


def _usage_to_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    if hasattr(usage, "dict"):
        return usage.dict()
    if isinstance(usage, dict):
        return usage
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Gemini backend
# ──────────────────────────────────────────────────────────────────────────────


class GeminiBackend(VLMBackend):

    def __init__(
        self,
        model: str = "gemini-3.1-pro-preview",
        api_key_file: str | None = None,
        vertexai: bool = False,
        project: str = "",
        location: str = "global",
        disable_thinking: bool = False,
        timeout: int = 120,
    ):
        from google import genai
        from google.genai import types

        self._types = types
        self.model = model
        self.disable_thinking = disable_thinking
        http_opts = types.HttpOptions(timeout=timeout * 1000)

        if vertexai:
            self.client = genai.Client(
                vertexai=True, project=project, location=location,
                http_options=http_opts,
            )
            self._use_files_api = False
            logger.info("Gemini Vertex AI (project=%s, location=%s, timeout=%ds)", project, location, timeout)
        else:
            api_key = ""
            if api_key_file and Path(api_key_file).exists():
                api_key = Path(api_key_file).read_text().strip()
            api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                raise RuntimeError(
                    "No Gemini API key. Set GEMINI_API_KEY env var or pass --api_key_file."
                )
            self.client = genai.Client(api_key=api_key, http_options=http_opts)
            self._use_files_api = True

        self._upload_cache: dict[str, Any] = {}
        self._upload_lock = threading.Lock()

    def _get_uploaded(self, video_path: str) -> Any:
        with self._upload_lock:
            if video_path in self._upload_cache:
                return self._upload_cache[video_path]
            uploaded = self.client.files.upload(file=video_path)
            while uploaded.state.name == "PROCESSING":
                time.sleep(2)
                uploaded = self.client.files.get(name=uploaded.name)
            if uploaded.state.name != "ACTIVE":
                raise RuntimeError(f"Upload failed: state={uploaded.state}")
            self._upload_cache[video_path] = uploaded
            return uploaded

    def call(self, video_path: str, prompt_text: str, system_prompt: str | None = None) -> str:
        if self._use_files_api:
            video_content: Any = self._get_uploaded(video_path)
        else:
            video_bytes = Path(video_path).read_bytes()
            video_content = self._types.Part.from_bytes(
                data=video_bytes, mime_type="video/mp4"
            )

        contents = [video_content, prompt_text]
        generate_kwargs: dict[str, Any] = {"model": self.model, "contents": contents}
        config_kwargs: dict[str, Any] = {"temperature": 0, "top_p": 1}
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        if self.disable_thinking:
            config_kwargs["thinking_config"] = self._types.ThinkingConfig(
                thinking_budget=0
            )
        generate_kwargs["config"] = self._types.GenerateContentConfig(**config_kwargs)

        resp = self.client.models.generate_content(**generate_kwargs)
        return resp.text

    def release_video(self, video_path: str) -> None:
        uploaded = self._upload_cache.pop(video_path, None)
        if uploaded is not None:
            try:
                self.client.files.delete(name=uploaded.name)
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# Qwen-VL (OpenAI-compatible vLLM) backend
# ──────────────────────────────────────────────────────────────────────────────


def _downscale_video_to_base64(video_path: str, max_height: int = 360, fps: int | None = None) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        vf_filters = f"scale=-2:{max_height}"
        if fps is not None:
            vf_filters = f"fps={fps}," + vf_filters
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", vf_filters,
                "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "28",
                tmp_path,
            ],
            capture_output=True, check=True,
        )
        with open(tmp_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
    finally:
        os.unlink(tmp_path)
    return f"data:video/mp4;base64,{b64}"


class OpenAIBackend(VLMBackend):
    """OpenAI-compatible client for a vLLM server hosting phyjudge."""

    def __init__(
        self,
        model: str,
        api_base: str = "http://localhost:29673/v1",
        api_key: str = "dummy",
        max_height: int = 360,
        fps: int | None = None,
        disable_thinking: bool = False,
        max_tokens: int = 1024,
    ):
        from openai import OpenAI

        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.max_height = max_height
        self.fps = fps
        self.disable_thinking = disable_thinking
        self.max_tokens = max_tokens
        self._b64_cache: dict[str, str] = {}

    def _get_video_b64(self, video_path: str) -> str:
        if video_path not in self._b64_cache:
            self._b64_cache[video_path] = _downscale_video_to_base64(
                video_path, self.max_height, fps=self.fps,
            )
        return self._b64_cache[video_path]

    def call(self, video_path: str, prompt_text: str, system_prompt: str | None = None) -> str:
        video_url = self._get_video_b64(video_path)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        )
        extra_body = {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
        }
        if self.disable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=1.0,
            top_p=0.95,
            presence_penalty=1.5,
            extra_body=extra_body,
        )
        msg = response.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning", None) or ""
        usage = _usage_to_dict(getattr(response, "usage", None))
        if reasoning:
            return VLMResponse(f"<think>{reasoning}</think>\n{content}", usage=usage)
        return VLMResponse(content, usage=usage)

    def release_video(self, video_path: str) -> None:
        self._b64_cache.pop(video_path, None)


# ──────────────────────────────────────────────────────────────────────────────
# Frame extraction (shared by GPT and Claude backends)
# ──────────────────────────────────────────────────────────────────────────────


def _extract_video_frames_b64(
    video_path: str,
    *,
    max_height: int,
    num_frames: int,
    sample_fps: float,
    max_images: int,
    jpeg_quality: int = 85,
    max_frame_bytes: int | None = None,
) -> list[tuple[float, str]]:
    """Sample frames from `video_path` and return list of (timestamp, base64-jpeg).

    If num_frames > 0, sample exactly that many evenly. Otherwise sample at
    `sample_fps`, then cap at `max_images`.
    """
    import io

    import numpy as np
    from decord import VideoReader, cpu
    from PIL import Image

    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        raise ValueError(f"No frames in video: {video_path}")

    source_fps = float(vr.get_avg_fps() or 0.0)
    if num_frames > 0:
        n = min(num_frames, total, max_images)
        indices = np.linspace(0, total - 1, n, dtype=int)
    else:
        step = max(1, int(round(source_fps / sample_fps))) if source_fps > 0 else 1
        indices = np.arange(0, total, step, dtype=int)
        if len(indices) > max_images:
            keep = np.linspace(0, len(indices) - 1, max_images, dtype=int)
            indices = indices[keep]

    frames = vr.get_batch(indices).asnumpy()
    sampled: list[tuple[float, str]] = []
    for idx, frame in zip(indices, frames):
        timestamp = float(idx) / source_fps if source_fps > 0 else 0.0
        img = Image.fromarray(frame)
        if img.height > max_height:
            ratio = max_height / img.height
            img = img.resize((int(img.width * ratio), max_height), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        data = buf.getvalue()
        if max_frame_bytes is not None and len(data) > max_frame_bytes:
            raise ValueError(
                f"Encoded frame is too large ({len(data)} bytes, limit {max_frame_bytes})"
            )
        sampled.append((timestamp, base64.b64encode(data).decode("utf-8")))
    return sampled


# ──────────────────────────────────────────────────────────────────────────────
# GPT (OpenAI direct API) backend
# ──────────────────────────────────────────────────────────────────────────────


class GPTBackend(VLMBackend):

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str = "",
        max_height: int = 360,
        num_frames: int = 0,
        sample_fps: int = 4,
        max_images: int = 32,
        reasoning_effort: str | None = None,
    ):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_height = max_height
        self.num_frames = num_frames
        self.sample_fps = sample_fps
        self.max_images = max_images
        self.reasoning_effort = reasoning_effort
        self._frames_cache: dict[str, list[tuple[float, str]]] = {}

    def _get_frames(self, video_path: str) -> list[tuple[float, str]]:
        if video_path not in self._frames_cache:
            self._frames_cache[video_path] = _extract_video_frames_b64(
                video_path,
                max_height=self.max_height,
                num_frames=self.num_frames,
                sample_fps=self.sample_fps,
                max_images=self.max_images,
            )
        return self._frames_cache[video_path]

    def call(self, video_path: str, prompt_text: str, system_prompt: str | None = None) -> str:
        frames = self._get_frames(video_path)

        image_contents = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
            for _, b64 in frames
        ]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    *image_contents,
                ],
            }
        )

        extra_kwargs: dict[str, Any] = {}
        if self.reasoning_effort:
            extra_kwargs["reasoning_effort"] = self.reasoning_effort
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=8192,
            temperature=0,
            **extra_kwargs,
        )
        return response.choices[0].message.content or ""

    def release_video(self, video_path: str) -> None:
        self._frames_cache.pop(video_path, None)


# ──────────────────────────────────────────────────────────────────────────────
# Claude backend (Vertex AI rawPredict)
# ──────────────────────────────────────────────────────────────────────────────


def _build_claude_content(
    frames_b64: list[tuple[float, str]], prompt_text: str,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Video frames sampled in chronological order. "
                f"Frame count: {len(frames_b64)}."
            ),
        }
    ]
    for i, (timestamp, b64) in enumerate(frames_b64, 1):
        content.append({"type": "text", "text": f"Frame {i} at {timestamp:.2f}s"})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            }
        )
    content.append({"type": "text", "text": prompt_text})
    return content


class ClaudeVertexBackend(VLMBackend):
    """Anthropic Claude on Vertex AI, using sampled video frames as images."""

    _SCOPE = "https://www.googleapis.com/auth/cloud-platform"
    _MAX_FRAME_BYTES = 5 * 1024 * 1024

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        project: str = "",
        location: str = "global",
        max_height: int = 360,
        num_frames: int = 0,
        sample_fps: float = 4.0,
        max_images: int = 32,
        jpeg_quality: int = 85,
        max_tokens: int = 4096,
        timeout: int = 120,
    ):
        import google.auth
        import requests
        from google.auth.transport.requests import Request

        self._requests = requests
        self._auth_request = Request()
        self._credentials, detected_project = google.auth.default(scopes=[self._SCOPE])

        self.model = model
        self.project = project or detected_project
        if not self.project:
            raise RuntimeError("No Vertex AI project configured for Claude backend")

        self.location = location
        self.max_height = max_height
        self.num_frames = num_frames
        self.sample_fps = sample_fps
        self.max_images = max_images
        self.jpeg_quality = jpeg_quality
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.endpoint = self._endpoint_for_location(self.project, self.location, self.model)

        self._token_lock = threading.Lock()
        self._frames_lock = threading.Lock()
        self._frames_cache: dict[str, list[tuple[float, str]]] = {}

        logger.info(
            "Claude Vertex AI (project=%s, location=%s, model=%s, fps=%s, num_frames=%s)",
            self.project, self.location, self.model, self.sample_fps, self.num_frames,
        )

    @staticmethod
    def _endpoint_for_location(project: str, location: str, model: str) -> str:
        if location == "global":
            host = "aiplatform.googleapis.com"
        elif location == "us":
            host = "aiplatform.us.rep.googleapis.com"
        elif location == "eu":
            host = "aiplatform.eu.rep.googleapis.com"
        else:
            host = f"{location}-aiplatform.googleapis.com"
        return (
            f"https://{host}/v1/projects/{project}/locations/{location}"
            f"/publishers/anthropic/models/{model}:rawPredict"
        )

    def _get_access_token(self) -> str:
        with self._token_lock:
            if not self._credentials.valid:
                self._credentials.refresh(self._auth_request)
            token = self._credentials.token
        if not token:
            raise RuntimeError("google.auth returned no access token")
        return token

    def _get_frames(self, video_path: str) -> list[tuple[float, str]]:
        with self._frames_lock:
            if video_path not in self._frames_cache:
                self._frames_cache[video_path] = _extract_video_frames_b64(
                    video_path,
                    max_height=self.max_height,
                    num_frames=self.num_frames,
                    sample_fps=self.sample_fps,
                    max_images=self.max_images,
                    jpeg_quality=self.jpeg_quality,
                    max_frame_bytes=self._MAX_FRAME_BYTES,
                )
            return self._frames_cache[video_path]

    @staticmethod
    def _extract_text(response_json: dict[str, Any]) -> str:
        parts = []
        for block in response_json.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part)

    def call(self, video_path: str, prompt_text: str, system_prompt: str | None = None) -> str:
        frames_b64 = self._get_frames(video_path)
        body: dict[str, Any] = {
            "anthropic_version": "vertex-2023-10-16",
            "max_tokens": self.max_tokens,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": _build_claude_content(frames_b64, prompt_text),
                }
            ],
        }
        if system_prompt:
            body["system"] = system_prompt

        response = self._requests.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self._get_access_token()}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json=body,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Claude Vertex error {response.status_code}: {response.text[:1000]}"
            )
        return self._extract_text(response.json())

    def release_video(self, video_path: str) -> None:
        with self._frames_lock:
            self._frames_cache.pop(video_path, None)


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────


def print_summary(results: list[dict], judge_name: str) -> None:
    if not results:
        return

    print("\n" + "=" * 60)
    print(f"Summary ({len(results)} videos scored by {judge_name})")
    print("=" * 60)

    print("\n--- General Dimensions ---")
    for dim in GENERAL_KEYS:
        vals = [
            r[dim]
            for r in results
            if r.get(dim) is not None and isinstance(r[dim], (int, float))
        ]
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  {dim:12s}: {avg:.2f}  (n={len(vals)})")

    general_avgs = [
        r["general_avg"] for r in results
        if r.get("general_avg") is not None
    ]
    if general_avgs:
        print(f"  {'OVERALL':12s}: {sum(general_avgs) / len(general_avgs):.2f}  (n={len(general_avgs)})")

    new_results = [r for r in results if "physical_laws" in r]
    if new_results:
        print("\n--- Physical Scores (per-law) ---")

        from collections import defaultdict
        law_scores: dict[str, list[float]] = defaultdict(list)

        for r in new_results:
            phys = r.get("physical")
            if not phys or not isinstance(phys, dict):
                continue
            laws = phys.get("laws", {})
            for law_name, law_data in laws.items():
                if law_data.get("status") == SCORED:
                    law_scores[law_name].append(law_data["score"])

        print("\n  Per-law averages:")
        law_means: dict[str, float] = {}
        for law in ALL_LAWS:
            scores = law_scores.get(law, [])
            if scores:
                mean = sum(scores) / len(scores)
                law_means[law] = mean
                print(f"    {law:20s}: {mean:.2f}  (n={len(scores)})")
            else:
                print(f"    {law:20s}: —  (no scored videos)")

        all_scored = [s for scores in law_scores.values() for s in scores]
        if all_scored:
            micro = sum(all_scored) / len(all_scored)
            print(f"    {'MICRO AVG':20s}: {micro:.2f}  (n={len(all_scored)} scores across {len(law_means)} laws)")


# ──────────────────────────────────────────────────────────────────────────────
# Backend factory
# ──────────────────────────────────────────────────────────────────────────────

_BACKEND_CHOICES = ["qwen9b", "qwen27b", "gemini", "gpt", "claude"]


def _read_key_file(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def create_backend(args: argparse.Namespace) -> VLMBackend:
    if args.backend == "gemini":
        return GeminiBackend(
            model=args.model or "gemini-3.1-pro-preview",
            api_key_file=getattr(args, "api_key_file", None),
            vertexai=getattr(args, "vertexai", False),
            project=getattr(args, "project", ""),
            location=getattr(args, "location", "global"),
            disable_thinking=getattr(args, "no_thinking", False),
            timeout=getattr(args, "request_timeout", 120),
        )
    if args.backend == "gpt":
        api_key = (
            _read_key_file(getattr(args, "api_key_file", None))
            or os.environ.get("OPENAI_API_KEY", "")
        )
        if not api_key:
            raise RuntimeError(
                "No OpenAI API key. Set OPENAI_API_KEY env var or pass --api_key_file."
            )
        return GPTBackend(
            model=args.model or "gpt-5.4",
            api_key=api_key,
            num_frames=getattr(args, "num_frames", 0),
            sample_fps=getattr(args, "fps", None) or 4,
            max_images=getattr(args, "max_images", 32),
            reasoning_effort=getattr(args, "reasoning_effort", None),
        )
    if args.backend == "claude":
        return ClaudeVertexBackend(
            model=args.model or "claude-opus-4-7",
            project=getattr(args, "project", ""),
            location=getattr(args, "location", "global"),
            num_frames=getattr(args, "num_frames", 0),
            sample_fps=getattr(args, "fps", None) or 4.0,
            max_images=getattr(args, "max_images", 32),
            max_tokens=getattr(args, "max_tokens", None) or 4096,
            timeout=getattr(args, "request_timeout", 120),
        )
    # qwen9b / qwen27b — local vLLM
    default_model = {
        "qwen9b": "phyjudge",
        "qwen27b": "phyjudge",
    }.get(args.backend, "phyjudge")
    return OpenAIBackend(
        model=args.model or default_model,
        api_base=args.api_base,
        max_height=getattr(args, "max_height", 360),
        fps=getattr(args, "fps", None),
        disable_thinking=getattr(args, "no_thinking", False),
        max_tokens=getattr(args, "max_tokens", None) or 1024,
    )
