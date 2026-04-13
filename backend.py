"""
NeuroAd Backend — FastAPI
=========================
Analyzes ad images through a real image-driven brain encoding pipeline:

  1. ResNet-50 (ImageNet pretrained) extracts multi-layer visual features
  2. Content-signal detectors compute face, scene, text, color, contrast,
     reward, and spatial signals from those features
  3. Per-region linear probes (derived from BOLD5000 + img2fmri methodology)
     map those signals to predicted fMRI β-weights across 20 HCP MMP1.0 parcels
  4. Perplexity Agent API (falls back to Anthropic Claude) interprets the
     activation pattern against what the ad already contains, returning
     non-redundant gap analysis and creative recommendations

Run with:  uvicorn backend:app --reload
"""

import os
import io
import json
import logging
import traceback
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("neuroadapp")

app = FastAPI(title="NeuroAd API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Real brain encoder (ResNet-50 + BOLD5000 linear probes)
# ─────────────────────────────────────────────────────────────────────────────

def run_encoder(image_bytes: bytes):
    """
    Returns (region_scores, mode, signals) via brain_encoder.encode_image.
    signals is a dict of raw content signals: face, scene, text, color, etc.
    """
    from brain_encoder import encode_image
    return encode_image(image_bytes)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Build prompt grounded in what the ad already contains
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a neuroscience consultant specialising in neuromarketing.
You interpret predicted fMRI brain activation patterns from ad-exposure studies.
You are given BOTH the brain activation data AND a description of what visual content
is already present in the ad — use this to avoid recommending things the ad already does.
Focus gaps and recommendations on what is MISSING or UNDERSERVED relative to the desired goal.
Always return ONLY valid JSON — no markdown fences, no commentary outside the JSON object."""


def _build_user_prompt(
    region_scores: dict,
    demographic: str,
    desired_response: str,
    image_description: str,
    archetype: str,
) -> str:
    top20 = list(region_scores.items())[:20]
    region_list = "\n".join(
        f"  {i+1:2d}. {name:<12} {score:.3f}"
        for i, (name, score) in enumerate(top20)
    )

    # Identify the top activated region group for context
    top_region = top20[0][0] if top20 else "unknown"
    top_score  = top20[0][1] if top20 else 0

    # Identify under-activated regions relative to desired response
    # (regions below the median that matter for the goal)
    scores_dict = dict(top20)
    median = sorted(scores_dict.values())[len(scores_dict)//2]
    low_regions = [(r, s) for r, s in top20 if s < median][:5]
    low_list = ", ".join(f"{r} ({s:.2f})" for r, s in low_regions)

    return f"""A neuromarketing brain-encoding model (ResNet-50 + BOLD5000 linear probes,
based on img2fmri methodology, Feilong et al. 2023) processed an ad image and returned
predicted cortical activation scores across 20 HCP MMP1.0 brain regions.

━━━ WHAT THE AD ALREADY CONTAINS (encoder-detected) ━━━
Ad archetype: {archetype}
{image_description}

━━━ BRAIN ACTIVATION RESULTS (scale 0–1) ━━━
{region_list}

Dominant activation: {top_region} at {top_score:.3f}
Relatively under-activated regions in this ad: {low_list}

━━━ CAMPAIGN CONTEXT ━━━
Target demographic: {demographic}
Desired psychological response: {desired_response}

━━━ YOUR TASK ━━━
Return a JSON object with exactly these three fields.

CRITICAL RULES:
- The "gaps" must identify brain systems that are NOT being sufficiently engaged
  given the desired response — do NOT flag things the ad already has (listed above).
- The "recommendations" must suggest SPECIFIC creative additions or changes that
  the ad currently lacks. Do NOT recommend adding faces if faces are already detected,
  do NOT recommend more color if the ad is already color-rich, etc.
- Be specific: name which brain region the recommendation targets, what it controls,
  and what exact creative element would activate it.

{{
  "summary": "<2 sentences: what does the dominant activation pattern reveal about how this ad is being processed neurologically? Reference the top 2-3 regions by name.>",
  "gaps": [
    "<2-3 specific brain systems that are underactivated relative to the desired response. Name the region, its function, and why it matters for this goal. Do NOT include things already in the ad.>",
    "..."
  ],
  "recommendations": [
    "<3-5 concrete creative changes that are NOT already present in the ad. Each rec must: name the target brain region, explain what it controls, and specify an exact creative element to add/change.>",
    "..."
  ]
}}

Return ONLY the JSON object. No preamble, no markdown."""


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — LLM interpretation
# ─────────────────────────────────────────────────────────────────────────────

def interpret_activations(
    region_scores: dict,
    demographic: str,
    desired_response: str,
    signals: dict,
) -> dict:
    """
    Calls Perplexity Agent API first, falls back to Anthropic Claude.
    Returns parsed JSON dict with keys: summary, gaps, recommendations.
    """
    from brain_encoder import describe_signals, classify_archetype
    image_description = describe_signals(signals)
    archetype = classify_archetype(signals)

    prompt = _build_user_prompt(
        region_scores, demographic, desired_response,
        image_description, archetype
    )

    # ── Try Perplexity Agent API ──────────────────────────────────────────────
    pplx_key = os.environ.get("PERPLEXITY_API_KEY")
    if pplx_key:
        try:
            log.info("Using Perplexity Agent API (sonar-pro)")
            from perplexity import Perplexity
            client = Perplexity(api_key=pplx_key)
            response = client.responses.create(
                model="perplexity/sonar-pro",
                input=prompt,
                instructions=SYSTEM_PROMPT,
                text={"format": {"type": "text"}},
                temperature=0.3,
                max_output_tokens=1400,
            )
            result = _parse_json(response.output_text)
            result["_llm"] = "perplexity/sonar-pro"
            return result
        except Exception as e:
            log.warning(f"Perplexity failed ({e}), falling back to Anthropic")

    # ── Fallback: Anthropic Claude ────────────────────────────────────────────
    log.info("Using Anthropic Claude as interpretation engine")
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        result = _parse_json(msg.content[0].text)
        result["_llm"] = "anthropic/claude-sonnet-4-5"
        return result
    except Exception as e:
        log.error(f"Anthropic fallback also failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Both Perplexity and Anthropic interpretation failed: {e}",
        )


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]).strip()
    parsed = json.loads(text)
    for key in ("summary", "gaps", "recommendations"):
        if key not in parsed:
            raise ValueError(f"Missing key '{key}' in model response")
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — FastAPI endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    demographic: str = Form(...),
    desired_response: str = Form(...),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="Uploaded file must be an image.")

    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=422, detail="Empty image file.")

    log.info(
        f"Analyzing: {image.filename!r} ({len(image_bytes)//1024}KB) "
        f"| demo={demographic!r} | goal={desired_response!r}"
    )

    # 1. Brain encoder
    try:
        all_scores, encoder_mode, signals = run_encoder(image_bytes)
    except Exception as e:
        log.error(f"Brain encoder failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Brain encoder error: {e}")

    top20 = dict(list(all_scores.items())[:20])
    log.info(f"Top region: {list(top20.items())[0]} | signals: face={signals['face']:.2f} scene={signals['scene']:.2f} text={signals['text']:.2f}")

    # 2. LLM interpretation (grounded in detected image content)
    interpretation = interpret_activations(top20, demographic, desired_response, signals)

    from brain_encoder import classify_archetype
    return {
        "region_scores":   top20,
        "interpretation":  interpretation,
        "encoder_mode":    encoder_mode,
        "encoder_method":  "ResNet-50 + BOLD5000 linear probes (img2fmri methodology)",
        "encoder_signals": signals,                      # raw content signals for UI
        "ad_archetype":    classify_archetype(signals),  # e.g. "Face / People"
    }


@app.get("/health")
async def health():
    """Health check — also confirms the brain encoder is loadable."""
    try:
        from brain_encoder import encode_image, REGION_PROBES
        encoder_ok = len(REGION_PROBES) == 20
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": f"Encoder import failed: {e}"}
        )
    return {
        "status": "ok",
        "encoder": "ResNet-50 + BOLD5000 linear probes",
        "regions": 20,
        "encoder_ready": encoder_ok,
    }


@app.get("/")
async def serve_index():
    return FileResponse(Path(__file__).parent / "index.html")
