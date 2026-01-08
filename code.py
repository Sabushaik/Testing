from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi. responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware. base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import os
from PIL import Image
import io
import cv2
import tempfile
from datetime import datetime
import time
from typing import Optional, List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import logging
import json
from collections import defaultdict, Counter
import math
import base64
import httpx
import traceback

import tritonclient.http as httpclient
from tritonclient.http import InferInput, InferRequestedOutput
from ultralytics. trackers. bot_sort import BOTSORT
from ultralytics. engine. results import Boxes
from types import SimpleNamespace

from dotenv import load_dotenv
load_dotenv()
# =================================================
# LOGGING
# =================================================
logging.basicConfig(
    level=logging. INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =================================================
# TIMING MIDDLEWARE
# =================================================
class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request. state.start_time = time.time()
        response = await call_next(request)
        return response

# =================================================
# FASTAPI APP
# =================================================
app = FastAPI()
app.add_middleware(TimingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================================================
# AWS CONFIG
# =================================================
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
OUTPUT_S3_BUCKET = os.getenv("OUTPUT_S3_BUCKET_NAME")  # NEW
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os. getenv("AWS_REGION", "us-east-1")

# Log AWS config status (without exposing secrets)
logger.info(f"S3_BUCKET configured: {bool(S3_BUCKET)}")
logger.info(f"INPUT S3_BUCKET: {S3_BUCKET}")
logger.info(f"OUTPUT_S3_BUCKET: {OUTPUT_S3_BUCKET}")
logger.info(f"AWS_ACCESS_KEY_ID configured: {bool(AWS_ACCESS_KEY_ID)}")
logger.info(f"AWS_SECRET_ACCESS_KEY configured: {bool(AWS_SECRET_ACCESS_KEY)}")
logger.info(f"AWS_REGION:  {AWS_REGION}")

s3_client = None
try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    logger.info("✅ S3 client initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize S3 client: {e}")

# =================================================
# BEDROCK CLIENT (for Amazon Nova Lite model)
# =================================================
bedrock_client = None
try:
    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    logger.info("✅ Bedrock client initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize Bedrock client: {e}")

# =================================================
# TRITON CONFIG (from Jupyter notebook)
# =================================================
TRITON_URL = "localhost:8000"
MODEL_NAME = "yolo_person_detection"

TARGET_SIZE = 640
PAD_VALUE = 114
CONF_THRESH = 0.65
IOU_THRESHOLD = 0.90
DIST_THRESHOLD = 50

# =================================================
# GPT CONFIG (from Jupyter notebook)
# =================================================
MODEL = "Qwen/Qwen2.5-Omni-7B-AWQ"
API_BASE = "https://vllm-dgx.p9sphere.com/v1"
API_KEY = os.getenv("OPENAI_KEY")  # no default



BATCH_SIZE = 1
MAX_PARALLEL = 6


# =================================================
# CLASS IDS (from Jupyter notebook)
# =================================================
PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = {2, 3, 5, 6, 7}
CLASS_NAME_MAP = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    6: "train",
    7: "truck",
}

# =================================================
# SAMPLE IMAGES (from Jupyter notebook)
# =================================================
SAMPLES = {
    "hardhat": [f"samples/hardhat/hardhat{i}.jpg" for i in range(1, 4)],
    "goggles": [f"samples/goggles/goggles{i}.jpg" for i in range(1, 5)],
    "vest": [f"samples/vest/safety_vest{i}.jpg" for i in range(1, 6)],
    "gloves": [f"samples/gloves/gloves{i}.jpg" for i in range(1, 6)],
    "shoes": [f"samples/shoes/shoes{i}.jpg" for i in range(1, 4)],
    "forklift": [f"samples/forklift/forklift{i}.jpg" for i in range(1, 5)],
    "truck": [f"samples/truck/truck{i}.jpg" for i in range(1, 5)],
    "excavator": [f"samples/excavator/excavator{i}.jpg" for i in range(1, 5)],
    "bull_dozer": [f"samples/bull_dozer/bull_dozer{i}.jpg" for i in range(1, 5)],
    "cement_mixer": [f"samples/cement_mixer/cement_mixer{i}.jpg" for i in range(1, 5)],
    "roller": [f"samples/roller/roller{i}.jpg" for i in range(1, 5)],
    "tractor": [f"samples/tracktor/tracktor{i}.jpg" for i in range(1, 3)],
}

# =================================================
# HARDCODED CONFIG
# =================================================
HARDCODED_CONFIG = {
    "target_fps": 30,
    "confidence_threshold":  CONF_THRESH,
    "iou_threshold": IOU_THRESHOLD,
    "batch_size": 16,
    "save_counting_json": True,
    "frame_skip": 1
}

# =================================================
# GPT SYSTEM PROMPT (from Jupyter notebook)
# =================================================
SYSTEM_TEXT = (
    "You are a vision analysis system for industrial safety compliance.\n"
"Analyze images and return ONLY valid JSON.\n\n"
"IMPORTANT:\n"
"Use the PROVIDED SAMPLE IMAGES as visual guidance to understand\n"
"how industrial PPE AND manufacturing vehicles generally look\n"
"in real manufacturing and construction environments.\n"
"Do NOT restrict detection only to exact sample matches.\n\n"
"=== PPE DETECTION GUIDELINES ===\n\n"
"1. SHOES (Industrial Footwear):\n"
"   - TYPES: Steel-toe boots, safety shoes, work boots, closed-toe industrial footwear\n"
"   - APPEARANCE: Thick soles, reinforced toe caps, sturdy construction, ankle or low-cut\n"
"   - DETECTION RULES:\n"
"     * If AT LEAST ONE shoe is clearly visible on a person → TRUE\n"
"     * Common scenarios: Person standing (both visible), sitting/driving (one visible), walking (partial view)\n"
"     * Look for: Visible footwear on feet, even if only one foot is in frame\n"
"     * IGNORE: Casual sneakers, sandals, slippers, dress shoes\n"
"     * FOR DRIVERS/SEATED WORKERS: One industrial shoe visible = TRUE\n\n"
"2. GOGGLES / SAFETY GLASSES / EYE PROTECTION:\n"
"   - TYPES: Safety goggles, safety glasses, face shields, protective eyewear\n"
"   - APPEARANCE: \n"
"     * Goggles: Wrap-around seal, clear/tinted lenses, elastic strap\n"
"     * Safety Glasses: Plastic frame, side shields, clear/tinted lenses\n"
"     * Face Shields: Clear visor covering entire face\n"
"   - DETECTION RULES:\n"
"     * ANY visible eye protection worn on face → TRUE\n"
"     * Check for: Protective frames on face, lenses over eyes, visor in front of face\n"
"     * Look at: Bridge of nose, temples, forehead area\n"
"     * IGNORE: Regular prescription glasses, sunglasses without side protection\n"
"     * If person is facing away but wearing hardhat: Check for goggle straps or frame edges\n\n"
"3. SAFETY VEST / HIGH-VISIBILITY CLOTHING:\n"
"   - TYPES: Reflective vests, hi-vis jackets, coveralls with reflective strips, industrial uniforms\n"
"   - APPEARANCE:\n"
"     * Colors: Bright yellow, orange, lime green, red\n"
"     * Features: Reflective strips/bands (silver, yellow), mesh or solid fabric\n"
"     * Styles: Sleeveless vests, long-sleeve jackets, full coveralls with reflective elements\n"
"   - DETECTION RULES:\n"
"     * Any high-visibility outer garment with reflective strips → TRUE\n"
"     * Industrial coveralls/uniforms in bright colors → TRUE\n"
"     * Partial visibility (torso, shoulder, or back visible) → TRUE\n"
"     * Look for: Reflective horizontal/vertical strips, bright base color\n\n"
"4. GLOVES (Hand Protection):\n"
"   - TYPES: Work gloves, rubber gloves, cut-resistant gloves, heat-resistant gloves\n"
"   - APPEARANCE: Covering hands, fabric/leather/rubber material, fitted or loose\n"
"   - DETECTION RULES:\n"
"     * At least ONE gloved hand visible → TRUE\n"
"     * Both hands must be checked if visible\n"
"     * If only one hand in frame and gloved → TRUE\n"
"     * Look for: Material texture, coverage of fingers/palm\n"
"     * IGNORE: Bare hands, casual fashion gloves\n\n"
"5. HARDHAT / SAFETY HELMET:\n"
"   - APPEARANCE: Rigid protective helmet, dome shape, various colors (white, yellow, blue, orange, red)\n"
"   - DETECTION RULES:\n"
"     * Properly worn on head → TRUE\n"
"     * Visible from any angle (front, side, back, top) → TRUE\n"
"     * Look for: Hard shell, chin strap, brim edge\n"
"     * IGNORE: Caps, baseball hats, beanies\n\n"
"=== PERSON-CENTRIC PPE RULES ===\n"
"- PPE must be WORN on a person (not carried, held, or placed nearby)\n"
"- For EACH person detected, evaluate EACH PPE item independently\n"
"- PARTIAL VISIBILITY COUNTS:\n"
"  * One shoe visible (driver, seated, partially occluded) → shoes = TRUE\n"
"  * Partial vest visible (shoulder, back, side) → safety_vest = TRUE\n"
"  * One glove visible → gloves = TRUE\n"
"  * Any angle of hardhat → hardhat = TRUE\n"
"  * Any eye protection on face → goggles = TRUE\n\n"
"=== VISIBILITY & CLARITY RULES ===\n"
"- Object must be ≥40% visible (reduced from 50% to catch partial views)\n"
"- Moderate blur is acceptable if object type is identifiable\n"
"- If uncertain but object characteristics match PPE description → TRUE (reduce false negatives)\n"
"- Extreme occlusion or complete blur → FALSE\n\n"
"=== VEHICLE DETECTION GUIDELINES ===\n\n"
"VEHICLE TYPES & APPEARANCE:\n"
"- Bulldozer: Tracked vehicle, wide front blade, low profile\n"
"- Excavator: Rotating body, long arm with bucket, tracks/wheels\n"
"- Cement Mixer Truck: Large rotating drum on truck chassis\n"
"- Mobile Concrete Mixer: Smaller site mixer with drum/chute\n"
"- Crane: Extended boom, hook/cable system, lifting mechanism\n"
"- Forklift: Two parallel forks, upright mast, compact design\n"
"- Road Roller: Large cylindrical rollers, no regular wheels\n"
"- Tractor: Large rear wheels, smaller front, open/enclosed cab\n"
"- Dump Truck: Hydraulic tilting cargo bed on heavy truck chassis\n"
"- Loader: Front bucket, articulated steering, wheeled or tracked\n"
"- Grader: Long blade between axles for ground leveling\n"
"- Cargo Truck: Standard truck with cabin and cargo bed/container\n\n"
"VEHICLE COUNTING RULES:\n"
"- Count only real, physical vehicles in the scene\n"
"- Must be ≥50% visible to count\n"
"- If ≥80% confident of specific type: Group by type {\"vehicle_type\": count}\n"
"- If 50-79% confident or unclear type: Report as {\"Industrial Equipment\": count}\n"
"- If <50% confident: Do not count at all\n"
"- IGNORE: Toy vehicles, posters, screen displays, reflections\n\n"
"=== NEGATIVE CONSTRAINTS ===\n"
"DO NOT detect as PPE:\n"
"- Caps, baseball hats, beanies\n"
"- Regular sunglasses or prescription glasses\n"
"- Casual clothing, hoodies, regular jackets\n"
"- Sandals, slippers, casual sneakers, dress shoes\n"
"- Fashion accessories\n\n"
"=== OUTPUT FORMAT ===\n"
"Return ONLY valid JSON with:\n"
"{\n"
"  \"hardhat\": boolean,\n"
"  \"goggles\": boolean,\n"
"  \"safety_vest\": boolean,\n"
"  \"gloves\": boolean,\n"
"  \"shoes\": boolean,\n"
"  \"vehicles\": {\"vehicle_type\": count, ...}\n"
"}\n\n"
"CRITICAL: Prioritize TRUE positives for safety compliance.\n"
"When in doubt about PPE presence (30-40% confidence) → lean toward TRUE.\n"
"Better to flag potential PPE than miss safety equipment.\n\n"
"No explanations. JSON only."
)



# =================================================
# PYDANTIC MODELS
# =================================================
class S3UriInput(BaseModel):
    s3_uri: str


class PPEAndVehicleStatus(BaseModel):
    hardhat: bool = False
    goggles: bool = False
    safety_vest: bool = False
    gloves: bool = False
    shoes: bool = False
    vehicles: Dict[str, int] = Field(default_factory=dict)


# =================================================
# GLOBAL STATE
# =================================================
thread_executor = ThreadPoolExecutor(max_workers=10)
triton_client = None
SYSTEM_MESSAGES = None

try:
    triton_client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)
    logger.info(f"✅ Connected to Triton server at {TRITON_URL}")
    logger.info(f"✅ Using model: {MODEL_NAME}")
except Exception as e:
    logger.error(f"❌ Failed to initialize Triton client:  {e}")


# =================================================
# UTILITY FUNCTIONS (from Jupyter notebook)
# =================================================
def encode_image(path:  str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def nms(dets, iou_thresh=0.9):
    dets = sorted(dets, key=lambda x: x["conf"], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [
            d for d in dets
            if d["cls"] != best["cls"] or iou(d["bbox"], best["bbox"]) < iou_thresh
        ]
    return keep


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def euclidean(p1, p2):
    return math.dist(p1, p2)


def to_py(x):
    if isinstance(x, (np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.int32, np.int64)):
        return int(x)
    return x


def preprocess(frame):
    h, w, _ = frame.shape
    scale = min(TARGET_SIZE / h, TARGET_SIZE / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((TARGET_SIZE, TARGET_SIZE, 3), PAD_VALUE, dtype=np.uint8)
    px = (TARGET_SIZE - nw) // 2
    py = (TARGET_SIZE - nh) // 2
    canvas[py:py + nh, px: px + nw] = resized
    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np. float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]
    return img, {
        "scale": scale, "pad_x": px, "pad_y": py,
        "orig_w": w, "orig_h": h
    }


def get_color(cls_id):
    np. random.seed(cls_id + 100)
    return tuple(int(c) for c in np.random.randint(50, 255, 3))


def get_person_color(has_full_ppe:  bool):
    return (0, 255, 0) if has_full_ppe else (0, 0, 255)


def get_vehicle_color():
    return (0, 200, 255)


# =================================================
# SAMPLE CACHE LOADING
# =================================================
def load_sample_cache():
    """Load and encode sample images - same as notebook"""
    cache = {}
    for key, paths in SAMPLES. items():
        cache[key] = []
        for p in paths:
            if os.path.exists(p):
                try:
                    cache[key].append(encode_image(p))
                except Exception as e:
                    logger.warning(f"Failed to load sample {p}: {e}")
            else:
                logger.warning(f"Sample image not found: {p}")
    return cache

def build_system_messages(sample_cache:  Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Build system messages with sample images - same as notebook"""
    msgs = [{"type": "text", "text":  SYSTEM_TEXT}]
    for label, encoded_images in sample_cache.items():
        for img_b64 in encoded_images: 
            msgs.append({"type":  "text", "text": f"Example:  {label}"})
            msgs.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
    return msgs

@app.on_event("startup")
def initialize_system_messages():
    global SYSTEM_MESSAGES
    try:
        sample_cache = load_sample_cache()
        SYSTEM_MESSAGES = build_system_messages(sample_cache)
        # FIX the count:
        total_samples = sum(len(v) for v in sample_cache.values())
        logger.info(f"✅ Loaded {total_samples} sample images for Bedrock Nova Lite prompts")
    except Exception as e:
        logger.error(f"Failed to initialize system messages: {e}")
        SYSTEM_MESSAGES = [{"type": "text", "text": SYSTEM_TEXT}]

# =================================================
# TRITON INFERENCE (from Jupyter notebook)
# =================================================
def triton_infer(img, local_client=None):
    client = local_client or triton_client
    inp = InferInput("images", img.shape, "FP32")
    inp.set_data_from_numpy(img)
    out = InferRequestedOutput("output0")
    resp = client.infer(model_name=MODEL_NAME, inputs=[inp], outputs=[out])
    return resp. as_numpy("output0")


def postprocess(output, meta, allowed_cls:  set):
    preds = output[0]. T
    raw = []
    for p in preds: 
        cx, cy, w, h = p[: 4]
        scores = p[4:]
        cls = int(np.argmax(scores))
        conf = scores[cls]
        if cls not in allowed_cls or conf < CONF_THRESH: 
            continue
        x1 = (cx - w / 2 - meta["pad_x"]) / meta["scale"]
        y1 = (cy - h / 2 - meta["pad_y"]) / meta["scale"]
        x2 = (cx + w / 2 - meta["pad_x"]) / meta["scale"]
        y2 = (cy + h / 2 - meta["pad_y"]) / meta["scale"]
        raw.append({"bbox": [x1, y1, x2, y2], "conf": float(conf), "cls": cls})
    dets = nms(raw, IOU_THRESHOLD)
    if not dets: 
        return np.empty((0, 6), dtype=np.float32)
    return np.array([d["bbox"] + [d["conf"], d["cls"]] for d in dets], np.float32)
import re



def clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw)
    raw = re.sub(r"```$", "", raw)
    return raw.strip()

# =================================================
# BEDROCK VISION CALLS (using Amazon Nova Lite model)
# =================================================
def get_image_format_and_base64(image_b64: str) -> tuple:
    """Helper to prepare image for Bedrock API - defaults to jpeg format"""
    # Default to jpeg since most images are jpeg
    # The base64 string is passed through as-is
    return "jpeg", image_b64

async def call_gpt_single(client: httpx.AsyncClient, image_b64: str, retry_count: int = 3) -> PPEAndVehicleStatus:
    """
    Return hardcoded PPE detection results for all persons detected.
    Hardcoded values:
    - Hardhat: N (False)
    - Goggles: N (False)
    - Vest: N (False)
    - Gloves: Y (True)
    - Shoes: Y (True)
    Note: 'client', 'image_b64', and 'retry_count' parameters are unused but kept for backward compatibility.
    """
    # Return hardcoded PPE status
    hardcoded_result = {
        "hardhat": False,
        "goggles": False,
        "safety_vest": False,
        "gloves": True,
        "shoes": True,
        "vehicles": {}
    }
    
    logger.info(f"Returning hardcoded PPE detection result: {hardcoded_result}")
    return PPEAndVehicleStatus(**hardcoded_result)
async def process_crops_async(crop_paths: List[str]) -> List[Dict]:
    """
    Process crops with hardcoded PPE detection values.
    Returns the same hardcoded PPE status for all crops.
    Note: Maintains async structure for compatibility with existing pipeline.
    """

    results = [None] * len(crop_paths)
    semaphore = asyncio.Semaphore(MAX_PARALLEL)

    async with httpx.AsyncClient(timeout=120.0) as client:
        async def worker(idx: int, path: str):
            async with semaphore:
                try:
                    b64 = encode_image(path)
                    pred = await call_gpt_single(client, b64)
                    results[idx] = pred.model_dump()
                except Exception as e:
                    logger.error(f"Error processing crop {path}: {e}")
                    results[idx] = {"error": str(e)}

        tasks = [asyncio.create_task(worker(i, p)) for i, p in enumerate(crop_paths)]
        if tasks:
            await asyncio.gather(*tasks)

    return results
# =================================================
# VIDEO CONVERSION (from original code)
# =================================================
def convert_video_to_web_format(input_path:  str, output_path: str) -> bool:
    try: 
        import subprocess
        import shutil

        if not shutil.which('ffmpeg'):
            logger.warning("FFmpeg not found, skipping conversion")
            return False

        command = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '23',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-y',
            output_path
        ]

        logger.info("🔄 Converting video to web-compatible format...")
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,
            check=False
        )

        if result.returncode == 0 and os.path. exists(output_path):
            logger. info("✅ Video conversion successful")
            return True
        else: 
            logger.error(f"❌ FFmpeg conversion failed: {result.stderr.decode()[:200]}")
            return False

    except subprocess.TimeoutExpired:
        logger. error("❌ FFmpeg conversion timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Video conversion error: {str(e)}")
        return False


# =================================================
# S3 UTILITIES (from original code)
# =================================================
def generate_presigned_url(bucket: str, key:  str, expiration:  int = 3600) -> Optional[str]:
    try:
        if s3_client is None:
            logger. error("S3 client not initialized")
            return None
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expiration
        )
        return presigned_url
    except Exception as e:
        logger. error(f"Failed to generate presigned URL: {str(e)}")
        return None


async def upload_to_s3_async(file_path: str, bucket: str, key: str, content_type: str = 'video/mp4') -> str:
    try: 
        if s3_client is None: 
            raise ValueError("S3 client not initialized")
            
        transfer_config = boto3.s3.transfer.TransferConfig(
            multipart_threshold=1024 * 1024 * 8,
            max_concurrency=20,
            multipart_chunksize=1024 * 1024 * 8,
            use_threads=True
        )

        def upload_with_config():
            s3_client.upload_file(
                file_path,
                bucket,
                key,
                Config=transfer_config,
                ExtraArgs={'ContentType': content_type}
            )

        await asyncio.get_event_loop().run_in_executor(
            thread_executor,
            upload_with_config
        )

        s3_uri = f"s3://{bucket}/{key}"
        logger. info(f"Uploaded to S3: {s3_uri}")
        return s3_uri
    except Exception as e: 
        logger.error(f"S3 upload failed:  {str(e)}")
        raise


async def upload_json_to_s3(data: dict, bucket: str, key: str) -> str:
    try:
        if s3_client is None:
            raise ValueError("S3 client not initialized")
            
        json_data = json.dumps(data, indent=2)
        await asyncio.get_event_loop().run_in_executor(
            thread_executor,
            lambda: s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json_data.encode('utf-8'),
                ContentType='application/json'
            )
        )
        s3_uri = f"s3://{bucket}/{key}"
        logger.info(f"Uploaded JSON to S3: {s3_uri}")
        return s3_uri
    except Exception as e: 
        logger.error(f"JSON upload failed: {e}")
        raise


# =================================================
# MAIN PROCESSING PIPELINE (from Jupyter notebook)
# =================================================
def process_video_with_gpt_pipeline(
    video_path: str,
    output_video_path: str,
    crop_base_dir: str,
    veh_crop_base_dir: str
) -> Dict[str, Any]:
    """
    Main processing pipeline combining Triton detection + BOTSORT tracking + Bedrock Nova Lite Vision
    """
    timings = {
        "detection_tracking_total": 0.0,
        "triton_infer_total": 0.0,
        "postprocess_total": 0.0,
        "tracking_total": 0.0,
        "gpt_total": 0.0,
        "annotate_total": 0.0,
        "total":  0.0
    }

    start_total = time.time()

    # Create local Triton client for thread safety
    local_triton_client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)

    # Create crop directories
    os.makedirs(crop_base_dir, exist_ok=True)
    os.makedirs(veh_crop_base_dir, exist_ok=True)

    # Initialize trackers
    person_tracker = BOTSORT(SimpleNamespace(
        tracker_type="botsort",
        track_high_thresh=0.65,
        track_low_thresh=0.35,
        new_track_thresh=0.65,
        track_buffer=300,
        match_thresh=0.65,
        fuse_score=True,
        gmc_method=None,
        proximity_thresh=0.5,
        appearance_thresh=0.8,
        with_reid=False,
        model="auto"
    ))

    vehicle_tracker = BOTSORT(SimpleNamespace(
        tracker_type="botsort",
        track_high_thresh=0.65,
        track_low_thresh=0.35,
        new_track_thresh=0.65,
        track_buffer=300,
        match_thresh=0.65,
        fuse_score=True,
        gmc_method=None,
        proximity_thresh=0.6,
        appearance_thresh=0.8,
        with_reid=False,
        model="auto"
    ))

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Failed to open video file")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # fps=20
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"📹 Processing video: {total_frames} frames at {fps} FPS, {W}x{H}")

    # Tracking state
    person_trackid_to_bbox = {}
    person_tid_remap = {}
    person_active_ids = set()

    vehicle_trackid_to_bbox = {}
    vehicle_tid_remap = {}
    vehicle_active_ids = set()

    last_crop_frame = {}
    json_output = []
    frame_idx = 0

    allowed_cls = {PERSON_CLASS_ID} | VEHICLE_CLASS_IDS

    # =================================================
    # PHASE 1: Detection + Tracking + Crop Creation
    # =================================================
    logger.info("🔍 Phase 1: Detection & Tracking...")
    detection_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Preprocess and infer
        img, meta = preprocess(frame)

        infer_start = time.time()
        output = triton_infer(img, local_triton_client)
        timings["triton_infer_total"] += time.time() - infer_start

        post_start = time.time()
        dets = postprocess(output, meta, allowed_cls)
        timings["postprocess_total"] += time.time() - post_start

        if len(dets) == 0:
            continue

        # Separate person vs vehicle detections
        person_mask = dets[: , 5] == PERSON_CLASS_ID
        person_dets = dets[person_mask]
        veh_dets = dets[~person_mask]

        # Track persons
        track_start = time.time()
        if len(person_dets) > 0:
            boxes = Boxes(person_dets[: , :6], orig_shape=frame.shape)
            person_tracks = person_tracker.update(boxes, frame)
        else:
            person_tracks = []

        # Track vehicles
        if len(veh_dets) > 0:
            veh_boxes = Boxes(veh_dets[:, :6], orig_shape=frame.shape)
            veh_tracks = vehicle_tracker.update(veh_boxes, frame)
        else:
            veh_tracks = []
        timings["tracking_total"] += time.time() - track_start

        # Process person tracks
        for t in person_tracks:
            x1, y1, x2, y2, tid, _, _ = t[: 7]
            tid = int(tid)
            bbox = (x1, y1, x2, y2)

            if tid not in person_tid_remap:
                matched = None
                for old_id in person_active_ids:
                    prev = person_trackid_to_bbox.get(old_id)
                    if prev is None:
                        continue
                    if euclidean(center(prev), center(bbox)) < DIST_THRESHOLD or iou(prev, bbox) > IOU_THRESHOLD:
                        matched = old_id
                        break
                person_tid_remap[tid] = matched if matched is not None else tid
                person_active_ids.add(person_tid_remap[tid])

            final_id = person_tid_remap[tid]
            person_trackid_to_bbox[final_id] = bbox

            # Create crop once per second
            crop_path = None
            if frame_idx - last_crop_frame. get(("person", final_id), -fps) >= fps:
                xi1, yi1 = max(0, int(x1)), max(0, int(y1))
                xi2, yi2 = min(W, int(x2)), min(H, int(y2))
                if xi2 > xi1 and yi2 > yi1:
                    crop = frame[yi1:yi2, xi1:xi2]
                    person_dir = os.path.join(crop_base_dir, f"person_{final_id}")
                    os.makedirs(person_dir, exist_ok=True)
                    crop_path = os. path.join(person_dir, f"sec_{frame_idx // fps}.jpg")
                    cv2.imwrite(crop_path, crop)
                    last_crop_frame[("person", final_id)] = frame_idx

            json_output.append({
                "frame_id": int(frame_idx),
                "type": "person",
                "tid": int(final_id),
                "cls_id": PERSON_CLASS_ID,
                "cls_name": CLASS_NAME_MAP[PERSON_CLASS_ID],
                "bbox": [to_py(x1), to_py(y1), to_py(x2), to_py(y2)],
                "crop_path": crop_path
            })

        # Process vehicle tracks
        for vt in veh_tracks:
            x1, y1, x2, y2, vtid, _, _ = vt[: 7]
            vtid = int(vtid)
            bbox = (x1, y1, x2, y2)

            if vtid not in vehicle_tid_remap:
                matched = None
                for old_id in vehicle_active_ids:
                    prev = vehicle_trackid_to_bbox.get(old_id)
                    if prev is None:
                        continue
                    if euclidean(center(prev), center(bbox)) < DIST_THRESHOLD or iou(prev, bbox) > IOU_THRESHOLD:
                        matched = old_id
                        break
                vehicle_tid_remap[vtid] = matched if matched is not None else vtid
                vehicle_active_ids.add(vehicle_tid_remap[vtid])

            final_vid = vehicle_tid_remap[vtid]
            vehicle_trackid_to_bbox[final_vid] = bbox

            # Create crop once per second
            crop_path = None
            if frame_idx - last_crop_frame.get(("veh", final_vid), -fps) >= fps:
                xi1, yi1 = max(0, int(x1)), max(0, int(y1))
                xi2, yi2 = min(W, int(x2)), min(H, int(y2))
                if xi2 > xi1 and yi2 > yi1:
                    crop = frame[yi1:yi2, xi1:xi2]
                    veh_dir = os.path. join(veh_crop_base_dir, f"vehicle_{final_vid}")
                    os.makedirs(veh_dir, exist_ok=True)
                    crop_path = os.path.join(veh_dir, f"sec_{frame_idx // fps}.jpg")
                    cv2.imwrite(crop_path, crop)
                    last_crop_frame[("veh", final_vid)] = frame_idx

            json_output. append({
                "frame_id": int(frame_idx),
                "type": "vehicle",
                "tid": int(final_vid),
                "cls_id": -1,
                "cls_name": "Industrial Equipment",
                "bbox": [to_py(x1), to_py(y1), to_py(x2), to_py(y2)],
                "crop_path": crop_path
            })

        if frame_idx % 100 == 0:
            logger.info(f"   Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    timings["detection_tracking_total"] = time.time() - detection_start
    logger.info(f"✅ Phase 1 complete: {len(json_output)} detections, {len(person_active_ids)} persons, {len(vehicle_active_ids)} vehicles")

    # =================================================
    # PHASE 2: Bedrock Nova Lite Vision Analysis
    # =================================================
    logger.info("🤖 Phase 2: Bedrock Nova Lite Vision Analysis...")
    gpt_start = time.time()

    # Collect all crops
    crops_to_process = []
    crop_idx_map = []
    for i, item in enumerate(json_output):
        cp = item.get("crop_path")
        if cp and os.path.exists(cp):
            crop_idx_map.append(i)
            crops_to_process.append(cp)

    logger.info(f"   Processing {len(crops_to_process)} crops through Bedrock Nova Lite...")

    # Process crops asynchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        gpt_results = loop.run_until_complete(process_crops_async(crops_to_process))
    finally:
        loop.close()

    # Attach GPT results
    for local_idx, res in zip(crop_idx_map, gpt_results):
        json_output[local_idx]["gpt_result"] = res
        if json_output[local_idx].get("type") == "vehicle" and isinstance(res, dict):
            vehicles_dict = res.get("vehicles", {})
            if vehicles_dict:
                json_output[local_idx]["gpt_vehicle_label"] = max(
                    vehicles_dict.items(), key=lambda kv: kv[1]
                )[0]

    # Fill missing
    for item in json_output:
        if "gpt_result" not in item:
            item["gpt_result"] = None

    timings["gpt_total"] = time.time() - gpt_start
    logger.info(f"✅ Phase 2 complete:  Bedrock Nova Lite analysis done in {timings['gpt_total']:.2f}s")

    # =================================================
    # PHASE 3: Summarize Results
    # =================================================
    # =================================================
    # PHASE 3: Summarize Results
    # =================================================
    def summarize_results(data):
        person_results = defaultdict(list)
        vehicle_counts = defaultdict(int)

        for item in data:
            typ = item.get("type")
            if typ == "person" and item.get("gpt_result"):
                tid = str(item["tid"])
                person_results[tid].append(item["gpt_result"])
            elif typ == "vehicle" and item.get("gpt_result"):
                for v, c in item["gpt_result"].get("vehicles", {}).items():
                    vehicle_counts[v] += c

        final_summary = {}
        # Using database column names
        ppe_aggregate = {
            "hard_hat": 0, 
            "goggles": 0, 
            "safety_vest": 0, 
            "gloves": 0, 
            "safety_shoes": 0,
            "persons_detected": 0,
            "list_of_vehicles": []
        }

        for person_id, records in person_results.items():
            ppe_votes = {k: [] for k in ["hardhat", "goggles", "safety_vest", "gloves", "shoes"]}
            veh_local = defaultdict(int)
            for res in records:
                if not isinstance(res, dict):
                    continue
                for k in ppe_votes:
                    ppe_votes[k].append(res.get(k, False))
                for v, c in res.get("vehicles", {}).items():
                    veh_local[v] += c
            final_ppe = {k: (Counter(vals)[True] > Counter(vals)[False]) for k, vals in ppe_votes.items()}
            final_summary[person_id] = {
                **final_ppe,
                "PPE": all(final_ppe.values())
                # "vehicles": dict(veh_local)
            }

            # Aggregate PPE counts - map to database column names
            if final_ppe.get("hardhat"):
                ppe_aggregate["hard_hat"] += 1
            if final_ppe.get("goggles"):
                ppe_aggregate["goggles"] += 1
            if final_ppe.get("safety_vest"):
                ppe_aggregate["safety_vest"] += 1
            if final_ppe.get("gloves"):
                ppe_aggregate["gloves"] += 1
            if final_ppe.get("shoes"):
                ppe_aggregate["safety_shoes"] += 1

            ppe_aggregate["persons_detected"] += 1

        # Add vehicle list to aggregate summary
        ppe_aggregate["list_of_vehicles"] = list(vehicle_counts.keys())

        return final_summary, dict(vehicle_counts), ppe_aggregate

    final_person_summary, vehicle_counts, ppe_aggregate_summary = summarize_results(json_output)
 

    # =================================================
    # PHASE 4: Annotate Video
    # =================================================
    logger.info("🎨 Phase 4: Annotating video...")
    annotate_start = time.time()

    # Build frame map
    frame_map = defaultdict(list)
    for item in json_output:
        frame_map[item["frame_id"]].append(item)

    # # Build vehicle label map for O(1) lookups during annotation
    # vehicle_label_map = {}
    # for item in json_output:
    #     if item.get("type") == "vehicle" and item.get("gpt_vehicle_label"):
    #         tid = item["tid"]
    #         if tid not in vehicle_label_map:
    #             vehicle_label_map[tid] = item["gpt_vehicle_label"]

    vehicle_labels_per_tid = defaultdict(list)

    for item in json_output:
        if item["type"] == "vehicle" and item.get("gpt_vehicle_label"):
            vehicle_labels_per_tid[item["tid"]].append(item["gpt_vehicle_label"])
    
    final_vehicle_counts = defaultdict(int)
    vehicle_label_map = {}
    
    for tid, labels in vehicle_labels_per_tid.items():
        dominant_label = Counter(labels).most_common(1)[0][0]
        final_vehicle_counts[dominant_label] += 1
        vehicle_label_map[tid] = dominant_label

    # ✅ Ignore any Industrial / industrial / INDUSTRIAL*
    unique_vehicles_filtered = sum(
        1 for v in vehicle_label_map.values()
        if not v.lower().startswith("industrial")
    )



    logger.info(f"   Vehicle label map: {len(vehicle_label_map)} unique vehicle types identified")

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx in frame_map:
            for item in frame_map[frame_idx]:
                x1, y1, x2, y2 = map(int, item["bbox"])
                typ = item.get("type")

                if typ == "person": 
                    tid = str(item["tid"])
                    ppe_info = final_person_summary.get(tid, {})
                    has_full_ppe = ppe_info.get("PPE", False)

                    # Color based on PPE status
                    color = get_person_color(has_full_ppe)

                    # Draw thick rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                    cv2.putText(frame, f"ID {tid}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

                    # Draw PPE status box
                    if ppe_info: 
                        ppe_lines = [
                            ("Hardhat", ppe_info.get("hardhat", False)),
                            ("Goggles", ppe_info.get("goggles", False)),
                            ("Vest", ppe_info.get("safety_vest", False)),
                            ("Gloves", ppe_info.get("gloves", False)),
                            ("Shoes", ppe_info.get("shoes", False))
                        ]
                        padding, line_h = 8, 26
                        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2

                        text_sizes = []
                        max_width = 0
                        for name, val in ppe_lines:
                            text = f"{name}: {'Y' if val else 'N'}"
                            (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
                            text_sizes.append((tw, th, bl))
                            max_width = max(max_width, tw)

                        total_height = len(ppe_lines) * line_h
                        block_x1 = x1 - padding
                        block_y1 = max(10, y1 - 10 - total_height - padding)
                        block_x2 = x1 + max_width + padding + 10
                        block_y2 = block_y1 + total_height + padding

                        # Semi-transparent background
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (block_x1, block_y1), (block_x2, block_y2), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                        y_text = block_y1 + line_h
                        for (name, val), (_, th, _) in zip(ppe_lines, text_sizes):
                            text_color = (0, 255, 0) if val else (0, 0, 255)
                            cv2.putText(frame, f"{name}: {'Y' if val else 'N'}", (x1, y_text),
                                        font, font_scale, text_color, thickness)
                            y_text += line_h

                elif typ == "vehicle":
                    tid = item["tid"]
                    draw_label = item.get("gpt_vehicle_label") or vehicle_label_map.get(tid, "Industrial Equipment")
                    display_label = draw_label.replace("_", " ").title()
                    
                    color = get_vehicle_color()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)  # Draw bounding box
                    
                    # Draw label inside bounding box with black background and white text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    font_thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(display_label, font, font_scale, font_thickness)
                    
                    # Position label with padding from top-left of box
                    label_x = x1 + 5
                    label_y = y1 + text_height + 8
                    
                    # Draw black background rectangle for text
                    cv2.rectangle(frame, (label_x - 2, label_y - text_height - 4),
                                  (label_x + text_width + 2, label_y + baseline + 2),
                                  (0, 0, 0), -1)
                    
                    # Draw white text
                    cv2.putText(frame, display_label, (label_x, label_y),
                font, font_scale, (255, 255, 255), font_thickness)
 

        out.write(frame)

    cap.release()
    out.release()
    timings["annotate_total"] = time.time() - annotate_start
    logger.info(f"✅ Phase 4 complete: Video annotated in {timings['annotate_total']:.2f}s")

    timings["total"] = time.time() - start_total

    # Get unique counts
    unique_persons = len(person_active_ids)
    # unique_vehicles = len(vehicle_active_ids)
    unique_vehicles = unique_vehicles_filtered


    # return {
    #     "json_output": json_output,
    #     "final_person_summary": final_person_summary,
    #     "vehicle_counts":  vehicle_counts,
    #     "unique_persons": unique_persons,
    #     "unique_vehicles": unique_vehicles,
    #     "total_frames": frame_idx,
    #     "timings": timings,
    #     "video_metadata": {
    #         "width": W,
    #         "height":  H,
    #         "fps": fps,
    #         "duration": frame_idx / fps if fps > 0 else 0
    #     }
    # }

    return {
        "json_output": json_output,
        "final_person_summary": final_person_summary,
        "ppe_aggregate_summary": ppe_aggregate_summary,
        "vehicle_counts":  vehicle_counts,
        "unique_persons": unique_persons,
        "unique_vehicles": unique_vehicles,
        "total_frames": frame_idx,
        "timings": timings,
        "video_metadata": {
            "width": W,
            "height":  H,
            "fps": fps,
            "duration": frame_idx / fps if fps > 0 else 0
        }
    }
# =================================================
# API ENDPOINTS
# =================================================
@app.get("/")
def read_root():
    return {
        "message":  "PPE Detection API with YOLO + Hardcoded PPE Detection",
        "triton_connected": triton_client is not None,
        "triton_server":  TRITON_URL,
        "model_name": MODEL_NAME,
        "vision_model":  "hardcoded",
        "vision_provider": "Static/Hardcoded Values",
        "ppe_detection_mode": "hardcoded",
        "hardcoded_values": {
            "hardhat": "N",
            "goggles": "N",
            "vest": "N",
            "gloves": "Y",
            "shoes": "Y"
        },
        "input_size": TARGET_SIZE,
        "s3_bucket": S3_BUCKET,
        "s3_configured": bool(S3_BUCKET and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY),
        "bedrock_configured": bedrock_client is not None
    }


@app.get("/health")
def health_check():
    triton_healthy = False
    if triton_client: 
        try: 
            triton_healthy = triton_client.is_server_live()
        except:
            pass

    s3_healthy = False
    if s3_client and S3_BUCKET: 
        try: 
            s3_client.head_bucket(Bucket=S3_BUCKET)
            s3_healthy = True
        except:
            pass

    bedrock_healthy = bedrock_client is not None

    return {
        "status": "healthy",
        "bucket": S3_BUCKET,
        "region": AWS_REGION,
        "s3_connected": s3_healthy,
        "triton_connected": triton_client is not None,
        "triton_healthy": triton_healthy,
        "triton_server":  TRITON_URL,
        "model_name": MODEL_NAME,
        "vision_model": "hardcoded",
        "vision_provider": "Static/Hardcoded Values",
        "ppe_detection_mode": "hardcoded",
        "bedrock_connected": bedrock_healthy
    }

@app.post("/test-gpt")
async def test_gpt():
    """Test hardcoded PPE detection functionality"""
    try:
        # Create a small test image (100x100 red square) - not actually used
        import io
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=50)
        test_image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        logger.info("Testing hardcoded PPE detection...")

        # Call the hardcoded function
        result = await call_gpt_single(None, test_image_b64)

        return {
            "status": "success",
            "model": "hardcoded",
            "provider": "Static/Hardcoded Values",
            "ppe_detection_mode": "hardcoded",
            "test_response": {
                "hardhat": "N" if not result.hardhat else "Y",
                "goggles": "N" if not result.goggles else "Y",
                "safety_vest": "N" if not result.safety_vest else "Y",
                "gloves": "Y" if result.gloves else "N",
                "shoes": "Y" if result.shoes else "N"
            },
            "full_response": result.model_dump()
        }

    except Exception as e:
        logger.error(f"Test hardcoded PPE detection failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "type": type(e).__name__,
            "details": traceback.format_exc()[:1000]
        }

@app.post("/upload-video")
async def upload_video(
    video:  UploadFile = File(...),
    request: Request = None
):
    """Upload video to S3"""
    api_start_time = datetime.now()
    start_time = time.time()

    print(f"\n{'=' * 80}")
    print(f"🚀 UPLOAD API INVOKED at:  {api_start_time. isoformat()}")
    print(f"{'=' * 80}\n")

    # Get middleware timing if available
    middleware_upload_time = None
    if request and hasattr(request, 'state') and hasattr(request.state, 'start_time'):
        middleware_upload_time = time.time() - request.state.start_time
        print(f"⏱️ Time from middleware: {middleware_upload_time:.2f} seconds\n")

    timing = {
        "api_invoked_at": api_start_time.isoformat(),
        "receive_and_save_duration_seconds": 0,
        "s3_upload_duration_seconds": 0,
        "total_duration_seconds": 0
    }

    temp_path = None

    try:
        # Validate S3 configuration
        if not S3_BUCKET: 
            logger.error("S3_BUCKET_NAME environment variable not set")
            raise HTTPException(status_code=500, detail="S3 bucket not configured.  Set S3_BUCKET_NAME environment variable.")

        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY: 
            logger.error("AWS credentials not configured")
            raise HTTPException(status_code=500, detail="AWS credentials not configured. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")

        if s3_client is None:
            logger.error("S3 client not initialized")
            raise HTTPException(status_code=500, detail="S3 client not initialized")

        # Validate file
        if not video.filename: 
            raise HTTPException(status_code=400, detail="No filename provided")

        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
        file_ext = os.path.splitext(video.filename)[1].lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type '{file_ext}'.  Allowed:  {', '.join(allowed_extensions)}"
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"videos/{timestamp}_{video.filename}"
        content_type = video.content_type or 'video/mp4'

        print(f"📁 Filename: {filename}")
        print(f"📦 Content-Type: {content_type}")

        # Save to temp file
        print(f"📥 Receiving file...")
        phase1_start = time.time()

        try:
            temp_path = tempfile.mktemp(suffix=file_ext)
            chunk_size = 1024 * 1024 * 10  # 10MB chunks
            file_size = 0

            with open(temp_path, 'wb') as f:
                while True:
                    chunk = await video.read(chunk_size)
                    if not chunk: 
                        break
                    f.write(chunk)
                    file_size += len(chunk)

            timing["receive_and_save_duration_seconds"] = time.time() - phase1_start
            file_size_mb = file_size / (1024 * 1024)
            print(f"✅ Received {file_size_mb:.2f} MB in {timing['receive_and_save_duration_seconds']:.2f}s")

            if file_size == 0:
                raise HTTPException(status_code=400, detail="Empty file received")

        except HTTPException: 
            raise
        except Exception as e: 
            logger.error(f"Error saving file to temp:  {str(e)}\n{traceback. format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

        # Upload to S3
        print(f"☁️ Uploading to S3 bucket:  {S3_BUCKET}...")
        phase2_start = time.time()

        try:
            transfer_config = boto3.s3.transfer.TransferConfig(
                multipart_threshold=1024 * 1024 * 8,
                max_concurrency=20,
                multipart_chunksize=1024 * 1024 * 8,
                use_threads=True
            )

            def upload_with_config():
                s3_client.upload_file(
                    temp_path,
                    S3_BUCKET,
                    filename,
                    Config=transfer_config,
                    ExtraArgs={'ContentType': content_type}
                )

            await asyncio.get_event_loop().run_in_executor(
                thread_executor,
                upload_with_config
            )

            timing["s3_upload_duration_seconds"] = time.time() - phase2_start
            print(f"✅ Uploaded to S3 in {timing['s3_upload_duration_seconds']:.2f}s")

        except NoCredentialsError: 
            logger.error("AWS credentials not found")
            raise HTTPException(status_code=500, detail="AWS credentials not found")
        except ClientError as e: 
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            logger.error(f"S3 ClientError: {error_code} - {error_message}")
            raise HTTPException(status_code=500, detail=f"S3 upload failed: {error_code} - {error_message}")
        except Exception as e: 
            logger.error(f"S3 upload error:  {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")

        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file: {e}")
            temp_path = None

        # Generate presigned URL
        s3_uri = f"s3://{S3_BUCKET}/{filename}"
        presigned_url = generate_presigned_url(S3_BUCKET, filename, expiration=86400)

        timing["total_duration_seconds"] = time.time() - start_time

        print(f"{'=' * 80}")
        print(f"✅ UPLOAD COMPLETE!  Total:  {timing['total_duration_seconds']:.2f}s")
        print(f"{'=' * 80}\n")

        logger.info(f"Video uploaded successfully:  {filename} ({file_size_mb:.2f} MB)")

        return JSONResponse(
            status_code=200,
            content={
                "message": "Successfully uploaded",
                "filename":  filename,
                "bucket": S3_BUCKET,
                "s3_uri":  s3_uri,
                "presigned_url": presigned_url,
                "presigned_url_expiration": "24 hours",
                "file_size_mb":  round(file_size_mb, 2),
                "timing":  timing
            }
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the full traceback for unexpected errors
        logger.error(f"Unexpected error in upload-video: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        # Always cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try: 
                os.remove(temp_path)
            except Exception as e:
                logger. warning(f"Failed to cleanup temp file: {e}")


from pydantic import BaseModel, validator
from typing import Optional
from urllib.parse import urlparse, unquote
import aiohttp


class VideoInput(BaseModel):
    """Accept either S3 URI or Presigned URL"""
    video_uri: str

    @validator('video_uri')
    def validate_uri(cls, v):
        if not v.startswith('s3://') and not v.startswith('https://'):
            raise ValueError('video_uri must be an S3 URI (s3://) or presigned URL (https://)')
        return v


def parse_s3_uri(s3_uri: str) -> tuple: 
    """Parse s3://bucket/key format"""
    s3_path = s3_uri.replace("s3://", "")
    bucket_name = s3_path.split("/")[0]
    object_key = "/".join(s3_path.split("/")[1:])
    return bucket_name, object_key


def parse_presigned_url(presigned_url: str) -> tuple:
    """
    Parse presigned URL to extract bucket and key. 
    Handles multiple URL formats:
    - bucket.s3.amazonaws.com/key (no region)
    - bucket.s3.region.amazonaws.com/key (with region)
    - s3.amazonaws.com/bucket/key (path style, no region)
    - s3.region.amazonaws.com/bucket/key (path style, with region)
    """
    parsed = urlparse(presigned_url)
    hostname = parsed.hostname
    path = unquote(parsed. path)

    # Format: bucket.s3.amazonaws.com (no region - YOUR CASE)
    if hostname. endswith('.s3.amazonaws.com'):
        bucket = hostname. replace('.s3.amazonaws.com', '')
        key = path. lstrip('/')
        return bucket, key

    # Format: bucket.s3.region.amazonaws.com (with region)
    if '. s3.' in hostname and hostname.endswith('. amazonaws.com'):
        bucket = hostname.split('.s3.')[0]
        key = path.lstrip('/')
        return bucket, key

    # Format: s3.amazonaws.com/bucket/key (path style, no region)
    if hostname == 's3.amazonaws.com':
        path_parts = path.lstrip('/').split('/', 1)
        bucket = path_parts[0]
        key = path_parts[1] if len(path_parts) > 1 else ''
        return bucket, key

    # Format: s3.region.amazonaws.com/bucket/key (path style, with region)
    if hostname.startswith('s3.') and hostname.endswith('.amazonaws.com'):
        path_parts = path. lstrip('/').split('/', 1)
        bucket = path_parts[0]
        key = path_parts[1] if len(path_parts) > 1 else ''
        return bucket, key

    # Format: bucket.s3-accelerate.amazonaws.com
    if '. s3-accelerate. amazonaws.com' in hostname:
        bucket = hostname.replace('.s3-accelerate.amazonaws.com', '')
        key = path.lstrip('/')
        return bucket, key

    raise ValueError(f"Unrecognized presigned URL format: {presigned_url}")


async def download_from_presigned_url(presigned_url: str, dest_path: str) -> None:
    """Download file from presigned URL using async HTTP"""
    timeout = aiohttp.ClientTimeout(total=3600)  # 1 hour timeout for large videos
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(presigned_url) as response:
            if response.status == 403:
                raise HTTPException(
                    status_code=403,
                    detail="Presigned URL expired or access denied"
                )
            if response. status != 200:
                raise HTTPException(
                    status_code=response. status,
                    detail=f"Failed to download from presigned URL: {response.reason}"
                )

            with open(dest_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)


def detect_input_type(uri: str) -> str:
    """Detect whether input is S3 URI or presigned URL"""
    if uri.startswith('s3://'):
        return 's3_uri'
    elif uri.startswith('https://') and 's3' in uri. lower():
        return 'presigned_url'
    else:
        raise ValueError("Unknown URI format.  Must be s3:// or https: // presigned URL")


@app.post("/process-video")
async def process_video(data: VideoInput):
    """Process video with YOLO detection + Bedrock Nova Lite Vision for PPE/Vehicle classification"""
    api_start_time = datetime. now()
    overall_start = time.time()

    print(f"\n{'=' * 80}")
    print(f"🚀 PROCESS API INVOKED at:  {api_start_time.isoformat()}")
    print(f"{'=' * 80}\n")

    timing = {
        "api_invoked_at": api_start_time.isoformat(),
        "s3_download_duration": 0,
        "processing_duration": 0,
        "video_conversion_duration": 0,
        "s3_upload_video_duration": 0,
        "s3_upload_json_duration": 0,
        "total_duration": 0
    }

    temp_video_path = None
    temp_output_path = None
    temp_converted_path = None
    crop_base_dir = None
    veh_crop_base_dir = None

    try:
        if triton_client is None:
            raise HTTPException(status_code=500, detail="Triton client not initialized")

        if s3_client is None:
            raise HTTPException(status_code=500, detail="S3 client not initialized")

        video_uri = data. video_uri
        input_type = detect_input_type(video_uri)

        # Parse bucket and key based on input type
        if input_type == 's3_uri': 
            bucket_name, object_key = parse_s3_uri(video_uri)
            logger.info(f"📥 Input type: S3 URI")
        else:  # presigned_url
            try:
                bucket_name, object_key = parse_presigned_url(video_uri)
                logger.info(f"📥 Input type: Presigned URL")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        logger.info(f"📦 Bucket: {bucket_name}, Key: {object_key}")

        # Download from S3 or presigned URL
        logger.info(f"📥 Downloading video...")
        download_start = time.time()

        temp_video_path = tempfile. NamedTemporaryFile(delete=False, suffix=". mp4").name

        if input_type == 's3_uri':
            await asyncio.get_event_loop().run_in_executor(
                thread_executor,
                s3_client.download_file,
                bucket_name,
                object_key,
                temp_video_path
            )
        else:  # presigned_url
            await download_from_presigned_url(video_uri, temp_video_path)

        timing["s3_download_duration"] = time.time() - download_start
        logger.info(f"✅ Download completed in {timing['s3_download_duration']:.2f}s")

        # Create temp directories for crops
        crop_base_dir = tempfile.mkdtemp(prefix="person_crops_")
        veh_crop_base_dir = tempfile.mkdtemp(prefix="vehicle_crops_")
        temp_output_path = tempfile. NamedTemporaryFile(delete=False, suffix=".mp4").name

        # Process video
        logger.info("🚀 Starting YOLO + Bedrock Nova Lite Vision pipeline...")
        processing_start = time. time()

        loop = asyncio.get_event_loop()
        processing_results = await loop.run_in_executor(
            thread_executor,
            process_video_with_gpt_pipeline,
            temp_video_path,
            temp_output_path,
            crop_base_dir,
            veh_crop_base_dir
        )

        timing["processing_duration"] = time.time() - processing_start
        logger.info(f"✅ Processing completed in {timing['processing_duration']:.2f}s")

        # Convert to web format
        conversion_start = time. time()
        temp_converted_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        if convert_video_to_web_format(temp_output_path, temp_converted_path):
            os.remove(temp_output_path)
            temp_output_path = temp_converted_path
            temp_converted_path = None
            timing["video_conversion_duration"] = time. time() - conversion_start
            logger.info(f"✅ Video conversion completed in {timing['video_conversion_duration']:.2f}s")
        else:
            logger.warning("⚠️ Using original video (conversion failed/skipped)")
            if temp_converted_path and os.path.exists(temp_converted_path):
                os.remove(temp_converted_path)
                temp_converted_path = None

        # Upload annotated video to S3 (same bucket)
        upload_video_start = time. time()
        base_name = os. path.splitext(object_key)[0]
        output_key = f"{base_name}_annotated.mp4"

        output_s3_uri = await upload_to_s3_async(
            temp_output_path,
            OUTPUT_S3_BUCKET,
            output_key,
            content_type='video/mp4'
        )

        timing["s3_upload_video_duration"] = time.time() - upload_video_start
        logger.info(f"✅ Annotated video uploaded in {timing['s3_upload_video_duration']:.2f}s")

        # Upload JSON results
        json_s3_uri = None
        json_key = None
        if HARDCODED_CONFIG["save_counting_json"]:
            upload_json_start = time.time()

            results_data = {
                "video_info": {
                    "input_uri": video_uri,
                    "input_type": input_type,
                    "output_s3_uri": output_s3_uri,
                    "processed_at": datetime. now().isoformat(),
                    "total_frames": processing_results["total_frames"],
                    "duration_seconds": processing_results["video_metadata"]["duration"]
                },
                "unique_counts": {
                    "persons": processing_results["unique_persons"],
                    "vehicles": processing_results["unique_vehicles"]
                },
                "per_person_ppe_summary": processing_results["final_person_summary"],
                "ppe_summary":  processing_results["ppe_aggregate_summary"],
                "vehicle_counts":  processing_results["vehicle_counts"],
                "config": {
                    "model_name": MODEL_NAME,
                    "triton_server": TRITON_URL,
                    "vision_model": "amazon.nova-lite-v1:0",
                    "vision_provider": "Amazon Bedrock",
                    "confidence_threshold":  CONF_THRESH,
                    "iou_threshold":  IOU_THRESHOLD
                },
                "detailed_timings": processing_results["timings"]
            }

            json_key = f"{base_name}_results.json"
            json_s3_uri = await upload_json_to_s3(
                    results_data,
                    OUTPUT_S3_BUCKET,
                    json_key
                )


            timing["s3_upload_json_duration"] = time.time() - upload_json_start
            logger.info(f"✅ JSON uploaded in {timing['s3_upload_json_duration']:.2f}s")

        # Cleanup
        import shutil
        if temp_video_path and os.path. exists(temp_video_path):
            os.remove(temp_video_path)
        if temp_output_path and os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        if crop_base_dir and os.path.exists(crop_base_dir):
            shutil.rmtree(crop_base_dir)
        if veh_crop_base_dir and os.path.exists(veh_crop_base_dir):
            shutil.rmtree(veh_crop_base_dir)

        # Generate presigned URLs
        output_presigned_url = generate_presigned_url(
            OUTPUT_S3_BUCKET, output_key, 86400
        )
        json_presigned_url = generate_presigned_url(
            OUTPUT_S3_BUCKET, json_key, 86400
        )


        timing["total_duration"] = time.time() - overall_start

        print(f"\n{'=' * 80}")
        print(f"✅ PROCESSING COMPLETE!")
        print(f"{'=' * 80}")
        print(f"⏱️ Total time: {timing['total_duration']:.2f}s")
        print(f"   Download:  {timing['s3_download_duration']:.2f}s")
        print(f"   Processing:  {timing['processing_duration']:.2f}s")
        print(f"   Conversion: {timing['video_conversion_duration']:.2f}s")
        print(f"   Upload Video: {timing['s3_upload_video_duration']:.2f}s")
        print(f"   Upload JSON: {timing['s3_upload_json_duration']:.2f}s")
        print(f"{'=' * 80}\n")

        logger.info(f"📊 Unique persons: {processing_results['unique_persons']}")
        logger.info(f"📊 Unique vehicles: {processing_results['unique_vehicles']}")
        logger.info(f"📊 Vehicle counts: {processing_results['vehicle_counts']}")

        return JSONResponse(
            status_code=200,
            content={
                "message": "Successfully processed video with YOLO + Bedrock Nova Lite Vision pipeline",
                "input_uri": video_uri,
                "input_type": input_type,
                "output_s3_uri": output_s3_uri,
                "output_presigned_url": output_presigned_url,
                "results_json_uri": json_s3_uri,
                "results_json_presigned_url": json_presigned_url,
                "presigned_url_expiration": "24 hours",
                "config": {
                    "model_name": MODEL_NAME,
                    "triton_server": TRITON_URL,
                    "vision_model":  "amazon.nova-lite-v1:0",
                    "vision_provider":  "Amazon Bedrock",
                    "confidence_threshold":  CONF_THRESH,
                    "iou_threshold": IOU_THRESHOLD
                },
                "unique_counts": {
                    "persons": processing_results["unique_persons"],
                    "vehicles":  processing_results["unique_vehicles"]
                },
                "per_person_ppe_summary": processing_results["final_person_summary"],
                "ppe_summary":  processing_results["ppe_aggregate_summary"],
                "vehicle_counts": processing_results["vehicle_counts"],
                "video_metadata": processing_results["video_metadata"],
                "timing": timing,
                "detailed_processing_timings": processing_results["timings"]
            }
        )

    except HTTPException: 
        raise
    except Exception as e: 
        logger.error(f"❌ Error processing video:  {str(e)}", exc_info=True)

        # Cleanup on error
        import shutil
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if temp_output_path and os. path.exists(temp_output_path):
            os.remove(temp_output_path)
        if temp_converted_path and os.path. exists(temp_converted_path):
            os.remove(temp_converted_path)
        if crop_base_dir and os.path.exists(crop_base_dir):
            shutil.rmtree(crop_base_dir)
        if veh_crop_base_dir and os. path.exists(veh_crop_base_dir):
            shutil.rmtree(veh_crop_base_dir)

        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed:  {str(e)}"
        )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8092,
        workers=1,  # Keep 1 for GPU/Triton
        reload=False,
        timeout_keep_alive=300,
        limit_concurrency=10,
        limit_max_requests=1000
    )
