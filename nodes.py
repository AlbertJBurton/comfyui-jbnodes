import json
import os
import numpy as np
import torch
import dataclasses
import logging

try:
    import moderngl
except ImportError:
    logging.warning("[comfyui-jbnodes]: moderngl not installed. Film grain node will not function. Run: pip install moderngl")
    moderngl = None

from comfy import model_management
from server import PromptServer
from aiohttp import web

from PIL import ImageEnhance, Image

from .filters import get_filter_image
from .spectral import get_spectral_image, get_camera_image
from .print import get_print_image
from .grayscale import get_grayscale_image
from .util import get_generalized_sigmoid_lut, get_hd_curve_lut, get_luminosity_lut, FilmStock, HDCurve, Camera

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
FILM_STOCK_JSON_PATH = os.path.join(CURRENT_DIR, "film_stocks.json")
FILTER_JSON_PATH = os.path.join(CURRENT_DIR, "wratten_filters.json")
ILLUMINANT_JSON_PATH = os.path.join(CURRENT_DIR, "illuminants.json")
CONTRAST_FILTER_JSON_PATH = os.path.join(CURRENT_DIR, "contrast_filters.json")
PAPER_JSON_PATH = os.path.join(CURRENT_DIR, "papers.json")
GRAYSCALE_JSON_PATH = os.path.join(CURRENT_DIR, "grayscale.json")
CAMERA_JSON_PATH = os.path.join(CURRENT_DIR, "cameras.json")

with open(FILM_STOCK_JSON_PATH, 'r') as film:
    STOCK_DATA = json.load(film)

with open(FILTER_JSON_PATH, 'r') as filter:
    FILTER_DATA = json.load(filter)

with open(ILLUMINANT_JSON_PATH, 'r') as source:
    SOURCE_DATA = json.load(source)

with open(CONTRAST_FILTER_JSON_PATH, 'r') as contrast:
    CONTRAST_FILTER_DATA = json.load(contrast)

with open(PAPER_JSON_PATH, 'r') as paper:
    PAPER_DATA = json.load(paper)

with open(CAMERA_JSON_PATH, 'r') as camera:
    CAMERA_DATA = json.load(camera)

with open(GRAYSCALE_JSON_PATH, 'r') as grayscale:
    GRAYSCALE_DATA = json.load(grayscale)

# H&D Curves API route
@PromptServer.instance.routes.get("/jbnodes/get_hd_curves")
async def get_hd_curves(request):
    """Returns a list of HD curve names for a given film stock name."""
    stock_name = request.rel_url.query.get("stock_name", "")
    curves = ["None"] # Safe default
    
    for group in STOCK_DATA.get("film_stock_groups", []):
        for stock in group.get("stocks", []):
            if stock.get("name") == stock_name:
                hd_curves = stock.get("hd_curves", [])
                if hd_curves:
                    # Format names as "Developer @ Time mins (Temp C)"
                    curves = [f"{c['name']} ({c['time']}m at {c['temp']}C)" for c in hd_curves]
                break
                
    return web.json_response(curves)

# Create mappings
STOCK_MAP = {}
STOCK_NAMES = []
for group in STOCK_DATA["film_stock_groups"]:
    for stock in group["stocks"]:
        STOCK_MAP[stock["name"]] = stock
        STOCK_NAMES.append(stock["name"])

GRAYSCALE_MAP = {}
GRAYSCALE_NAMES = []
for grayscale in GRAYSCALE_DATA["grayscale"]:
    GRAYSCALE_MAP[grayscale["name"]] = grayscale
    GRAYSCALE_NAMES.append(grayscale["name"])

FILTER_MAP = {}
FILTER_NAMES = []
for filter in FILTER_DATA["filters"]:
    FILTER_MAP[filter["name"]] = filter
    FILTER_NAMES.append(filter["name"])

SOURCE_MAP = {}
SOURCE_NAMES = []
for source in SOURCE_DATA["sources"]:
    SOURCE_MAP[source["label"]] = source
    SOURCE_NAMES.append(source["label"])

CONTRAST_MAP = {}
CONTRAST_NAMES = []
for filter in CONTRAST_FILTER_DATA["filters"]:
    CONTRAST_MAP[filter["label"]] = filter
    CONTRAST_NAMES.append(filter["label"])

GRADED_PAPER_MAP = {}
GRADED_PAPER_NAMES = []
for paper in PAPER_DATA["graded_papers"]:
    GRADED_PAPER_MAP[paper["name"]] = paper
    GRADED_PAPER_NAMES.append(paper["name"])

CAMERA_MAP = {}
CAMERA_NAMES = []
for camera in CAMERA_DATA["cameras"]:
    CAMERA_MAP[camera["name"]] = camera
    CAMERA_NAMES.append(camera["name"])

DEVELOPERS = [
    "None", 
    "Standard (D-76)", 
    "Contrast (HC-110)", 
    "Acutance (Rodinal)", 
    "Fine Grain (Xtol)"
]

class CameraLab:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image": ("IMAGE",),
                "camera": (CAMERA_NAMES, {"default": "Mamiya RB67"}),
                "film": (STOCK_NAMES, {"default": "Kodak / Tri-X 400"}),
            },
            "optional": {
                "light_source": (SOURCE_NAMES, {"default": "Noon Daylight (6500 K)"}),
            }
        }

    RETURN_TYPES = ("CAMERA", "IMAGE")
    RETURN_NAMES = ("camera_image", "preview")
    FUNCTION = "get_camera"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Classic black-and-white film cameras."""

    def get_camera(self, image, camera, film, light_source):
        camera_obj = Camera.from_dict(CAMERA_MAP.get(camera))

        film_obj = FilmStock.from_dict(STOCK_MAP.get(film))
        camera_obj.film_stock = film_obj

        source_data = SOURCE_MAP.get(light_source)
        camera_obj.illuminant_key = source_data["key"] if source_data else "D65"

        camera_obj.image = get_camera_image(image, camera_obj)

        return(camera_obj, camera_obj.image)
    
class FilmLab:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "film_stock": (STOCK_NAMES, {"default": "Kodak / Tri-X 400"}),
                "hd_curve": (["None"], ),
            }, 
        }
    
    # Bypass ComfyUI's strict dropdown validation for dynamic widgets
    @classmethod
    def VALIDATE_INPUTS(s, film_stock, hd_curve):
        return True

    RETURN_TYPES = ("FILM_STOCK", "HDCURVE")
    RETURN_NAMES = ("film_stock", "hd_curve")
    FUNCTION = "get_film_stock"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Simulated black and white film stocks."""

    def get_film_stock(self, film_stock, hd_curve):
        stock_data = STOCK_MAP.get(film_stock)
        stock = FilmStock.from_dict(stock_data)

        # Re-associate the string from the dropdown with the actual HDCurve object
        curve = None
        if stock.hd_curves and hd_curve != "None":
            for c in stock.hd_curves:
                # This must perfectly match the string format generated in the API route
                display_name = f"{c.name} ({c.time}m at {c.temp}C)"
                if display_name == hd_curve:
                    curve = c
                    break

        return (stock, curve)

class PrintLabMultigrade:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "film_negative": ("IMAGE",),
            },
            "optional": {
                "contrast_filter": (CONTRAST_NAMES, {"default": "2"}),
                "exposure_secs": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "print_image"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Simulate printing a negative to photographic paper."""

    def print_image(self, film_negative, contrast_filter, exposure_secs):
        filter_data = CONTRAST_MAP.get(contrast_filter)
        contrast_factor = filter_data.get("factor")
        return get_print_image(film_negative, contrast_factor=contrast_factor, exposure_secs=exposure_secs)

class PrintLabGraded:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "film_negative": ("IMAGE",),
            },
            "optional": {
                "graded_paper": (GRADED_PAPER_NAMES, {"default": "Ilford / Ilfobrom Galerie FB / Grade 2"}),
                "exposure_secs": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 60.0, "step": 0.1}),
                "precision": ("INT", {"default": 4096, "min": 256, "max": 65536, "step": 256}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "print_image"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Simulate printing a negative to graded photographic paper."""

    def print_image(self, film_negative, graded_paper, exposure_secs, precision=4096):
        paper_data = GRADED_PAPER_MAP.get(graded_paper)
        grade = paper_data.get("grade")
        
        # Read the raw empirical points
        hd_curve_raw = paper_data.get("hd_curve_points", None)
        d_max = paper_data.get("d_max", 2.1)
        d_min = paper_data.get("d_min", 0.04)
        
        hd_curve_points = None
        contrast_factor = None
        
        if hd_curve_raw:
            # Cast the JSON array of coordinates into a PyTorch tensor
            device = model_management.get_torch_device()
            hd_curve_points = torch.tensor(hd_curve_raw, dtype=torch.float32, device=device)
        else:
            # Fallback to the algorithmic method if no empirical data exists
            filter_data = CONTRAST_MAP.get(grade)
            contrast_factor = filter_data.get("factor")

        return get_print_image(film_negative, contrast_factor=contrast_factor, exposure_secs=exposure_secs, hd_curve_points=hd_curve_points, d_max=d_max, d_min=d_min, precision=precision)

class FilterLab:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "filter": (FILTER_NAMES, {"default": "No. 0 - Colorless"}),
            },
            "optional": {
                "filter_factor": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 10.00, "step": 0.01}),
                "auto_filter_factor": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_filter"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Apply a Wratten filter to an image."""

    def apply_filter(self, image, filter, filter_factor, auto_filter_factor):
        filter_data = FILTER_MAP.get(filter)
        transmission = filter_data.get("transmission")
        auto_factor = 1.0 / filter_data.get("visual_transmission")

        return get_filter_image(image, transmission, filter_factor, auto_filter_factor, auto_factor)

class SpectralLab:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image": ("IMAGE",),
                "film_stock": ("FILM_STOCK",), 
                "developer": (DEVELOPERS, {"default": "Standard (D-76)"}),
                "light_source": (SOURCE_NAMES, {"default": "Noon Daylight (6500 K)"}),
            },
            "optional": {
                "hd_curve": ("HDCURVE",),
                "precision": ("INT", {"default": 4096, "min": 256, "max": 65536, "step": 256}),
                "contrast_index_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "push_pull_stops": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1}),
                "exposure_index": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "N_development": ("INT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 1}),
            }        
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("preview", "film_negative")
    FUNCTION = "build_spectral_image"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Simulate black and white film stocks with customizable development processes."""

    def build_spectral_image(self, image, film_stock, developer, light_source, hd_curve=None, precision=1024, contrast_index_offset=0.0, push_pull_stops=0.0, exposure_index=0.1, N_development=0):
        # Extract data cleanly depending on whether FilmLab passed the object or a string
        if isinstance(film_stock, FilmStock):
            weights = film_stock.weights
            spectral_points = film_stock.spectral_points
            params = film_stock.params
            stock_name = film_stock.name
            
            # Prioritize the wired HDCURVE from FilmLab. Fallback to the first curve if unwired.
            if hd_curve:
                hd_data = hd_curve
            #elif film_stock.hd_curves:
            #    hd_data = film_stock.hd_curves[0]
            else:
                hd_data = None
        else:
            stock_data = STOCK_MAP.get(film_stock, {})
            weights = stock_data.get("weights", [0.33, 0.33, 0.33])
            spectral_points = stock_data.get("spectral_points", None)
            params = stock_data.get("params", {"slope": 1.8, "toe": 0.2, "shoulder": 0.8})
            stock_name = str(film_stock)
            hd_data = hd_curve

        slope = params.get("slope", 1.8)
        toe = params.get("toe", 0.2)
        shoulder = params.get("shoulder", 0.8)

        if contrast_index_offset != 0.0:
            ci = 1 / slope
            slope = 1 / (ci + contrast_index_offset)
            
        if "Contrast" in developer:
            slope += 0.2
            toe = max(0.0, toe - 0.05)
        elif "Acutance" in developer:
            slope = max(0.1, slope - 0.05)
            shoulder = max(0.0, shoulder - 0.10)
        elif "Fine Grain" in developer:
            slope = max(0.1, slope - 0.10)
            toe += 0.05
        elif "None" in developer:
            slope = 2.2
            toe = 0

        slope += (push_pull_stops * 0.1)
        
        source_data = SOURCE_MAP.get(light_source)
        illuminant_name = source_data["key"] if source_data else "D65"

        if not hd_data:
            char_lut = get_generalized_sigmoid_lut(slope, toe, shoulder, precision)
        else:
            char_lut = get_hd_curve_lut(hd_data, precision, ei=exposure_index, dev_offset=N_development)
            try:
                logging.info(f"[SpectralLab] using {stock_name} - {hd_data.name} characteristic curve with EI: {exposure_index}, Dev Offset: {N_development}")
            except:
                pass

        return get_spectral_image(image, weights, spectral_points, illuminant_name, char_lut)

class GrayscaleLab:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image": ("IMAGE",),            
                "method": (GRAYSCALE_NAMES, {"default": "ITU-R BT.709 Luminance"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "build_grayscale_image"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Create a grayscale image with a specific sprectral curve."""

    def build_grayscale_image(self, image, method):
        grayscale_data = GRAYSCALE_MAP.get(method)
        weights = grayscale_data.get("weights", [0.33, 0.33, 0.33])       
        return get_grayscale_image(image, weights)
    
class ShaderLab:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "iso": ("INT", {"default": 100, "min": 25, "max": 3200, "step": 1}),
                "film_size": (["35mm","120","4x5","8x10"], {}),
                "emulsion_type": (["Cubic", "Tabular"], {}),
                "film_grit": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "halation": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "emulsion_softness": ("FLOAT", {"default": 0.75, "min": 0.00, "max": 1.50, "step": 0.01}),
                "blend_mode": (["Soft Light", "Overlay", "Linear Light"], {}),
                "luminance_peak_bias": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "signal_noise_ratio": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 2.0, "step": 0.01}),
                "algorithmic_octaves": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
                "morphological_variance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "temporal_entropy": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_shader"
    CATEGORY = "JBNodes"
    DESCRIPTION = "Execute custom GLSL shaders with an extended parameter set using moderngl."

    def apply_shader(self, image, **kwargs):
        if moderngl is None:
            logging.error("[comfyui-jbnodes] moderngl is not installed. Please run: pip install moderngl")
            return (image,)
            
        batch_size, height, width, channels = image.shape

        film_width = 36.0 if kwargs.get("film_size", "35mm") == "35mm" else (70.0 if kwargs.get("film_size") == "120" else (120.0 if kwargs.get("film_size") == "4x5" else 240.0))
        grain_type = 0 if kwargs.get("emulsion_type", "Cubic") == "Cubic" else 1
        blend = 0 if kwargs.get("blend_mode", "Soft Light") == "Soft Light" else (1 if kwargs.get("blend_mode") == "Overlay" else 2)

        shader_code = '''#version 330
// --- REQUIRED COMFYUI DECLARATIONS ---
in vec2 uv;
out vec4 FragColor;
uniform sampler2D image;

// --- CUSTOM UNIFORMS ---
uniform int iso; // ISO Scale 
uniform float morphological_variance; // Morphological Variance 
uniform float luminance_peak_bias; // Luminance Peak Bias 
uniform float signal_noise_ratio; // signal_noise Ratio 
uniform float temporal_entropy; // Temporal Entropy 
uniform float film_grit; // Film Grit
uniform float halation; // Halation Strength
uniform int grain_type;     // Emulsion Type (0 = Cubic, 1 = Tabular)
uniform int algorithmic_octaves;     // Algorithmic Octaves 
uniform int blend;     // Blend Mode (0=Soft Light, 1=Overlay, 2=Linear Light)
uniform float emulsion_softness; // Emulsion Softness (Input 0.0-2.0)
uniform float film_width; // Width of film (based on film format)

// --- EMULSION SOFTNESS (IRRADIATION) ---
vec3 sampleEmulsion(sampler2D tex, vec2 uv, float softness) {
    // Bypass blur if set to 0 for performance
    if (softness <= 0.0) {
        return texture(tex, uv).rgb;
    }
    
    vec2 texSize = vec2(textureSize(tex, 0));
    vec2 texel = 1.0 / texSize;
    vec3 color = vec3(0.0);
    float weightSum = 0.0;
    
    // Make softness resolution-independent (e.g., softness 1.0 = 0.1% of image width blur)
    float blurRadius = softness * (texSize.x * 0.001);
    
    // 5x5 Micro-Gaussian Blur
    for (float x = -2.0; x <= 2.0; x += 1.0) {
        for (float y = -2.0; y <= 2.0; y += 1.0) {
            vec2 offset = vec2(x, y) * texel * blurRadius;
            float w = exp(-(x*x + y*y) / 2.0); 
            color += texture(tex, uv + offset).rgb * w;
            weightSum += w;
        }
    }
    return color / weightSum;
}

// --- HALATION (LIGHT PIPING) ---
vec3 sampleHalation(sampler2D tex, vec2 uv, float strength, float formatMultiplier) {
    if (strength <= 0.0) return vec3(0.0);
    
    vec2 texSize = vec2(textureSize(tex, 0));
    vec2 texel = 1.0 / texSize;
    vec3 halo = vec3(0.0);
    float weightSum = 0.0;
    
    // Halation needs a fairly large radius to be visible.
    // We scale it by formatMultiplier (smaller film = relatively larger halation)
    // AND we scale it by the image resolution (texSize.x) so the effect is consistent
    // regardless of the input image size. 0.02 means 2% of the image width for 35mm.
    float radius = texSize.x * 0.02 * formatMultiplier; 
    float goldenAngle = 2.39996323;
    
    // 32-tap sparse spiral blur for a smooth, broad glow
    for (int i = 0; i < 32; i++) {
        // Normalize r so the max radius is actually 'radius'
        float r = sqrt(float(i) + 0.5) * (radius / 5.65685); 
        float theta = float(i) * goldenAngle;
        vec2 offset = vec2(cos(theta), sin(theta)) * r * texel;
        
        // Sample the negative image
        vec3 s_neg = texture(tex, uv + offset).rgb;
        
        // Convert to POSITIVE to easily identify highlights
        vec3 s_pos = 1.0 - s_neg;
        float luma = dot(s_pos, vec3(0.299, 0.587, 0.114));
        
        // Isolate highlights (anything brighter than 50% gray in the positive scene)
        float bright = smoothstep(0.5, 1.0, luma); 
        
        float w = exp(-float(i) / 10.0); 
        halo += s_pos * bright * w;
        weightSum += w;
    }
    
    vec3 haloColor = vec3(0.0);
    if (weightSum > 0.0) {
        haloColor = halo / weightSum;
    }
    
    // Return the POSITIVE glow
    return haloColor * strength;
}

// --- 3D SIMPLEX NOISE (For Cubic Grains) ---
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}

float snoise3D(vec3 v){
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 = v - i + dot(i, C.xxx) ;
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );
    vec3 x1 = x0 - i1 + 1.0 * C.xxx;
    vec3 x2 = x0 - i2 + 2.0 * C.xxx;
    vec3 x3 = x0 - 1.0 + 3.0 * C.xxx;
    i = mod(i, 289.0 );
    vec4 p = permute( permute( permute(
                i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
              + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
              + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
    float n_ = 1.0/7.0;
    vec3  ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z *ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );
    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3) ) );
}

// --- ARTIFACT-FREE HASH ---
vec3 hash33(vec3 p3) {
    p3 = fract(p3 * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz+33.33);
    return fract((p3.xxy + p3.yxx)*p3.zyx);
}

// --- 3D VORONOI / CELLULAR NOISE ---
vec2 voronoi3D(vec3 x) {
    vec3 n = floor(x);
    vec3 f = fract(x);
    float F1 = 8.0;
    float F2 = 8.0;
    for(int k=-1; k<=1; k++)
    for(int j=-1; j<=1; j++)
    for(int i=-1; i<=1; i++) {
        vec3 g = vec3(float(i),float(j),float(k));
        vec3 o = hash33( n + g );
        vec3 r = g - f + o;
        float d = dot(r,r);
        if( d < F1 ) {
            F2 = F1;
            F1 = d;
        } else if( d < F2 ) {
            F2 = d;
        }
    }
    return vec2(sqrt(F1), sqrt(F2));
}

// --- BLEND MODES ---
float blendMode(float base, float blend, int mode) {
    if (mode == 1) {
        // Overlay
        return (base < 0.5) ? 2.0 * base * blend : 1.0 - 2.0 * (1.0 - base) * (1.0 - blend);
    } else if (mode == 2) {
        // Linear Light
        return clamp(base + 2.0 * blend - 1.0, 0.0, 1.0);
    } else {
        // Soft Light (Default)
        return (blend < 0.5) 
            ? base - (1.0 - 2.0 * blend) * base * (1.0 - base) 
            : base + (2.0 * blend - 1.0) * (((base <= 0.25) ? ((16.0 * base - 12.0) * base + 4.0) * base : sqrt(base)) - base);
    }
}

vec3 applyBlend(vec3 base, vec3 blend, int mode) {
    return vec3(blendMode(base.r, blend.r, mode), 
                blendMode(base.g, blend.g, mode), 
                blendMode(base.b, blend.b, mode));
}

void main() {
    vec2 res = vec2(textureSize(image, 0));
    
    // --- FILM FORMAT SCALING ---
    // Multiplier relative to the 35mm baseline
    float formatMultiplier = 36.0 / film_width;
    
    // Step 1: Coordinate Normalization
    float safe_iso = max(float(iso), 1.0); 
    
    // Base grain size as a fraction of image width (e.g., 0.2% of width for ISO 100, 35mm)
    // This makes the grain size resolution-independent.
    float baseGrainUV = 0.002;
    
    // Scale the grain size down as the physical format gets larger
    float scaleFactorUV = sqrt(safe_iso / 100.0) * baseGrainUV * formatMultiplier;
    
    // Calculate aspect ratio to keep grains square
    float aspect = res.y / res.x;
    vec2 uv_aspect = uv * vec2(1.0, aspect);
    
    vec3 uv_noise = vec3(uv_aspect / scaleFactorUV, temporal_entropy);

    // Step 2: Base Luminance Extraction & Emulsion Softness
    // Scale softness by format: larger formats have less relative emulsion blur per-pixel
    float softness = emulsion_softness * formatMultiplier;
    vec4 originalColor = texture(image, uv);
    vec3 blurredRGB = sampleEmulsion(image, uv, softness);
    
    // Step 2.5: Add Halation
    // We sample halation from the ORIGINAL image, not the blurred one, to keep the glow sharp.
    // The sampleHalation function now returns a POSITIVE glow.
    vec3 positiveGlow = sampleHalation(image, uv, halation, formatMultiplier);
    
    // Convert our current negative blurred image to positive
    vec3 positiveBase = 1.0 - blurredRGB;
    
    // Add the glow using a Screen blend mode to prevent clipping
    vec3 positiveFinal = 1.0 - (1.0 - positiveBase) * (1.0 - positiveGlow);
    
    // Convert back to negative for the rest of the pipeline
    blurredRGB = 1.0 - positiveFinal;
    
    vec4 baseColor = vec4(blurredRGB, originalColor.a); // Preserve original alpha
    
    float L = 0.299 * baseColor.r + 0.587 * baseColor.g + 0.114 * baseColor.b;

    // Step 3: Luminance-Dependent Attenuation Masking
    float denom = max(luminance_peak_bias, 1.0 - luminance_peak_bias);
    denom = max(denom, 0.0001);
    float M = 1.0 - pow((L - luminance_peak_bias) / denom, 2.0);
    M = clamp(M, 0.0, 1.0);
    M *= smoothstep(1.0, 0.8, L); 

    // Step 4: Emulsion Morphology Generation
    float n_raw = 0.0;
    float amp = 1.0;
    float max_amp = 0.0;
    vec3 p = uv_noise;
    mat2 m2 = mat2(0.8, -0.6, 0.6, 0.8);

    if (grain_type == 0) {
        // Cubic Protocol (Simplex)
        for(int i = 0; i < 4; i++) {
            if(i >= algorithmic_octaves) break;
            n_raw += snoise3D(p) * amp;
            max_amp += amp;
            p.xy = m2 * p.xy; 
            p *= 2.0;         
            amp *= morphological_variance;  
        }
        max_amp = max(max_amp, 0.0001);
        n_raw = (n_raw / max_amp) * 0.5 + 0.5;
        
        // Map film_grit (0.0 to 1.0) to a spread factor.
        // 0.0 = softest (spread 0.5 -> bounds 0.0, 1.0)
        // 1.0 = harshest (spread 0.01 -> bounds 0.49, 0.51)
        float spread = mix(0.5, 0.01, film_grit);
        n_raw = smoothstep(0.5 - spread, 0.5 + spread, n_raw);
    } else {
        // Tabular Protocol (Voronoi)
        for(int i = 0; i < 4; i++) {
            if(i >= algorithmic_octaves) break;
            vec2 v = voronoi3D(p);
            n_raw += (v.y - v.x) * amp; 
            max_amp += amp;
            p.xy = m2 * p.xy; 
            p *= 2.0;         
            amp *= morphological_variance;  
        }
        max_amp = max(max_amp, 0.0001);
        n_raw = n_raw / max_amp; 
        
        // LUMINOSITY FIX: Center the Tabular noise
        n_raw = (n_raw - 0.25) * 2.0 + 0.5; 
    }

    // Step 5: Integration and Photometric Blending
    float n_final = (n_raw - 0.5) * M * signal_noise_ratio;
    
    vec3 grainBlend = clamp(vec3(n_final + 0.5), 0.0, 1.0);
    vec3 finalColor = applyBlend(baseColor.rgb, grainBlend, blend);

    FragColor = vec4(finalColor, baseColor.a);
}
'''

        try:
            # Force EGL backend. Ubuntu 24.04 (Wayland) explicitly blocks headless GLX/X11 
            # contexts, causing 'BadAccess' during X_GLXMakeCurrent.
            ctx = moderngl.create_context(standalone=True, backend='egl')
        except Exception as e:
            logging.warning(f"[comfyui-jbnodes] EGL context failed, falling back to auto-detect: {e}")
            try:
                ctx = moderngl.create_context(standalone=True)
            except Exception as e2:
                logging.error(f"[comfyui-jbnodes] Failed to initialize moderngl context: {e2}")
                return (image,)

        # Simple 2D full screen quad
        vertices = np.array([
            # x,    y,      u,   v
            -1.0, -1.0,   0.0, 0.0,
             1.0, -1.0,   1.0, 0.0,
            -1.0,  1.0,   0.0, 1.0,
             1.0,  1.0,   1.0, 1.0,
        ], dtype='f4')
        
        vbo = ctx.buffer(vertices.tobytes())
        
        vertex_shader = '''
        #version 330
        in vec2 in_vert;
        in vec2 in_texcoord;
        out vec2 uv;
        void main() {
            uv = in_texcoord;
            gl_Position = vec4(in_vert, 0.0, 1.0);
        }
        '''
        
        try:
            prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=shader_code)
        except Exception as e:
            logging.error(f"[comfyui-jbnodes] Shader Compilation Failed:\n{e}")
            ctx.release()
            return (image,)
            
        vao = ctx.vertex_array(prog, [(vbo, '2f 2f', 'in_vert', 'in_texcoord')])
        out_images = []
        
        # Process each image in the latent batch sequentially
        for i in range(batch_size):
            img_batch = image[i].numpy()
            
            # Moderngl prefers 4-channel textures for float32 (RGBA)
            if channels == 3:
                alpha = np.ones((height, width, 1), dtype=np.float32)
                img_batch = np.concatenate([img_batch, alpha], axis=-1)
                
            texture = ctx.texture((width, height), 4, img_batch.tobytes(), dtype='f4')
            texture.use(0)
            
            # Safely bind uniforms if they are requested in the shader program
            if 'image' in prog:
                prog['image'].value = 0
            if 'resolution' in prog:
                prog['resolution'].value = (width, height)
                
            # Inject calculated custom variables
            if 'film_width' in prog: prog['film_width'].value = film_width
            if 'grain_type' in prog: prog['grain_type'].value = grain_type
            if 'blend' in prog: prog['blend'].value = blend
                
            for key, val in kwargs.items():
                if key in prog:
                    prog[key].value = val
                    
            # Setup an empty framebuffer to catch the GL execution
            fbo_texture = ctx.texture((width, height), 4, dtype='f4')
            fbo = ctx.framebuffer(color_attachments=[fbo_texture])
            fbo.use()
            
            # Render the quad using the active shader program
            vao.render(moderngl.TRIANGLE_STRIP)
            
            # Pull the data back to CPU and reshape it
            out_data = fbo_texture.read()
            out_img = np.frombuffer(out_data, dtype=np.float32).reshape((height, width, 4))
            
            # Strip the temporary alpha channel if the original input was standard RGB
            if channels == 3:
                out_img = out_img[..., :3]
                
            out_images.append(torch.from_numpy(out_img))
            
            # Clean up the iteration resources immediately to prevent VRAM spiking
            texture.release()
            fbo_texture.release()
            fbo.release()
            
        # Nuke the context footprint
        vbo.release()
        prog.release()
        vao.release()
        ctx.release()
        
        # Clamp bounds strictly since GL math can push values outside the 0.0-1.0 PyTorch standard
        result = torch.clamp(torch.stack(out_images), 0.0, 1.0)
        return (result,)
    