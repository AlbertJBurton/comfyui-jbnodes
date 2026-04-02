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
from .util import get_generalized_sigmoid_lut, get_hd_curve_lut, get_luminosity_lut
from .classes import FilmFormat, HDCurve, FilmStock, Camera

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
FILM_STOCK_JSON_PATH = os.path.join(CURRENT_DIR, "film_stocks.json")
FILTER_JSON_PATH = os.path.join(CURRENT_DIR, "wratten_filters.json")
ILLUMINANT_JSON_PATH = os.path.join(CURRENT_DIR, "illuminants.json")
CONTRAST_FILTER_JSON_PATH = os.path.join(CURRENT_DIR, "contrast_filters.json")
PAPER_JSON_PATH = os.path.join(CURRENT_DIR, "papers.json")
GRAYSCALE_JSON_PATH = os.path.join(CURRENT_DIR, "grayscale.json")
CAMERA_JSON_PATH = os.path.join(CURRENT_DIR, "cameras.json")
SHADER_PATH = os.path.join(CURRENT_DIR, "film_grain.glsl")

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
    curves = ["None"] # default
    
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

FILM_SIZE_MAP = {}
FILM_SIZE_NAMES = []
for film_size in STOCK_DATA["film_formats"]:
        FILM_SIZE_MAP[film_size["name"]] = film_size
        FILM_SIZE_NAMES.append(film_size["name"])

GRAYSCALE_MAP = {}
GRAYSCALE_NAMES = []
for grayscale in GRAYSCALE_DATA["grayscale"]:
    GRAYSCALE_MAP[grayscale["name"]] = grayscale
    GRAYSCALE_NAMES.append(grayscale["name"])

FILTER_MAP = {}
FILTER_NAMES = []
FILTER_MAP["None"] = None
FILTER_NAMES.append("None")
for filter in FILTER_DATA["filters"]:
    FILTER_MAP[filter["name"]] = filter
    FILTER_NAMES.append(filter["name"])

BW_FILTER_MAP = {}
BW_FILTER_NAMES = []
BW_FILTER_MAP["None"] = None
BW_FILTER_NAMES.append("None")
for filter in FILTER_DATA["filters"]:
    if filter.get("id") in [8, 11, 15, 25, 29, 47]: 
        name = filter["name"].split("/")[-1].strip()
        BW_FILTER_MAP[name] = filter
        BW_FILTER_NAMES.append(name)

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

RESOLUTIONS = [
    "Standard", 
    "High Resolution", 
    "Ultra High Resolution" 
]


class FilmAspectRatio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "film_size": (FILM_SIZE_NAMES, {}),
                "resolution": (RESOLUTIONS, {}),
                "swap_dimensions": ("BOOLEAN", {"default": False}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "LATENT")
    RETURN_NAMES = ("width", "height", "aspect_ratio", "empty_latent")
    FUNCTION = "get_aspect_ratio"
    CATEGORY = "JBNodes/Utility"
    DESCRIPTION = """Get a latent image with the aspect ratio of a specific film format."""

    def get_aspect_ratio(self, film_size, resolution, swap_dimensions, batch_size):
        size_data = FILM_SIZE_MAP.get(film_size)

        width = 1024
        height = 1024
        target_aspect = width / height

        for lsize in size_data["latent_sizes"]:
            if lsize["name"].startswith(resolution):
                width = lsize.get("width", 1024)
                height = lsize.get("height", 1024)
                target_aspect = width / height
                break

        if swap_dimensions:
            target_aspect = 1 / target_aspect
            width, height = height, width

        empty_latent = torch.zeros((batch_size, 4, width // 8, height // 8), dtype=torch.float32)
        
        return (width, height, target_aspect, {"samples":empty_latent})

class CropFilmAspectRatio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image": ("IMAGE",),
                "film_size": (FILM_SIZE_NAMES, {}),
                "orientation": (["Auto", "Landscape", "Portrait"], {}),
                "shift": ("FLOAT", {"default": 0.00, "min": -1.00, "max": 1.00, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "enforce_aspect_ratio"
    CATEGORY = "JBNodes/Utility"
    DESCRIPTION = """Crop the image to match the aspect ratio of a specific film format."""

    def enforce_aspect_ratio(self, image, film_size, orientation, shift):

        size_data = FILM_SIZE_MAP.get(film_size)
        if not size_data:
            return (image,) # Safety fallback
            
        _, height, width, _ = image.shape
        current_aspect = width / height

        if orientation == "Auto":
            if width > height:
                target_aspect = size_data["frame_size"][0] / size_data["frame_size"][1]
            else:            
                target_aspect = size_data["frame_size"][1] / size_data["frame_size"][0]
        else:
            if orientation == "Landscape":
                target_aspect = size_data["frame_size"][0] / size_data["frame_size"][1]
            else:
                target_aspect = size_data["frame_size"][1] / size_data["frame_size"][0]

        # Use a small tolerance for floating point comparisons to prevent microscopic 1-pixel jitters
        if abs(current_aspect - target_aspect) < 0.001:
            return (image,)
        
        # Calculate Target Dimensions
        if current_aspect > target_aspect:
            # Image is too wide, preserve height and crop width
            new_width = int(round(height * target_aspect))
            new_height = height
        else:
            # Image is too tall, preserve width and crop height
            new_width = width
            new_height = int(round(width / target_aspect))

        x_offset = 0 
        y_offset = 0
        shift_offset = 0
        
        x_offset = (width - new_width) // 2
        y_offset = (height - new_height) // 2        
        shift_offset = int(((height - new_height) // 2) * (-shift))

        # Apply user adjustment for shift in cropping position. Default is 1.0 (no change). Max 2.0 (double the shift).
        y_offset += shift_offset

        image = image[:, y_offset:y_offset + new_height, x_offset:x_offset + new_width, :]

        return (image,)

class CameraLab:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image": ("IMAGE",),
                "camera": (CAMERA_NAMES, {}),
                "filter": (BW_FILTER_NAMES, {"default": "None"}),
                "film": (STOCK_NAMES, {}),
                "film_size": (FILM_SIZE_NAMES, {}),
            },
            "optional": {
                "light_source": (SOURCE_NAMES, {"default": "Noon Daylight (6500 K)"}),
            }
        }

    RETURN_TYPES = ("CAMERA", "IMAGE")
    RETURN_NAMES = ("camera", "preview")
    FUNCTION = "get_camera"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Classic black-and-white film cameras."""

    def get_camera(self, image, camera, filter, film, light_source, film_size):
        camera_obj = Camera.from_dict(CAMERA_MAP.get(camera))

        film_obj = FilmStock.from_dict(STOCK_MAP.get(film))
        camera_obj.film_stock = film_obj

        source_data = SOURCE_MAP.get(light_source)
        camera_obj.illuminant_key = source_data["key"] if source_data else "D65"

        film_width = 36.0 if film_size == "135" else (70.0 if film_size == "120" else (120.0 if film_size == "4x5" else 240.0))
        camera_obj.film_width = film_width

        if filter != "None":
            filter_data = BW_FILTER_MAP.get(filter)
            if filter_data:
                transmission = filter_data.get("transmission")
                auto_factor = 1.0 / filter_data.get("visual_transmission")
                image = get_filter_image(image, transmission, 1.0, True, auto_factor)
                if isinstance(image, tuple):
                    image = image[0] if len(image) > 0 else image

        camera_image, preview = get_camera_image(image, camera_obj)
        camera_obj.image = camera_image

        return(camera_obj, preview)
    
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
        if not filter_data:
            return (image,) 
        
        transmission = filter_data.get("transmission")
        auto_factor = 1.0 / filter_data.get("visual_transmission")

        return get_filter_image(image, transmission, filter_factor, auto_filter_factor, auto_factor)

class DeveloperLab:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image": ("IMAGE",),
                "camera": ("CAMERA",),
                "developer": (["None"], ),
            },
            "optional": {
                "precision": ("INT", {"default": 4096, "min": 256, "max": 65536, "step": 256}),
                "exposure_index": ("FLOAT", {"default": 0.03, "min": 0.00, "max": 1.00, "step": 0.01}),
                "N_development": ("INT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 1}),
            }        
        }

    # Bypass ComfyUI's strict dropdown validation for dynamic widgets
    @classmethod
    def VALIDATE_INPUTS(s, developer):
        return True

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("preview", "film_negative")
    FUNCTION = "build_spectral_image"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Simulate black and white film stocks with customizable development processes."""

    def build_spectral_image(self, image, camera, precision, exposure_index, N_development, developer = None):
        
        if isinstance(camera, Camera):
            film_stock = camera.film_stock
            illuminant_key = camera.illuminant_key
        else:
            return (None, None) 
        
        if isinstance(film_stock, FilmStock):
            weights = film_stock.weights
            spectral_points = film_stock.spectral_points
            params = film_stock.params
            stock_name = film_stock.name
        else:
            stock_data = STOCK_MAP.get(film_stock, {})
            weights = stock_data.get("weights", [0.33, 0.33, 0.33])
            spectral_points = stock_data.get("spectral_points", None)
            params = stock_data.get("params", {"slope": 1.8, "toe": 0.2, "shoulder": 0.8})
            stock_name = str(film_stock)

        # Re-associate the string from the dropdown with the actual HDCurve object
        curve = None
        if film_stock.hd_curves and developer != "None":
            for c in film_stock.hd_curves:
                # This must perfectly match the string format generated in the API route
                display_name = f"{c.name} ({c.time}m at {c.temp}C)"
                if display_name == developer:
                    curve = c
                    break

        slope = params.get("slope", 1.8)
        toe = params.get("toe", 0.2)
        shoulder = params.get("shoulder", 0.8)

        if not curve:
            char_lut = get_generalized_sigmoid_lut(slope, toe, shoulder, precision)
        else:
            char_lut = get_hd_curve_lut(curve, precision, ei=exposure_index, dev_offset=N_development)
            try:
                logging.info(f"[comfyui-jbnodes] applying {stock_name} - {curve.name} characteristic curve with EI: {exposure_index}, Dev Offset: {N_development}")
            except:
                pass

        return get_spectral_image(image, weights, None, illuminant_key, char_lut)

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
    CATEGORY = "JBNodes/Utility"
    DESCRIPTION = """Create a grayscale image with a specific sprectral curve."""

    def build_grayscale_image(self, image, method):
        grayscale_data = GRAYSCALE_MAP.get(method)
        weights = grayscale_data.get("weights", [0.33, 0.33, 0.33])       
        return get_grayscale_image(image, weights)
    
class FilmGrainLab:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "iso": ("INT", {"default": 100, "min": 25, "max": 3200, "step": 1}),
                "film_size": (["135","120","4x5","8x10"], {}),
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

        film_width = 36.0 if kwargs.get("film_size", "135") == "135" else (70.0 if kwargs.get("film_size") == "120" else (120.0 if kwargs.get("film_size") == "4x5" else 240.0))
        grain_type = 0 if kwargs.get("emulsion_type", "Cubic") == "Cubic" else 1
        blend = 0 if kwargs.get("blend_mode", "Soft Light") == "Soft Light" else (1 if kwargs.get("blend_mode") == "Overlay" else 2)

        shader_path = os.path.join(CURRENT_DIR, "film_grain.glsl")
        with open(shader_path, "r") as f:
            shader_code = f.read()

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