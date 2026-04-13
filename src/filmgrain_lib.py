'''
    Film Grain Emulation Library Functions
    --------------------------------------
    Copyright (C) 2026  Albert J. Burton

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import os
import logging
import torch
import numpy as np

try:
    import moderngl
except ImportError:
    logging.warning("[comfyui-jbnodes]: moderngl not installed. Film grain node will not function. Run: pip install moderngl")
    moderngl = None

from ..node_config import GLSL_DIR, FILM_FORMAT_MAP, FILM_FORMAT_NAME_TO_ID
from ..models.filmformat import FilmFormat

def get_film_grain_image(image, **kwargs):
        
    if moderngl is None:
        logging.error("[comfyui-jbnodes] moderngl is not installed. Please run: pip install moderngl")
        return (image,)

    batch_size, height, width, channels = image.shape

    film_size_str = kwargs.get("film_size", "135")
    film_format_id = FILM_FORMAT_NAME_TO_ID.get(film_size_str)
    
    # Fallback to ID if it's already an ID, or try to match prefix for "120"
    if not film_format_id:
        if film_size_str in FILM_FORMAT_MAP:
            film_format_id = film_size_str
        elif film_size_str == "120":
            film_format_id = FILM_FORMAT_NAME_TO_ID.get("120 (6x6)") # Default 120 format

    film_format = FilmFormat.from_dict(FILM_FORMAT_MAP.get(film_format_id, {}))

    if film_format and film_format.frame_size:
        film_width = film_format.frame_size.width
    else:
        film_width = 36.0
        logging.warning("[comfyui-jbnodes] Film format ({film_size_str}) could not be loaded, reverting to default 135 size (36mm).")

        
    grain_type = 0 if kwargs.get("emulsion_type", "Cubic") == "Cubic" else 1
    blend = 0 if kwargs.get("blend_mode", "Soft Light") == "Soft Light" else (1 if kwargs.get("blend_mode") == "Overlay" else 2)

    try:
        shader_path = os.path.join(GLSL_DIR, "film_grain.glsl")
        with open(shader_path, "r") as f:
            shader_code = f.read()

        shader_path = os.path.join(GLSL_DIR, "vertex.glsl")
        with open(shader_path, "r") as f:
            vertex_shader = f.read()
    except FileNotFoundError:
        logging.error("[comfyui-jbnodes] Shader GLSL file not found.")
        return (image,)

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
    
    try:
        prog = ctx.program(vertex_shader = vertex_shader, fragment_shader = shader_code)
    except Exception as e:
        logging.error(f"[comfyui-jbnodes] Shader compilation failed:\n{e}\n")
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
        out_img = np.frombuffer(out_data, dtype = np.float32).copy().reshape((height, width, 4))
        
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
    
    return result