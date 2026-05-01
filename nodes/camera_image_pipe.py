'''
    Camera Image Pipe Nodes for ComfyUI Custom Nodes
    ------------------------------------------------
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

class CameraImagePipeLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "filename": ("STRING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),            
            },
        }

    RETURN_TYPES = ("CAMERA_PIPE_LINE", )
    RETURN_NAMES = ("pipe", )
    FUNCTION = "execute"
    CATEGORY = "JBNodes/Utility/Pipe"
    DESCRIPTION = """Package multiple inputs into a single tuple for easier management of complex pipelines. This node does not perform any processing on the inputs, it simply passes them through as a single object."""      

    def execute(self, model = None, positive = None, negative = None, latent = None, vae = None, clip = None, filename = "", seed = 0):
        
        camera_pipe_line = (model, positive, negative, latent, vae, clip, filename, seed)

        return (camera_pipe_line, )

class CameraImagePipeDecomposer:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"pipe": ("CAMERA_PIPE_LINE",)},
            }

    RETURN_TYPES = ("CAMERA_PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "STRING", "INT", )
    RETURN_NAMES = ("pipe", "model", "positive", "negative", "latent", "vae", "clip", "filename", "seed", )
    FUNCTION = "execute"
    CATEGORY = "JBNodes/Utility/Pipe"
    DESCRIPTION = """Unpack a camera pipe line tuple into its individual components for use in downstream nodes. This node does not perform any processing on the outputs, it simply passes them through as separate objects."""

    def execute(self, pipe):

        model, positive, negative, latent, vae, clip, filename, seed = pipe
        
        return (pipe, model, positive, negative, latent, vae, clip, filename, seed, )


class CameraImagePipeComposer:
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pipe": ("CAMERA_PIPE_LINE",)},
                "optional": {
                    "model": ("MODEL",),
                    "positive": ("CONDITIONING",),
                    "negative": ("CONDITIONING",),
                    "latent": ("LATENT",),
                    "vae": ("VAE",),
                    "clip": ("CLIP",),
                    "filename": ("STRING",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                },
            }

    RETURN_TYPES = ("CAMERA_PIPE_LINE", )
    RETURN_NAMES = ("pipe", )
    FUNCTION = "execute"
    CATEGORY = "JBNodes/Utility/Pipe"
    DESCRIPTION = """Update specific elements of the camera pipe line tuple. This node allows you to modify one or more components of the pipe line without having to decompose and recompose the entire tuple manually."""

    def execute(self, pipe, model = None, positive = None, negative = None, latent = None, vae = None, clip = None, filename = None, seed = None):
       
        new_model, new_pos, new_neg, new_latent, new_vae, new_clip, new_filename, new_seed = pipe

        if model is not None:
            new_model = model
        
        if positive is not None:
            new_pos = positive

        if negative is not None:
            new_neg = negative

        if latent is not None:
            new_latent = latent

        if vae is not None:
            new_vae = vae

        if clip is not None:
            new_clip = clip
            
        if filename is not None:
            new_filename = filename
            
        if seed is not None:
            new_seed = seed
       
        pipe = new_model, new_pos, new_neg, new_latent, new_vae, new_clip, new_filename, new_seed
       
        return (pipe,)
    

class CaameraImagePipeSwitch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Input": ("INT", {"default": 1, "min": 1, "max": 2}),
                "pipe1": ("CAMERA_PIPE_LINE",),
                "pipe2": ("CAMERA_PIPE_LINE",)
            }
        }
 
    RETURN_TYPES = ("CAMERA_PIPE_LINE", )
    RETURN_NAMES = ("pipe", )
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "JBNodes/Utility/Pipe"
    DESCRIPTION = """Switch between two camera pipe lines based on the value of the Input integer. If Input is 1, pipe1 will be output. If Input is 2, pipe2 will be output. This allows for dynamic switching between different branches of a pipeline without having to manually reconnect nodes."""

    def execute(self, Input, pipe1, pipe2):
    
        if Input == 1:
            return (pipe1, )
        else:
            return (pipe2, )

