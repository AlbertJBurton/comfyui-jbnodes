
from ..node_config import FILM_PROMPT_DECK_NAMES, FILM_PROMPT_DECK_MAP

class PromptLab:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "deck": (FILM_PROMPT_DECK_NAMES, {}),
                "set": ([None], {}),
                "prompt_index": ("INT", {"default": 1, "min": 1, "max": 50, "step": 1, "control_after_generate": True}),
            },
            "optional": {
                "prepend_positive": ("STRING", {"default": ""}),
                "prepend_negative": ("STRING", {"default": ""}),
            }        
        }

    # Bypass ComfyUI's strict dropdown validation for dynamic widgets
    @classmethod
    def VALIDATE_INPUTS(s, set, prompt_index):
        return True

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "filename")
    FUNCTION = "get_prompt"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Retrieve a prompt for the selected film prompt deck."""

    def get_prompt(self, deck, set, prompt_index, prepend_positive, prepend_negative):

        deck_dict = FILM_PROMPT_DECK_MAP.get(deck, {})
        model_name = deck_dict.get("model_name", "")

        sets = deck_dict.get("sets", [])
        set_dict = None
        for s in sets:
            if s.get("name") == set:
                set_dict = s
                break

        if not set_dict:
            print(f"[comfyui-jbnodes] Warning: Set '{set}' not found in deck '{deck}'.")
            return ("")
        
        boilerplate = set_dict.get("boilerplate", "") if set_dict.get("boilerplate") else None
        boilerplate_positive = boilerplate.get("positive", "") if boilerplate else ""
        boilerplate_negative = boilerplate.get("negative", "") if boilerplate else ""
        
        prompts = set_dict.get("prompts", [])
        if not prompts:
            print(f"[comfyui-jbnodes] Warning: No prompts found in set '{set}' of deck '{deck}'.")
            return ("")
        
        if prompt_index < 1 or prompt_index > len(prompts):
            print(f"[comfyui-jbnodes] Warning: Prompt index {prompt_index} is out of range for set '{set}' of deck '{deck}'.")
            return ("") 
        
        filename = f"{model_name}_{set_dict.get('abbr', '')}_Prompt-{prompt_index}".replace(" ", "_").replace("/", "_").replace("_&_", "_")

        positive_prompt =  prepend_positive + ", " + prompts[prompt_index - 1][0] + ", " + boilerplate_positive
        negative_prompt =  prepend_negative + ", " + prompts[prompt_index - 1][1] + ", " + boilerplate_negative

        return (positive_prompt, negative_prompt, filename)