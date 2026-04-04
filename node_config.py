'''
    ComfyUI Nodes for Film Simulation and Photographic Principles
    -------------------------------------------------------------
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

import json
import os

from server import PromptServer
from aiohttp import web

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
IMAGES_DIR = os.path.join(CURRENT_DIR, "images")
CONFIG_DIR = os.path.join(CURRENT_DIR, "config")
GLSL_DIR = os.path.join(CURRENT_DIR, "glsl")

FILM_STOCK_JSON_PATH = os.path.join(CONFIG_DIR, "film_stocks.json")
FILTER_JSON_PATH = os.path.join(CONFIG_DIR, "wratten_filters.json")
ILLUMINANT_JSON_PATH = os.path.join(CONFIG_DIR, "illuminants.json")
CONTRAST_FILTER_JSON_PATH = os.path.join(CONFIG_DIR, "contrast_filters.json")
PAPER_JSON_PATH = os.path.join(CONFIG_DIR, "papers.json")
GRAYSCALE_JSON_PATH = os.path.join(CONFIG_DIR, "grayscale.json")
CAMERA_JSON_PATH = os.path.join(CONFIG_DIR, "cameras.json")

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
