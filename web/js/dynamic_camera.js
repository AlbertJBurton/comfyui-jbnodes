// Dynamic Camera Handler for ComfyUI Custom Nodes
// -----------------------------------------------
// Copyright (C) 2026  Albert J. Burton
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

async function updateCameraLabWidgets(cameraNode, formatName, savedCamera = null, savedFilm = null) {
    const cameraWidget = cameraNode.widgets?.find((w) => w.name === "camera");
    const filmWidget = cameraNode.widgets?.find((w) => w.name === "film");

    if (cameraWidget && filmWidget && formatName) {
        try {
            // Query our custom camera API route
            let response = await api.fetchApi(`/jbnodes/cameras?film_format=${encodeURIComponent(formatName)}`);
            const cameras = await response.json();
            
            // Mutate the target widget's available options
            cameraWidget.options.values = cameras;
            
            // Determine the target value: saved value, current value, or fallback to first option
            let targetCamera = savedCamera || cameraNode.savedCamera || cameraWidget.value;
            if (cameras.length > 0 && !cameras.includes(targetCamera)) {
                targetCamera = cameras[0];
            }
            cameraWidget.value = targetCamera;
            if (cameras.includes(targetCamera)) {
                cameraNode.savedCamera = null;
            }

            response = await api.fetchApi(`/jbnodes/film_stocks?film_format=${encodeURIComponent(formatName)}`);
            const films = await response.json();

            // Mutate the target widget's available options
            filmWidget.options.values = films;
            
            let targetFilm = savedFilm || cameraNode.savedFilm || filmWidget.value;
            if (films.length > 0 && !films.includes(targetFilm)) {
                targetFilm = films[0];
            }
            filmWidget.value = targetFilm;
            if (films.includes(targetFilm)) {
                cameraNode.savedFilm = null;
            }
            
            if (filmWidget.callback) {
                filmWidget.callback(targetFilm);
            }
            
            // Force ComfyUI to redraw the node visually so the UI updates
            app.graph.setDirtyCanvas(true, false);
            
        } catch (err) {
            console.error("[comfyui-jbnodes] Error fetching film format cameras and film stocks:", err);
        }
    }
}

function getFilmFormatFromLink(linkId, app) {
    let currentLinkId = linkId;
    // Prevent infinite loops in case of weird graph cycles
    let visited = new Set();
    
    while (currentLinkId != null) {
        if (visited.has(currentLinkId)) return null;
        visited.add(currentLinkId);
        
        const link = app.graph.links[currentLinkId];
        if (!link) return null;
        
        const originNode = app.graph.getNodeById(link.origin_id);
        if (!originNode) return null;
        
        if (originNode.type === "Reroute") {
            const input = originNode.inputs ? originNode.inputs[0] : null;
            if (input && input.link != null) {
                currentLinkId = input.link;
                continue;
            } else {
                return null;
            }
        }
        
        const formatWidget = originNode.widgets?.find(w => w.name === "film_format");
        if (formatWidget) {
            return formatWidget.value;
        }
        
        return null;
    }
    return null;
}

app.registerExtension({
    name: "JBNodes.DynamicCamera",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "CameraLab") { 
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
                if (onConnectionsChange) {
                    onConnectionsChange.apply(this, arguments);
                }
                
                // type === 1 is LiteGraph.INPUT
                if (type === 1 && connected && link_info) {
                    const input = this.inputs[index];
                    if (input.name === "film_format") {
                        const formatName = getFilmFormatFromLink(input.link, app);
                        if (formatName) {
                            updateCameraLabWidgets(this, formatName);
                        }
                    }
                }
            };

            // Add onConfigure to handle workflow loading
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                let savedCamera = null;
                let savedFilm = null;
                
                if (info && info.widgets_values) {
                    const cameraIndex = this.widgets?.findIndex(w => w.name === "camera");
                    const filmIndex = this.widgets?.findIndex(w => w.name === "film");
                    
                    if (cameraIndex !== -1 && cameraIndex < info.widgets_values.length) {
                        savedCamera = info.widgets_values[cameraIndex];
                        this.savedCamera = savedCamera;
                    }
                    if (filmIndex !== -1 && filmIndex < info.widgets_values.length) {
                        savedFilm = info.widgets_values[filmIndex];
                        this.savedFilm = savedFilm;
                    }
                }

                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                
                // We use setTimeout to wait for the entire graph to finish configuring.
                // This ensures that the originNode has its saved widget values restored
                // before we try to read them.
                setTimeout(() => {
                    const input = this.inputs?.find(inp => inp.name === "film_format");
                    if (input && input.link != null) {
                        const formatName = getFilmFormatFromLink(input.link, app);
                        if (formatName) {
                            updateCameraLabWidgets(this, formatName, savedCamera, savedFilm);
                        }
                    }
                }, 0);
                
                return r;
            };
        }

        if (nodeData.name === "FilmAspectRatio" || nodeData.name === "CropFilmAspectRatio") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                const node = this; // Capture the node instance

                const formatWidget = node.widgets?.find((w) => w.name === "film_format");
                if (formatWidget) {
                    const originalCallback = formatWidget.callback;
                    formatWidget.callback = function (value, ...args) {
                        if (originalCallback) {
                            originalCallback.apply(this, [value, ...args]);
                        }

                        let formatName = value;
                        if (typeof value === "object" && value !== null) {
                            formatName = formatWidget.value;
                        } else {
                            formatName = value || formatWidget.value;
                        }
                        if (!formatName || typeof formatName !== "string") return;

                        // Find all connected CameraLab nodes and update them
                        const output = node.outputs?.find(o => o.name === "film_format" || o.type === "FILMFORMAT");
                        if (output && output.links) {
                            const findConnectedCameraLabs = (linkIds) => {
                                let cameraLabs = [];
                                for (const linkId of linkIds) {
                                    const link = app.graph.links[linkId];
                                    if (!link) continue;
                                    
                                    const targetNode = app.graph.getNodeById(link.target_id);
                                    if (!targetNode) continue;
                                    
                                    if (targetNode.type === "CameraLab") {
                                        cameraLabs.push(targetNode);
                                    } else if (targetNode.type === "Reroute") {
                                        const rerouteOutput = targetNode.outputs ? targetNode.outputs[0] : null;
                                        if (rerouteOutput && rerouteOutput.links) {
                                            cameraLabs = cameraLabs.concat(findConnectedCameraLabs(rerouteOutput.links));
                                        }
                                    }
                                }
                                return cameraLabs;
                            };
                            
                            const cameraLabs = findConnectedCameraLabs(output.links);
                            for (const targetNode of cameraLabs) {
                                updateCameraLabWidgets(targetNode, formatName);
                            }
                        }
                    };
                }
                
                return r;
            };
        }
    }
});
