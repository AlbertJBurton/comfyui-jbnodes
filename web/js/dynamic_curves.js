// Dynamic H&D Curves Handler for ComfyUI Custom Nodes
// ---------------------------------------------------
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

async function updateDeveloperLabDeveloperWidget(spectralNode, stockName, savedDeveloper = null) {
    const developerWidget = spectralNode.widgets?.find((w) => w.name === "developer");
    if (!developerWidget) return;

    if (!stockName) {
        developerWidget.options.values = ["None"];
        if (developerWidget.value !== "None") {
            developerWidget.value = "None";
        }
        app.graph.setDirtyCanvas(true, true);
        return;
    }

    try {
        const response = await api.fetchApi(`/jbnodes/hd_curves?stock_name=${encodeURIComponent(stockName)}`);
        const curves = await response.json();
        
        developerWidget.options.values = curves;
        
        let targetDeveloper = savedDeveloper || spectralNode.savedDeveloper || developerWidget.value;
        if (!curves.includes(targetDeveloper)) {
            targetDeveloper = curves[0];
        }
        developerWidget.value = targetDeveloper;
        
        if (curves.includes(targetDeveloper)) {
            spectralNode.savedDeveloper = null;
        }
        
        app.graph.setDirtyCanvas(true, true);
    } catch (err) {
        console.error("[comfyui-jbnodes] Error fetching HD curves:", err);
    }
}

function getFilmFromLink(linkId, app) {
    let currentLinkId = linkId;
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
        
        if (originNode.type === "CameraLab") {
            const filmWidget = originNode.widgets?.find(w => w.name === "film");
            if (filmWidget) {
                return filmWidget.value;
            }
        }
        
        return null;
    }
    return null;
}

app.registerExtension({
    name: "JBNodes.DynamicHDCurves",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
        // 1. Handle CameraLab film widget changes
        if (nodeData.name === "CameraLab") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const filmWidget = this.widgets?.find((w) => w.name === "film");
                if (filmWidget) {
                    const originalCallback = filmWidget.callback;
                    filmWidget.callback = async function (value, ...args) {
                        if (originalCallback) {
                            originalCallback.apply(this, [value, ...args]);
                        }
                        
                        const stockName = value || filmWidget.value;
                        if (!stockName) return;
                        
                        // Find connected DeveloperLab nodes and update them
                        if (this.outputs) {
                            const findConnectedDeveloperLabs = (linkIds) => {
                                let developerLabs = [];
                                for (const linkId of linkIds) {
                                    const link = app.graph.links[linkId];
                                    if (!link) continue;
                                    
                                    const targetNode = app.graph.getNodeById(link.target_id);
                                    if (!targetNode) continue;
                                    
                                    if (targetNode.type === "DeveloperLab") {
                                        developerLabs.push(targetNode);
                                    } else if (targetNode.type === "Reroute") {
                                        const rerouteOutput = targetNode.outputs ? targetNode.outputs[0] : null;
                                        if (rerouteOutput && rerouteOutput.links) {
                                            developerLabs = developerLabs.concat(findConnectedDeveloperLabs(rerouteOutput.links));
                                        }
                                    }
                                }
                                return developerLabs;
                            };

                            for (const out of this.outputs) {
                                if (out.type === "CAMERA" && out.links) {
                                    const developerLabs = findConnectedDeveloperLabs(out.links);
                                    for (const targetNode of developerLabs) {
                                        updateDeveloperLabDeveloperWidget(targetNode, stockName);
                                    }
                                }
                            }
                        }
                    }.bind(this);
                }
                return r;
            };
        }

        // 2. Handle DeveloperLab connections and initial load
        if (nodeData.name === "DeveloperLab") { 
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, slotIndex, isConnected, link_info, ioSlot) {
                const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
                
                // type === 1 is LiteGraph.INPUT
                if (type === 1 && link_info) {
                    const input = this.inputs[slotIndex];
                    if (input.name === "camera") {
                        if (isConnected) {
                            const stockName = getFilmFromLink(input.link, app);
                            if (stockName) {
                                updateDeveloperLabDeveloperWidget(this, stockName);
                            }
                        } else {
                            // Disconnected
                            updateDeveloperLabDeveloperWidget(this, null);
                        }
                    }
                }
                
                return r;
            };
            
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                let savedDeveloper = null;
                
                if (info && info.widgets_values) {
                    const developerIndex = this.widgets?.findIndex(w => w.name === "developer");
                    if (developerIndex !== -1 && developerIndex < info.widgets_values.length) {
                        savedDeveloper = info.widgets_values[developerIndex];
                        this.savedDeveloper = savedDeveloper;
                    }
                }

                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                
                // Wait for graph to be fully populated
                setTimeout(() => {
                    const cameraInput = this.inputs?.find(i => i.name === "camera");
                    if (cameraInput && cameraInput.link) {
                        const stockName = getFilmFromLink(cameraInput.link, app);
                        if (stockName) {
                            updateDeveloperLabDeveloperWidget(this, stockName, savedDeveloper);
                        }
                    }
                }, 0);
                
                return r;
            };
        }
    }
});
