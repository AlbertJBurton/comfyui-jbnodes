// Dynamic Film Format Latent Image Sizes Handler for ComfyUI Custom Nodes
// -----------------------------------------------------------------------
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

app.registerExtension({
    name: "JBNodes.DynamicLatentSizes",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Ensure this matches the exact name of the node in Python
        if (nodeData.name === "FilmAspectRatio") { 
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Find the widgets by their Python dictionary keys
                const formatWidget = this.widgets.find((w) => w.name === "film_format");
                const resolutionWidget = this.widgets.find((w) => w.name === "resolution");

                if (formatWidget && resolutionWidget) {
                    const originalCallback = formatWidget.callback;
                    
                    // Attach an event listener to the film_size dropdown
                    formatWidget.callback = async function (value, ...args) {
                        if (originalCallback) {
                            originalCallback.apply(this, [value, ...args]);
                        }
                        
                        // Prefer value from callback args (LiteGraph standard), fallback to widget value
                        const formatName = value || formatWidget.value;
                        if (!formatName) return;
                        
                        try {
                            // Query our custom Python API route
                            const response = await api.fetchApi(`/jbnodes/latent_sizes?film_format=${encodeURIComponent(formatName)}`);
                            const sizes = await response.json();
                            
                            // Mutate the target widget's available options
                            resolutionWidget.options.values = sizes;
                            
                            // Check if the current value is still valid; if not, reset to the first available option
                            if (!sizes.includes(resolutionWidget.value)) {
                                resolutionWidget.value = sizes[0];
                            }
                            
                            // Force ComfyUI to redraw the node visually so the UI updates
                            app.graph.setDirtyCanvas(true, false);
                            
                        } catch (err) {
                            console.error("[comfyui-jbnodes] Error fetching film format latent sizes:", err);
                        }
                    };
                    
                    // Trigger the callback once on instantiation to populate the initial state
                    setTimeout(() => {
                        if (formatWidget.callback) {
                            formatWidget.callback.call(formatWidget, formatWidget.value);
                        }
                    }, 100);
                }
                
                return r;
            };

            // Add onConfigure to handle workflow loading
            // Without this, loading a saved workflow won't trigger the UI to fetch the film formats
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                
                const formatWidget = this.widgets?.find((w) => w.name === "film_format");
                if (formatWidget && formatWidget.callback) {
                    formatWidget.callback.call(formatWidget, formatWidget.value);
                }
                
                return r;
            };
        }
    }
});