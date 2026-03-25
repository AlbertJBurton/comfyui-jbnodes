import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

app.registerExtension({
    name: "JBNodes.DynamicHDCurves",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Ensure this matches the exact name of the node in Python
        if (nodeData.name === "FilmLab") { 
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Find the widgets by their Python dictionary keys
                const stockWidget = this.widgets.find((w) => w.name === "film_stock");
                const curveWidget = this.widgets.find((w) => w.name === "hd_curve");

                if (stockWidget && curveWidget) {
                    const originalCallback = stockWidget.callback;
                    
                    // Attach an event listener to the film_stock dropdown
                    stockWidget.callback = async function (value, ...args) {
                        if (originalCallback) {
                            originalCallback.apply(this, [value, ...args]);
                        }
                        
                        // Prefer value from callback args (LiteGraph standard), fallback to widget value
                        const stockName = value || stockWidget.value;
                        if (!stockName) return;
                        
                        try {
                            // Query our custom Python API route
                            const response = await api.fetchApi(`/jbnodes/get_hd_curves?stock_name=${encodeURIComponent(stockName)}`);
                            const curves = await response.json();
                            
                            // Mutate the target widget's available options
                            curveWidget.options.values = curves;
                            
                            // Check if the current value is still valid; if not, reset to the first available option
                            if (!curves.includes(curveWidget.value)) {
                                curveWidget.value = curves[0];
                            }
                            
                            // Force ComfyUI to redraw the node visually so the UI updates
                            app.graph.setDirtyCanvas(true, false);
                            
                        } catch (err) {
                            console.error("[JBNodes] Error fetching HD curves:", err);
                        }
                    };
                    
                    // Trigger the callback once on instantiation to populate the initial state
                    setTimeout(() => {
                        if (stockWidget.callback) {
                            stockWidget.callback.call(stockWidget, stockWidget.value);
                        }
                    }, 100);
                }
                
                return r;
            };

            // Add onConfigure to handle workflow loading
            // Without this, loading a saved workflow won't trigger the UI to fetch the custom curves
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                
                const stockWidget = this.widgets?.find((w) => w.name === "film_stock");
                if (stockWidget && stockWidget.callback) {
                    stockWidget.callback.call(stockWidget, stockWidget.value);
                }
                
                return r;
            };
        }
    }
});