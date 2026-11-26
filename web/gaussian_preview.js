/**
 * ComfyUI SHARP Gaussian Stereo Preview Widget
 * Interactive 3D Gaussian Splatting viewer using gsplat.js
 * Modified from PozzettiAndrea/comfy-3d-viewers for this standalone stereo node.
 */

import { app } from "../../../scripts/app.js";

// Auto-detect extension folder name (detects the installed custom-node folder)
const EXTENSION_FOLDER = (() => {
    const url = import.meta.url;
    const match = url.match(/\/extensions\/([^/]+)\//);
    return match ? match[1] : "ComfyUI-SHARP-Gaussian-Stereo";
})();

console.log("[Gaussian Stereo] Loading extension...");

app.registerExtension({
    name: "comfyui.gaussianstereo",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (
            nodeData.name === "SharpGaussianStereoPreview"
        ) {
            console.log("[Gaussian Stereo] Registering Preview Gaussian node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Create container for viewer + info panel
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.backgroundColor = "#1a1a1a";
                container.style.overflow = "hidden";

                // Create iframe for gsplat.js viewer
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.flex = "1 1 0";
                iframe.style.minHeight = "0";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#1a1a1a";

                // Point to gsplat.js HTML viewer (with cache buster)
                iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer_gaussian.html?v=` + Date.now();

                // Create info panel
                const infoPanel = document.createElement("div");
                infoPanel.style.backgroundColor = "#1a1a1a";
                infoPanel.style.borderTop = "1px solid #444";
                infoPanel.style.padding = "6px 12px";
                // Keep the metadata row reserved even when temp-file details are hidden.
                infoPanel.style.boxSizing = "border-box";
                infoPanel.style.height = "39px";
                infoPanel.style.minHeight = "39px";
                infoPanel.style.flex = "0 0 39px";
                infoPanel.style.fontSize = "10px";
                infoPanel.style.fontFamily = "monospace";
                infoPanel.style.color = "#ccc";
                infoPanel.style.lineHeight = "1.3";
                infoPanel.style.flexShrink = "0";
                infoPanel.style.overflow = "hidden";
                infoPanel.innerHTML = "";

                // Add iframe and info panel to container
                container.appendChild(iframe);
                container.appendChild(infoPanel);

                // Add widget with required options
                const widget = this.addDOMWidget("preview_gaussian", "GAUSSIAN_PREVIEW", container, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                // Store references
                this.gaussianViewerIframe = iframe;
                this.gaussianInfoPanel = infoPanel;


                // Track iframe load state
                let iframeLoaded = false;
                iframe.addEventListener('load', () => {
                    iframeLoaded = true;
                });

                // Listen for messages from iframe
                window.addEventListener('message', async (event) => {
                    // Ignore messages from other viewer iframes/nodes.
                    if (event.source !== iframe.contentWindow) return;

                    // Keep the user's current node width and adjust only its height.
                    // The iframe reports a target height that already includes its controls.
                    if (event.data?.type === 'VIEWER_RESIZE_REQUEST' && Number.isFinite(event.data.iframeHeight)) {
                        const currentIframeHeight = iframe.clientHeight;
                        const currentNodeWidth = this.size?.[0] || 768;
                        const currentNodeHeight = this.size?.[1] || 1024;
                        const delta = Math.ceil(event.data.iframeHeight - currentIframeHeight);
                        if (Math.abs(delta) >= 2) {
                            this.setSize([currentNodeWidth, Math.max(240, currentNodeHeight + delta)]);
                        }
                        return;
                    }

                    // Handle screenshot messages
                    if (event.data.type === 'SCREENSHOT' && event.data.image) {
                        try {
                            // Convert base64 data URL to blob
                            const base64Data = event.data.image.split(',')[1];
                            const byteString = atob(base64Data);
                            const arrayBuffer = new ArrayBuffer(byteString.length);
                            const uint8Array = new Uint8Array(arrayBuffer);

                            for (let i = 0; i < byteString.length; i++) {
                                uint8Array[i] = byteString.charCodeAt(i);
                            }

                            const blob = new Blob([uint8Array], { type: 'image/png' });

                            // Generate filename with timestamp
                            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                            const filename = `gaussian-screenshot-${timestamp}.png`;

                            // Create FormData for upload
                            const formData = new FormData();
                            formData.append('image', blob, filename);
                            formData.append('type', 'output');
                            formData.append('subfolder', '');

                            // Upload to ComfyUI backend
                            const response = await fetch('/upload/image', {
                                method: 'POST',
                                body: formData
                            });

                            if (response.ok) {
                                const result = await response.json();
                                console.log('[Gaussian Stereo] Screenshot saved:', result.name);
                            } else {
                                throw new Error(`Upload failed: ${response.status}`);
                            }

                        } catch (error) {
                            console.error('[Gaussian Stereo] Error saving screenshot:', error);
                        }
                    }
                    // Handle copy image to clipboard messages
                    else if (event.data.type === 'COPY_IMAGE' && event.data.success) {
                        console.log('[Gaussian Stereo] Image copied to clipboard successfully');
                    }
                    else if (event.data.type === 'COPY_IMAGE' && !event.data.success) {
                        console.error('[Gaussian Stereo] Failed to copy image to clipboard:', event.data.error);
                    }
                    // Handle error messages from iframe
                    else if (event.data.type === 'MESH_ERROR' && event.data.error) {
                        console.error('[Gaussian Stereo] Error from viewer:', event.data.error);
                        if (infoPanel) {
                            infoPanel.innerHTML = `<div style="color: #ff6b6b;">Error: ${event.data.error}</div>`;
                        }
                    }
                });

                // Set initial node size
                this.setSize([768, 1024]);

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    console.log("[Gaussian Stereo] onExecuted called with:", message);
                    onExecuted?.apply(this, arguments);

                    // Check for errors
                    if (message?.error && message.error[0]) {
                        infoPanel.innerHTML = `<div style="color: #ff6b6b;">Error: ${message.error[0]}</div>`;
                        return;
                    }

                    // The message IS the UI data (not message.ui)
                    if (message?.ply_file && message.ply_file[0]) {
                        const filename = message.ply_file[0];
                        const displayName = message.filename?.[0] || filename;
                        const fileSizeMb = message.file_size_mb?.[0] || 'N/A';

                        // Extract camera parameters if provided
                        const extrinsics = message.extrinsics?.[0] || null;
                        const intrinsics = message.intrinsics?.[0] || null;

                        // Show PLY metadata only for files explicitly saved to output.
                        // For temp previews, retain the same reserved panel height but leave it blank.
                        if (message.save_to_output?.[0] !== false) {
                            infoPanel.innerHTML = `
                                <div style="display: grid; grid-template-columns: auto 1fr; gap: 2px 8px;">
                                    <span style="color: #888;">File:</span>
                                    <span style="color: #6cc;">${displayName}</span>
                                    <span style="color: #888;">Size:</span>
                                    <span>${fileSizeMb} MB</span>
                                </div>
                            `;
                        } else {
                            infoPanel.innerHTML = "";
                        }

                        // ComfyUI serves files via /view API — parse type/subfolder/filename
                        // Handles both absolute (/home/.../output/sub/file) and relative (output/sub/file) paths
                        let filepath;
                        const normalized = filename.replace(/\\/g, '/');
                        const pathMatch = normalized.match(/(?:^|\/)(output|input|temp)\/(.+)$/);
                        if (pathMatch) {
                            const [, type, relPath] = pathMatch;
                            const parts = relPath.split('/');
                            const fname = parts.pop();
                            const subfolder = parts.join('/');
                            filepath = `/view?filename=${encodeURIComponent(fname)}&type=${type}&subfolder=${encodeURIComponent(subfolder)}`;
                        } else {
                            // Fallback: use just the basename
                            const basename = normalized.split('/').pop();
                            filepath = `/view?filename=${encodeURIComponent(basename)}&type=output&subfolder=`;
                        }

                        // Send URL to iframe — gsplat.js fetches directly with streaming
                        const sendUrl = () => {
                            if (!iframe.contentWindow) {
                                console.error("[Gaussian Stereo] Iframe contentWindow not available");
                                return;
                            }
                            iframe.contentWindow.postMessage({
                                type: "LOAD_MESH_URL",
                                url: filepath,
                                filename: filename,
                                extrinsics: extrinsics,
                                intrinsics: intrinsics,
                                timestamp: Date.now()
                            }, "*");
                        };

                        if (iframeLoaded) {
                            sendUrl();
                        } else {
                            setTimeout(sendUrl, 500);
                        }
                    }
                };

                return r;
            };
        }
    }
});
