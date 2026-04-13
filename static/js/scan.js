function analyzeImageQuality(imageElement) {
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d", { willReadFrequently: true });
    const width = 240;
    const height = Math.max(1, Math.round((imageElement.naturalHeight / imageElement.naturalWidth) * width));

    canvas.width = width;
    canvas.height = height;
    context.drawImage(imageElement, 0, 0, width, height);

    const imageData = context.getImageData(0, 0, width, height).data;
    const grayscale = new Float32Array(width * height);
    let brightnessTotal = 0;

    for (let index = 0; index < width * height; index += 1) {
        const offset = index * 4;
        const gray =
            0.299 * imageData[offset] +
            0.587 * imageData[offset + 1] +
            0.114 * imageData[offset + 2];
        grayscale[index] = gray;
        brightnessTotal += gray;
    }

    const laplacianValues = [];
    for (let y = 1; y < height - 1; y += 1) {
        for (let x = 1; x < width - 1; x += 1) {
            const center = grayscale[y * width + x];
            const top = grayscale[(y - 1) * width + x];
            const bottom = grayscale[(y + 1) * width + x];
            const left = grayscale[y * width + (x - 1)];
            const right = grayscale[y * width + (x + 1)];
            laplacianValues.push(Math.abs(top + bottom + left + right - 4 * center));
        }
    }

    const mean = laplacianValues.reduce((sum, value) => sum + value, 0) / laplacianValues.length;
    const variance = laplacianValues.reduce((sum, value) => sum + (value - mean) ** 2, 0) / laplacianValues.length;
    const brightness = brightnessTotal / grayscale.length;

    if (brightness < 40) {
        return {
            label: "Poor lighting",
            detail: `Brightness: ${brightness.toFixed(1)}. Retake in brighter, even light.`,
            isGood: false
        };
    }

    if (variance < 100) {
        return {
            label: "Blurry image",
            detail: `Blur score: ${variance.toFixed(1)}. Hold the camera steady and refocus.`,
            isGood: false
        };
    }

    if (brightness > 240) {
        return {
            label: "Good quality",
            detail: `Brightness: ${brightness.toFixed(1)}. The image may be slightly overexposed.`,
            isGood: true
        };
    }

    return {
        label: "Good quality",
        detail: `Blur score: ${variance.toFixed(1)}. Image is suitable for screening.`,
        isGood: true
    };
}

function updateQualityStatus(analysis) {
    const qualityStatus = document.getElementById("qualityStatus");
    const qualityScore = document.getElementById("qualityScore");
    if (!qualityStatus || !qualityScore) return;

    qualityStatus.textContent = analysis.label;
    qualityStatus.classList.toggle("text-success", analysis.isGood);
    qualityStatus.classList.toggle("text-danger", !analysis.isGood);
    qualityScore.textContent = analysis.detail;
}

function showPreviewFromDataUrl(dataUrl) {
    const preview = document.getElementById("imagePreview");
    const placeholder = document.getElementById("previewPlaceholder");
    if (!preview || !placeholder) return;

    preview.src = dataUrl;
    preview.classList.remove("d-none");
    placeholder.classList.add("d-none");
    preview.onload = () => updateQualityStatus(analyzeImageQuality(preview));
}

function resetPreview() {
    const preview = document.getElementById("imagePreview");
    const placeholder = document.getElementById("previewPlaceholder");
    if (!preview || !placeholder) return;

    preview.src = "";
    preview.classList.add("d-none");
    placeholder.classList.remove("d-none");
}

function setCameraStatus(message, isError = false) {
    const status = document.getElementById("cameraStatus");
    if (!status) return;
    status.textContent = message;
    status.classList.toggle("text-danger", isError);
}

function isLocalHost() {
    return ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
}

function shouldWarnAboutHttps() {
    return !window.isSecureContext && !isLocalHost();
}

function toggleCameraWarning(show) {
    const banner = document.getElementById("cameraHttpWarning");
    if (!banner) return;
    banner.classList.toggle("d-none", !show);
}

function bindUploadZone() {
    const dropzone = document.getElementById("dropzone");
    const imageInput = document.getElementById("imageInput");
    const capturedInput = document.getElementById("capturedImageData");
    const uploadTabButton = document.getElementById("upload-tab");
    if (!dropzone || !imageInput) return;

    const readFile = (file) => {
        const reader = new FileReader();
        reader.onload = (event) => {
            if (capturedInput) capturedInput.value = "";
            showPreviewFromDataUrl(event.target?.result || "");
        };
        reader.readAsDataURL(file);
    };

    imageInput.addEventListener("change", () => {
        const file = imageInput.files?.[0];
        if (file) readFile(file);
    });

    ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (event) => {
        const file = event.dataTransfer?.files?.[0];
        if (file) {
            imageInput.files = event.dataTransfer.files;
            readFile(file);
        }
    });

    uploadTabButton?.addEventListener("click", () => toggleCameraWarning(false));
}

function bindCameraCapture() {
    const startButton = document.getElementById("startCameraButton");
    const captureButton = document.getElementById("captureButton");
    const retakeButton = document.getElementById("retakeButton");
    const confirmButton = document.getElementById("confirmCaptureButton");
    const video = document.getElementById("cameraFeed");
    const canvas = document.getElementById("cameraCanvas");
    const capturedInput = document.getElementById("capturedImageData");
    const imageInput = document.getElementById("imageInput");
    const uploadTabButton = document.getElementById("upload-tab");
    const cameraTabButton = document.getElementById("camera-tab");
    let stream = null;

    if (!startButton || !captureButton || !video || !canvas || !capturedInput) return;

    const toggleCaptureState = (hasCapture) => {
        retakeButton?.classList.toggle("d-none", !hasCapture);
        confirmButton?.classList.toggle("d-none", !hasCapture);
        captureButton.classList.toggle("d-none", hasCapture);
    };

    const stopStreamTracks = () => {
        if (!stream) return;
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
    };

    startButton.addEventListener("click", async () => {
        if (shouldWarnAboutHttps()) {
            toggleCameraWarning(true);
            setCameraStatus("Camera access needs HTTPS on mobile browsers. Use file upload or open the secure (https) site.", true);
            uploadTabButton?.click();
            return;
        }

        if (!navigator.mediaDevices?.getUserMedia) {
            setCameraStatus("Camera access is not available on this device. Please use file upload instead.", true);
            uploadTabButton?.click();
            return;
        }

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { ideal: "environment" },
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            });
            video.srcObject = stream;
            setCameraStatus("Camera ready. Frame the eye conjunctiva or fingernail clearly, then capture.");
        } catch (error) {
            setCameraStatus("Unable to access the camera. Please allow permission or switch to file upload.", true);
            uploadTabButton?.click();
        }
    });

    captureButton.addEventListener("click", () => {
        if (!video.videoWidth || !video.videoHeight) {
            setCameraStatus("Start the camera first, then capture the image.", true);
            return;
        }

        const context = canvas.getContext("2d");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
        capturedInput.value = dataUrl;
        if (imageInput) imageInput.value = "";
        showPreviewFromDataUrl(dataUrl);
        toggleCaptureState(true);
        setCameraStatus("Photo captured. Review the preview and retake if needed.");
    });

    retakeButton?.addEventListener("click", () => {
        capturedInput.value = "";
        toggleCaptureState(false);
        resetPreview();
        setCameraStatus("Retake the photo when the image is sharp and evenly lit.");
    });

    confirmButton?.addEventListener("click", () => {
        setCameraStatus("Captured image confirmed and ready for AI screening.");
    });

    window.addEventListener("beforeunload", stopStreamTracks);

    cameraTabButton?.addEventListener("click", () => {
        toggleCameraWarning(shouldWarnAboutHttps());
    });
}

function bindScanSubmit() {
    const form = document.getElementById("scanForm");
    const overlay = document.getElementById("loadingOverlay");
    const imageInput = document.getElementById("imageInput");
    const capturedInput = document.getElementById("capturedImageData");

    if (!form || !overlay) return;

    form.addEventListener("submit", (event) => {
        const hasUpload = Boolean(imageInput?.files?.length);
        const hasCapture = Boolean(capturedInput?.value);
        if (!hasUpload && !hasCapture) {
            event.preventDefault();
            alert("Please upload or capture an image before submitting.");
            return;
        }

        overlay.classList.add("show");
        overlay.setAttribute("aria-hidden", "false");
    });
}

document.addEventListener("DOMContentLoaded", () => {
    bindUploadZone();
    bindCameraCapture();
    bindScanSubmit();
});
