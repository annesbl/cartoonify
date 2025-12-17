const apiUrl = "http://localhost:8000/simpsonify";

const el = (id) => document.getElementById(id);

const file = el("file");
const preview = el("preview");
const result = el("result");
const status = el("status");
const download = el("download");

const strength = el("strength");
const steps = el("steps");
const guidance = el("guidance");
const lora = el("lora");
const seed = el("seed");
const prompt = el("prompt");
const neg = el("neg");

el("strengthVal").textContent = strength.value;
el("stepsVal").textContent = steps.value;
el("guidanceVal").textContent = guidance.value;
el("loraVal").textContent = lora.value;

[strength, steps, guidance, lora].forEach((r) =>
    r.addEventListener("input", () => {
        el("strengthVal").textContent = strength.value;
        el("stepsVal").textContent = steps.value;
        el("guidanceVal").textContent = guidance.value;
        el("loraVal").textContent = lora.value;
    })
);

// Tabs
const tabUpload = el("tabUpload");
const tabCamera = el("tabCamera");
const uploadPane = el("uploadPane");
const cameraPane = el("cameraPane");

function setTab(which) {
    const isUpload = which === "upload";
    tabUpload.classList.toggle("active", isUpload);
    tabCamera.classList.toggle("active", !isUpload);
    uploadPane.classList.toggle("active", isUpload);
    cameraPane.classList.toggle("active", !isUpload);
}
tabUpload.onclick = () => setTab("upload");
tabCamera.onclick = () => setTab("camera");

// Camera
const video = el("video");
const canvas = el("canvas");
const ctx = canvas.getContext("2d");
let capturedBlob = null;

el("startCam").onclick = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    status.textContent = "Kamera läuft.";
};

el("snap").onclick = () => {
    canvas.style.display = "block";
    // simple square crop from center
    const w = video.videoWidth, h = video.videoHeight;
    const s = Math.min(w, h);
    const sx = (w - s) / 2, sy = (h - s) / 2;
    canvas.width = 768; canvas.height = 768;
    ctx.drawImage(video, sx, sy, s, s, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
        capturedBlob = blob;
        preview.src = URL.createObjectURL(blob);
        status.textContent = "Foto aufgenommen.";
    }, "image/png");
};

// Upload preview
file.onchange = () => {
    const f = file.files?.[0];
    if (!f) return;
    preview.src = URL.createObjectURL(f);
    capturedBlob = null; // upload takes precedence
};

el("go").onclick = async () => {
    status.textContent = "Generiere… (Backend muss laufen)";
    result.src = "";

    const form = new FormData();

    // Choose input source
    if (capturedBlob) {
        form.append("image", capturedBlob, "capture.png");
    } else {
        const f = file.files?.[0];
        if (!f) {
            status.textContent = "Bitte Bild hochladen oder Foto aufnehmen.";
            return;
        }
        form.append("image", f);
    }

    form.append("strength", strength.value);
    form.append("steps", steps.value);
    form.append("guidance", guidance.value);
    form.append("lora_scale", lora.value);
    if (seed.value) form.append("seed", seed.value);

    if (prompt.value.trim()) form.append("prompt", prompt.value.trim());
    if (neg.value.trim()) form.append("negative_prompt", neg.value.trim());

    const res = await fetch(apiUrl, { method: "POST", body: form });
    if (!res.ok) {
        status.textContent = `Fehler: ${res.status} ${await res.text()}`;
        return;
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    result.src = url;
    download.href = url;
    status.textContent = "Fertig.";
};
