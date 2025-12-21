const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const btnStart = document.getElementById("btnStart");
const btnSnap = document.getElementById("btnSnap");
const btnSend = document.getElementById("btnSend");
const resultImg = document.getElementById("result");
const log = document.getElementById("log");
const health = document.getElementById("health");

const promptEl = document.getElementById("prompt");
const negativeEl = document.getElementById("negative");

let stream = null;

async function checkHealth() {
    try {
        const res = await fetch("/api/health");
        const data = await res.json();
        health.textContent = data.status;
    } catch {
        health.textContent = "failed";
    }
}

async function loadConfig() {
    try {
        const res = await fetch("/api/config");
        if (!res.ok) throw new Error("config fetch failed");
        const cfg = await res.json();

        if (promptEl && cfg.default_prompt) promptEl.value = cfg.default_prompt;
        if (negativeEl && cfg.default_negative_prompt) negativeEl.value = cfg.default_negative_prompt;

        log.textContent = "Config geladen. Du kannst starten.";
    } catch (e) {
        log.textContent = "Konnte Config nicht laden (fallback).";
    }
}

btnStart.addEventListener("click", async () => {
    log.textContent = "Starte Kamera...";
    try {
        // Wichtig: funktioniert nur auf https oder http://localhost / 127.0.0.1
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user" },
            audio: false
        });

        video.srcObject = stream;

        btnSnap.disabled = false;
        log.textContent = "Kamera läuft. Du kannst jetzt ein Foto aufnehmen.";
    } catch (e) {
        log.textContent = "Kamera-Fehler: " + e;
    }
});

btnSnap.addEventListener("click", () => {
    if (!video.videoWidth) {
        log.textContent = "Video noch nicht bereit.";
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    btnSend.disabled = false;
    log.textContent = "Snapshot aufgenommen. Du kannst es jetzt ans Backend senden.";
});

function canvasToBlob(canvasEl) {
    return new Promise((resolve) => {
        canvasEl.toBlob((blob) => resolve(blob), "image/png", 1.0);
    });
}

btnSend.addEventListener("click", async () => {
    log.textContent = "Sende Bild ans Backend...";
    resultImg.removeAttribute("src");

    try {
        const blob = await canvasToBlob(canvas);

        const form = new FormData();
        form.append("image", blob, "snapshot.png");
        form.append("prompt", promptEl ? promptEl.value : "");
        form.append("negative_prompt", negativeEl ? negativeEl.value : "");
        form.append("use_lora", "1"); // testweise "0" zum Vergleich

        const res = await fetch("/api/simpsonify", {
            method: "POST",
            body: form
        });

        if (!res.ok) {
            const txt = await res.text();
            throw new Error(`HTTP ${res.status}: ${txt}`);
        }

        const outBlob = await res.blob();
        const url = URL.createObjectURL(outBlob);
        resultImg.src = url;

        log.textContent = "Fertig. Ergebnisbild ist da.";
    } catch (e) {
        log.textContent = "Fehler: " + e;
    }
});

// init
loadConfig();
checkHealth();
