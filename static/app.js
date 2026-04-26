const form = document.querySelector("#uploadForm");
const input = document.querySelector("#imageInput");
const dropZone = document.querySelector(".drop-zone");
const filePreview = document.querySelector("#filePreview");
const previewImage = document.querySelector("#previewImage");
const previewName = document.querySelector("#previewName");
const previewSize = document.querySelector("#previewSize");
const toolStatus = document.querySelector("#toolStatus");
const emptyState = document.querySelector("#emptyState");
const loadingState = document.querySelector("#loadingState");
const resultView = document.querySelector("#resultView");
const verdictLabel = document.querySelector("#verdictLabel");
const verdictTitle = document.querySelector("#verdictTitle");
const confidenceValue = document.querySelector("#confidenceValue");
const confidenceBar = document.querySelector("#confidenceBar");
const shaValue = document.querySelector("#shaValue");
const typeValue = document.querySelector("#typeValue");
const dimensionValue = document.querySelector("#dimensionValue");
const elapsedValue = document.querySelector("#elapsedValue");
const evidenceList = document.querySelector("#evidenceList");
const warningList = document.querySelector("#warningList");
const exifBlock = document.querySelector("#exifBlock");
const visualBlock = document.querySelector("#visualBlock");
const modelBlock = document.querySelector("#modelBlock");
const toolBlock = document.querySelector("#toolBlock");
const rawJsonBlock = document.querySelector("#rawJsonBlock");
const copyJsonButton = document.querySelector("#copyJsonButton");

let latestReport = null;

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function setState(state) {
  emptyState.classList.toggle("hidden", state !== "empty");
  loadingState.classList.toggle("hidden", state !== "loading");
  resultView.classList.toggle("hidden", state !== "result");
  form.querySelector("button").disabled = state === "loading";
}

function renderList(target, items) {
  target.innerHTML = "";
  for (const item of items || []) {
    const li = document.createElement("li");
    li.textContent = item;
    target.appendChild(li);
  }
}

function stableJson(value) {
  return JSON.stringify(value ?? {}, null, 2);
}

function setVerdictTone(label) {
  verdictLabel.classList.remove("warn", "danger");
  confidenceBar.style.background = "var(--green)";
  if (label === "confirmed_ai_generated" || label === "likely_ai_generated") {
    verdictLabel.classList.add("danger");
    confidenceBar.style.background = "var(--red)";
  } else if (
    label === "possibly_ai_generated" ||
    label === "possibly_ai_edited" ||
    label === "inconclusive"
  ) {
    verdictLabel.classList.add("warn");
    confidenceBar.style.background = "var(--amber)";
  }
}

function renderReport(report) {
  latestReport = report;
  const verdict = report.verdict || {};
  const confidence = Math.round((verdict.confidence || 0) * 100);
  verdictLabel.textContent = verdict.label || "inconclusive";
  verdictTitle.textContent = verdict.title || "证据不足，无法判断";
  confidenceValue.textContent = `${confidence}%`;
  confidenceBar.style.width = `${confidence}%`;
  setVerdictTone(verdict.label);

  shaValue.textContent = report.sha256 || "-";
  typeValue.textContent = (report.image_type || "-").toUpperCase();
  dimensionValue.textContent = report.dimensions
    ? `${report.dimensions.width} x ${report.dimensions.height}`
    : "-";
  elapsedValue.textContent = `${report.elapsed_ms || 0} ms`;

  renderList(evidenceList, verdict.evidence || []);
  renderList(warningList, verdict.warnings || []);

  exifBlock.textContent = stableJson(report.metadata?.embedded_exif);
  visualBlock.textContent = stableJson(report.metadata?.visual_forensics);
  modelBlock.textContent = stableJson(report.model);
  toolBlock.textContent = stableJson(report.metadata?.tools);
  rawJsonBlock.textContent = stableJson(report);
  setState("result");
}

function showError(message) {
  latestReport = {
    error: message,
    created_at: new Date().toISOString(),
  };
  verdictLabel.textContent = "error";
  verdictTitle.textContent = message;
  confidenceValue.textContent = "0%";
  confidenceBar.style.width = "0%";
  verdictLabel.classList.add("danger");
  shaValue.textContent = "-";
  typeValue.textContent = "-";
  dimensionValue.textContent = "-";
  elapsedValue.textContent = "-";
  renderList(evidenceList, []);
  renderList(warningList, ["请确认上传的是受支持的图片文件。"]);
  exifBlock.textContent = "{}";
  visualBlock.textContent = "{}";
  modelBlock.textContent = "{}";
  toolBlock.textContent = "{}";
  rawJsonBlock.textContent = stableJson(latestReport);
  setState("result");
}

async function analyze(file) {
  const formData = new FormData();
  formData.append("image", file);
  setState("loading");
  const response = await fetch("/api/analyze", {
    method: "POST",
    body: formData,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "分析失败");
  }
  return payload;
}

function setPreview(file) {
  if (!file) {
    filePreview.classList.add("hidden");
    return;
  }
  previewName.textContent = file.name;
  previewSize.textContent = formatBytes(file.size);
  previewImage.src = URL.createObjectURL(file);
  filePreview.classList.remove("hidden");
}

input.addEventListener("change", () => {
  setPreview(input.files?.[0]);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = input.files?.[0];
  if (!file) {
    showError("请先选择一张图片。");
    return;
  }
  try {
    const report = await analyze(file);
    renderReport(report);
  } catch (error) {
    showError(error.message || "分析失败");
  }
});

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
}

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  input.files = event.dataTransfer.files;
  setPreview(file);
});

copyJsonButton.addEventListener("click", async () => {
  if (!latestReport) return;
  await navigator.clipboard.writeText(stableJson(latestReport));
  copyJsonButton.textContent = "已复制";
  setTimeout(() => {
    copyJsonButton.textContent = "复制 JSON";
  }, 1200);
});

async function refreshHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    const c2pa = data.tools?.c2patool ? "c2patool 可用" : "c2patool 未安装";
    const exif = data.tools?.exiftool ? "exiftool 可用" : "exiftool 未安装";
    const model = data.model?.exists ? "模型已加载" : "模型未配置";
    toolStatus.textContent = `${c2pa} · ${exif} · ${model}`;
  } catch {
    toolStatus.textContent = "本机服务未连接";
  }
}

refreshHealth();
