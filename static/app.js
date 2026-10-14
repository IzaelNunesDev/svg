/**
 * Anim8 Vector AI Studio - Frontend Application Controller
 * Handles image & video flipbook vectorization, interactive SVG canvas,
 * live color recoloring, layer hierarchy, and timeline playback.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ==========================================
  // STATE MANAGEMENT
  // ==========================================
  let currentMode = "image"; // 'image' or 'video'
  let currentFile = null;
  let currentImageObjectUrl = null;
  let currentResult = null; // Single image vectorization result
  let flipbookResult = null; // Video flipbook result: { cels, unique_colors, video_info, ... }
  let activeCelIndex = 1; // 1-based index
  let selectedLayerId = null;
  let selectedLayerColor = null;

  // Live edits state
  let recolorMap = {}; // { "#3D3D3D": "#5F0000" }
  let hiddenLayerIds = new Set();
  let hiddenColors = new Set();

  // Canvas Viewport & Zoom State
  let zoomLevel = 1.0;
  let currentViewMode = "vector"; // 'vector', 'original', 'split'
  let isPanning = false;
  let panStartX = 0, panStartY = 0;
  let panOffsetX = 0, panOffsetY = 0;

  // Video Flipbook Playback State
  let isPlaying = false;
  let playbackTimer = null;
  let playbackFps = 15;
  let isLooping = true;

  // ==========================================
  // DOM ELEMENTS
  // ==========================================
  // Header
  const btnModeImage = document.getElementById("btn-mode-image");
  const btnModeVideo = document.getElementById("btn-mode-video");
  const presetSelect = document.getElementById("preset-select");
  const btnVectorizeAction = document.getElementById("btn-vectorize-action");
  const btnVectorizeText = document.getElementById("btn-vectorize-text");
  const btnExportToggle = document.getElementById("btn-export-dropdown-toggle");
  const exportMenu = document.getElementById("export-menu");
  const dlSvgClean = document.getElementById("dl-svg-clean");
  const dlAllCelsZip = document.getElementById("dl-all-cels-zip");

  // Left Layers Panel
  const badgeLayersCount = document.getElementById("badge-layers-count");
  const activeCelTag = document.getElementById("active-cel-tag");
  const activeColorsStat = document.getElementById("active-colors-stat");
  const layersScrollList = document.getElementById("layers-scroll-list");
  const btnToggleAllVis = document.getElementById("btn-toggle-all-vis");
  const btnResetRecolors = document.getElementById("btn-reset-recolors");
  const layerGlobalActions = document.getElementById("layer-global-actions");
  const btnRemoveFromEveryFrame = document.getElementById("btn-remove-from-every-frame");

  // Canvas & Stage
  const canvasViewport = document.getElementById("canvas-viewport");
  const canvasTransformLayer = document.getElementById("canvas-transform-layer");
  const dropzoneBox = document.getElementById("dropzone-box");
  const fileInput = document.getElementById("file-input");
  const btnBrowseFile = document.getElementById("btn-browse-file");
  const vectorStageWrapper = document.getElementById("vector-stage-wrapper");
  const svgCanvasContainer = document.getElementById("svg-canvas-container");
  const imgOriginalCompare = document.getElementById("img-original-compare");

  // Floating Controls
  const btnZoomIn = document.getElementById("btn-zoom-in");
  const btnZoomOut = document.getElementById("btn-zoom-out");
  const btnZoomFit = document.getElementById("btn-zoom-fit");
  const zoomText = document.getElementById("zoom-text");
  const btnViewVector = document.getElementById("btn-view-vector");
  const btnViewOriginal = document.getElementById("btn-view-original");
  const btnViewSplit = document.getElementById("btn-view-split");

  // Timeline (Video Mode)
  const timelineFooter = document.getElementById("timeline-footer");
  const btnPlayPause = document.getElementById("btn-play-pause");
  const btnStepPrev = document.getElementById("btn-step-prev");
  const btnStepNext = document.getElementById("btn-step-next");
  const btnToggleLoop = document.getElementById("btn-toggle-loop");
  const celCurrentNum = document.getElementById("cel-current-num");
  const celTotalNum = document.getElementById("cel-total-num");
  const celTimeDisplay = document.getElementById("cel-time-display");
  const fpsSelect = document.getElementById("fps-select");
  const timelineScrubberTrack = document.getElementById("timeline-scrubber-track");
  const timelineTicks = document.getElementById("timeline-ticks");
  const timelinePlayhead = document.getElementById("timeline-playhead");

  // Right Inspector
  const propPointsCount = document.getElementById("prop-points-count");
  const propPos = document.getElementById("prop-pos");
  const propDim = document.getElementById("prop-dim");
  const propColorPicker = document.getElementById("prop-color-picker");
  const propColorHex = document.getElementById("prop-color-hex");
  const rangePrecision = document.getElementById("range-precision");
  const valPrecision = document.getElementById("val-precision");
  const rangeLayerDiff = document.getElementById("range-layer-diff");
  const valLayerDiff = document.getElementById("val-layer-diff");
  const rangeUnsharp = document.getElementById("range-unsharp");
  const valUnsharp = document.getElementById("val-unsharp");
  const rangeCorner = document.getElementById("range-corner");
  const valCorner = document.getElementById("val-corner");
  const rangeSmooth = document.getElementById("range-smooth");
  const valSmooth = document.getElementById("val-smooth");
  const rangeSpeckle = document.getElementById("range-speckle");
  const valSpeckle = document.getElementById("val-speckle");
  const rangeBilateral = document.getElementById("range-bilateral");
  const valBilateral = document.getElementById("val-bilateral");
  const btnRevectorize = document.getElementById("btn-revectorize");

  // Metrics
  const metricDuration = document.getElementById("metric-duration");
  const metricPathsCount = document.getElementById("metric-paths-count");
  const metricColorsCount = document.getElementById("metric-colors-count");

  // Spinner & Toast
  const processingOverlay = document.getElementById("processing-overlay");
  const spinnerTitle = document.getElementById("spinner-title");
  const spinnerSubtitle = document.getElementById("spinner-subtitle");
  const toastContainer = document.getElementById("toast-container");

  // ==========================================
  // TOAST NOTIFICATIONS
  // ==========================================
  function showToast(message, type = "info", duration = 3500) {
    if (!toastContainer) return;
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(8px)";
      setTimeout(() => toast.remove(), 250);
    }, duration);
  }

  function showSpinner(title, subtitle) {
    if (spinnerTitle) spinnerTitle.textContent = title;
    if (spinnerSubtitle) spinnerSubtitle.textContent = subtitle;
    if (processingOverlay) processingOverlay.classList.remove("hidden");
  }

  function hideSpinner() {
    if (processingOverlay) processingOverlay.classList.add("hidden");
  }

  // ==========================================
  // MODE SWITCHING (IMAGE vs VIDEO)
  // ==========================================
  btnModeImage.addEventListener("click", () => switchMode("image"));
  btnModeVideo.addEventListener("click", () => switchMode("video"));

  function switchMode(mode) {
    if (currentMode === mode) return;
    currentMode = mode;
    btnModeImage.classList.toggle("active", mode === "image");
    btnModeVideo.classList.toggle("active", mode === "video");

    // Reset playback if switching away from video
    if (mode === "image") {
      pausePlayback();
      timelineFooter.classList.add("hidden");
    } else {
      if (flipbookResult && flipbookResult.cels.length > 0) {
        timelineFooter.classList.remove("hidden");
      }
    }
    showToast(`Modo alterado para ${mode === "image" ? "Imagem Estática" : "Vídeo Flipbook"}`, "info", 1800);
  }

  // ==========================================
  // DRAG & DROP & FILE INGESTION
  // ==========================================
  if (btnBrowseFile && fileInput) {
    btnBrowseFile.addEventListener("click", (e) => {
      e.stopPropagation();
      fileInput.click();
    });
  }

  if (dropzoneBox && fileInput) {
    // Clique em qualquer área da dropzone abre o seletor de arquivos
    dropzoneBox.addEventListener("click", (e) => {
      if (e.target !== fileInput) {
        fileInput.click();
      }
    });

    dropzoneBox.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzoneBox.classList.add("dragover");
    });
    dropzoneBox.addEventListener("dragleave", () => {
      dropzoneBox.classList.remove("dragover");
    });
    dropzoneBox.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzoneBox.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelected(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelected(e.target.files[0]);
      }
      fileInput.value = ""; // Permite re-selecionar o mesmo arquivo se necessário
    });
  }

  function handleFileSelected(file) {
    currentFile = file;
    const isVideo = file.type.startsWith("video/") || file.name.endsWith(".mp4") || file.name.endsWith(".webm");

    // Automatically switch mode matching file type
    if (isVideo && currentMode !== "video") {
      switchMode("video");
    } else if (!isVideo && currentMode !== "image") {
      switchMode("image");
    }

    if (currentImageObjectUrl) {
      URL.revokeObjectURL(currentImageObjectUrl);
      currentImageObjectUrl = null;
    }
    if (!isVideo) {
      currentImageObjectUrl = URL.createObjectURL(file);
      if (imgOriginalCompare) {
        imgOriginalCompare.src = currentImageObjectUrl;
      }
    } else {
      if (imgOriginalCompare) {
        imgOriginalCompare.removeAttribute("src");
      }
    }

    showToast(`Arquivo selecionado: ${file.name}`, "info", 2000);

    // Auto-trigger vectorization
    runVectorization();
  }

  // ==========================================
  // VECTORIZATION DISPATCHER
  // ==========================================
  btnVectorizeAction.addEventListener("click", () => {
    if (!currentFile) {
      fileInput.click();
      return;
    }
    runVectorization();
  });

  btnRevectorize.addEventListener("click", () => {
    if (!currentFile) {
      showToast("Carregue uma imagem ou vídeo primeiro.", "warning");
      return;
    }
    runVectorization();
  });

  async function runVectorization() {
    if (!currentFile) return;

    // Reset edits
    recolorMap = {};
    hiddenLayerIds.clear();
    hiddenColors.clear();
    selectedLayerId = null;

    const preset = presetSelect ? presetSelect.value : "duolingo_flat";
    const customOpts = {
      color_precision: parseInt(rangePrecision.value),
      layer_difference: rangeLayerDiff ? parseInt(rangeLayerDiff.value) : 16,
      unsharp_strength: rangeUnsharp ? parseFloat(rangeUnsharp.value) : 0.6,
      corner_threshold: parseInt(rangeCorner.value),
      length_threshold: parseFloat(rangeSmooth.value),
      filter_speckle: parseInt(rangeSpeckle.value),
      bilateral_d: parseInt(rangeBilateral.value),
    };

    if (currentMode === "image") {
      await vectorizeImageFile(currentFile, preset, customOpts);
    } else {
      const targetFps = parseInt(fpsSelect.value) || 15;
      await vectorizeVideoFile(currentFile, preset, targetFps, customOpts);
    }
  }

  // Single Image Vectorization
  async function vectorizeImageFile(file, preset, customOpts) {
    showSpinner("Vetorizando com Curvas de Bézier...", "Eliminando ruído e calculando splines cúbicas");
    btnVectorizeAction.disabled = true;

    const formData = new FormData();
    formData.append("image", file);
    formData.append("preset", preset);
    for (const [k, v] of Object.entries(customOpts)) {
      formData.append(k, v);
    }

    try {
      const response = await fetch("/api/vectorize/image", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
        throw new Error(err.detail || "Falha na vetorização da imagem.");
      }

      const data = await response.json();
      currentResult = data;
      flipbookResult = null;
      activeCelIndex = 1;

      if (imgOriginalCompare) {
        if (data.original_image) {
          imgOriginalCompare.src = data.original_image;
        } else if (currentImageObjectUrl) {
          imgOriginalCompare.src = currentImageObjectUrl;
        }
      }

      // Render on Stage
      renderSvgOnStage(data.svg);
      renderLayersList(data.layers);
      applyViewMode();
      updateMetrics(data.duration_ms, data.metadata.total_paths, data.metadata.total_colors);

      dropzoneBox.classList.add("hidden");
      vectorStageWrapper.classList.remove("hidden");
      timelineFooter.classList.add("hidden");
      layerGlobalActions.classList.add("hidden");

      zoomFit();
      showToast(`Vetorizado com sucesso em ${data.duration_ms} ms! (${data.metadata.total_paths} caminhos)`, "success");

    } catch (err) {
      console.error("Erro vetorização:", err);
      showToast(`Erro ao vetorizar: ${err.message}`, "error", 6000);
    } finally {
      hideSpinner();
      btnVectorizeAction.disabled = false;
    }
  }

  // Video Flipbook Vectorization (MP4 -> Cels)
  async function vectorizeVideoFile(file, preset, targetFps, customOpts) {
    showSpinner("Processando Vídeo em Flipbook...", "Extraindo quadros e ajustando curvas de Bézier para cada cel");
    btnVectorizeAction.disabled = true;

    const formData = new FormData();
    formData.append("video", file);
    formData.append("preset", preset);
    formData.append("target_fps", targetFps);
    formData.append("max_frames", 45);
    for (const [k, v] of Object.entries(customOpts)) {
      formData.append(k, v);
    }

    try {
      const response = await fetch("/api/vectorize/video", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
        throw new Error(err.detail || "Falha ao processar o vídeo.");
      }

      const data = await response.json();
      flipbookResult = data;
      currentResult = null;
      activeCelIndex = 1;

      // Build Timeline
      buildTimelineScrubber(data.cels);
      timelineFooter.classList.remove("hidden");
      layerGlobalActions.classList.remove("hidden");

      // Render First Cel
      showCel(1);

      dropzoneBox.classList.add("hidden");
      vectorStageWrapper.classList.remove("hidden");

      updateMetrics(data.total_processing_ms, data.cels[0].total_paths, data.unique_colors.length);
      zoomFit();

      showToast(`Vídeo convertido em ${data.total_cels} cels vetoriais! (${data.avg_ms_per_cel} ms/cel)`, "success");

    } catch (err) {
      console.error("Erro vídeo:", err);
      showToast(`Erro ao processar vídeo: ${err.message}`, "error", 6000);
    } finally {
      hideSpinner();
      btnVectorizeAction.disabled = false;
    }
  }

  // ==========================================
  // SVG RENDERING & INTERACTIVE PATH SELECTION
  // ==========================================
  function renderSvgOnStage(svgString) {
    if (!svgCanvasContainer) return;
    svgCanvasContainer.innerHTML = svgString;

    // Sincroniza dimensões exatas da imagem original com a ViewBox do SVG
    const svgEl = svgCanvasContainer.querySelector("svg");
    if (svgEl && imgOriginalCompare) {
      const w = svgEl.getAttribute("width") || (svgEl.viewBox && svgEl.viewBox.baseVal ? svgEl.viewBox.baseVal.width : null);
      const h = svgEl.getAttribute("height") || (svgEl.viewBox && svgEl.viewBox.baseVal ? svgEl.viewBox.baseVal.height : null);
      if (w && h) {
        imgOriginalCompare.style.width = `${w}px`;
        imgOriginalCompare.style.height = `${h}px`;
        svgCanvasContainer.style.width = `${w}px`;
        svgCanvasContainer.style.height = `${h}px`;
      }
    }

    // Apply live recolor and visibility states to SVG paths
    applyStatesToDomSvg();

    // Attach click and hover handlers to individual vector paths
    const paths = svgCanvasContainer.querySelectorAll(".vector-path-element");
    paths.forEach((path) => {
      path.addEventListener("click", (e) => {
        e.stopPropagation();
        const layerId = path.getAttribute("data-layer-id");
        const color = path.getAttribute("data-color");
        selectLayer(layerId, color);
      });
    });

    // Deselect if clicking on SVG background
    if (svgEl) {
      svgEl.addEventListener("click", (e) => {
        if (e.target === svgEl) {
          deselectLayer();
        }
      });
    }
  }

  function applyStatesToDomSvg() {
    if (!svgCanvasContainer) return;
    const paths = svgCanvasContainer.querySelectorAll(".vector-path-element");
    paths.forEach((path) => {
      const layerId = path.getAttribute("data-layer-id");
      const origColor = path.getAttribute("data-color");

      // Check if hidden by layer ID or global color
      if (hiddenLayerIds.has(layerId) || hiddenColors.has(origColor)) {
        path.style.display = "none";
      } else {
        path.style.display = "";
      }

      // Check if recolored
      if (recolorMap[origColor]) {
        path.setAttribute("fill", recolorMap[origColor]);
      } else {
        path.setAttribute("fill", origColor);
      }

      // Selected state
      path.classList.toggle("selected", layerId === selectedLayerId);
    });
  }

  // ==========================================
  // LEFT PANEL: LAYERS LIST (Anim8 STYLE)
  // ==========================================
  function renderLayersList(layers) {
    if (!layersScrollList) return;
    layersScrollList.innerHTML = "";

    if (!layers || layers.length === 0) {
      layersScrollList.innerHTML = '<div class="layers-empty-state"><p>Nenhuma camada encontrada.</p></div>';
      if (badgeLayersCount) badgeLayersCount.textContent = "0";
      return;
    }

    if (badgeLayersCount) badgeLayersCount.textContent = layers.length;
    if (activeCelTag) activeCelTag.textContent = `Cel ${activeCelIndex} / ${flipbookResult ? flipbookResult.total_cels : 1}`;

    const uniqueColorsInCel = new Set(layers.map((l) => l.color_hex));
    if (activeColorsStat) activeColorsStat.textContent = `${uniqueColorsInCel.size} cores`;

    layers.forEach((layer) => {
      const isSelected = layer.id === selectedLayerId;
      const isHidden = hiddenLayerIds.has(layer.id) || hiddenColors.has(layer.color_hex);
      const currentColor = recolorMap[layer.color_hex] || layer.color_hex;

      const row = document.createElement("div");
      row.className = `layer-row ${isSelected ? "active" : ""}`;
      row.dataset.layerId = layer.id;
      row.dataset.color = layer.color_hex;

      row.innerHTML = `
        <div class="layer-swatch" style="background-color: ${currentColor};"></div>
        <div class="layer-info">
          <span class="layer-name-text">Frame ${layer.cel_index || 1} - ${currentColor}</span>
          <span class="layer-points-tag">${layer.points_count || 1} pontos</span>
        </div>
        <div class="layer-actions-cell">
          <button class="btn-layer-tool btn-vis" title="Ocultar/Exibir">${isHidden ? "🚫" : "👁️"}</button>
          <button class="btn-layer-tool btn-del" title="Excluir">✕</button>
        </div>
      `;

      // Click to select
      row.addEventListener("click", (e) => {
        if (e.target.closest(".btn-layer-tool")) return;
        selectLayer(layer.id, layer.color_hex);
      });

      // Toggle visibility
      const visBtn = row.querySelector(".btn-vis");
      visBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleLayerVisibility(layer.id);
      });

      // Delete / Hide layer
      const delBtn = row.querySelector(".btn-del");
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        hideLayer(layer.id);
      });

      layersScrollList.appendChild(row);
    });
  }

  function drawSelectionBoundingBox(bboxStr) {
    if (!svgCanvasContainer) return;
    const svgEl = svgCanvasContainer.querySelector("svg");
    if (!svgEl) return;

    removeSelectionBoundingBox();
    if (!bboxStr) return;

    const parts = bboxStr.split(",").map(Number);
    if (parts.length < 4 || parts.some(isNaN)) return;
    const [x, y, w, h] = parts;
    if (w <= 0 || h <= 0) return;

    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.id = "anim8-selection-box";
    g.setAttribute("class", "anim8-selection-box-group");

    // Bounding Box Rect
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", x);
    rect.setAttribute("y", y);
    rect.setAttribute("width", w);
    rect.setAttribute("height", h);
    rect.setAttribute("class", "anim8-bbox-rect");
    g.appendChild(rect);

    // 8 transform handles (Anim8 style)
    const handlePositions = [
      [x, y], [x + w / 2, y], [x + w, y],
      [x + w, y + h / 2], [x + w, y + h],
      [x + w / 2, y + h], [x, y + h], [x, y + h / 2]
    ];
    handlePositions.forEach(([hx, hy]) => {
      const handle = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      handle.setAttribute("x", hx - 3.5);
      handle.setAttribute("y", hy - 3.5);
      handle.setAttribute("width", 7);
      handle.setAttribute("height", 7);
      handle.setAttribute("class", "anim8-bbox-handle");
      g.appendChild(handle);
    });

    svgEl.appendChild(g);
  }

  function removeSelectionBoundingBox() {
    if (!svgCanvasContainer) return;
    const existing = svgCanvasContainer.querySelector("#anim8-selection-box");
    if (existing) existing.remove();
  }

  function selectLayer(layerId, colorHex) {
    selectedLayerId = layerId;
    selectedLayerColor = colorHex;

    // Update layer list visual
    const rows = layersScrollList.querySelectorAll(".layer-row");
    rows.forEach((r) => {
      const match = r.dataset.layerId === layerId;
      r.classList.toggle("active", match);
      if (match) r.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });

    // Update SVG selection highlight
    applyStatesToDomSvg();

    // Update Right Inspector
    const path = svgCanvasContainer.querySelector(`[data-layer-id="${layerId}"]`);
    if (path) {
      const pts = path.getAttribute("data-points") || "--";
      if (propPointsCount) propPointsCount.textContent = `${pts} pontos`;

      const bboxStr = path.getAttribute("data-bbox");
      if (bboxStr) {
        const [x, y, w, h] = bboxStr.split(",").map((v) => Math.round(Number(v) * 10) / 10);
        if (propPos) propPos.textContent = `X: ${x}, Y: ${y}`;
        if (propDim) propDim.textContent = `${w} × ${h} px`;
        drawSelectionBoundingBox(bboxStr);
      } else {
        if (propPos) propPos.textContent = "--";
        if (propDim) propDim.textContent = "--";
        removeSelectionBoundingBox();
      }
    }

    const effectiveColor = recolorMap[colorHex] || colorHex;
    if (propColorPicker) propColorPicker.value = effectiveColor;
    if (propColorHex) propColorHex.value = effectiveColor;
  }

  function deselectLayer() {
    selectedLayerId = null;
    selectedLayerColor = null;
    const rows = layersScrollList.querySelectorAll(".layer-row");
    rows.forEach((r) => r.classList.remove("active"));
    applyStatesToDomSvg();
    removeSelectionBoundingBox();
    if (propPointsCount) propPointsCount.textContent = "-- pontos";
    if (propPos) propPos.textContent = "--";
    if (propDim) propDim.textContent = "--";
  }

  function toggleLayerVisibility(layerId) {
    if (hiddenLayerIds.has(layerId)) {
      hiddenLayerIds.delete(layerId);
    } else {
      hiddenLayerIds.add(layerId);
    }
    applyStatesToDomSvg();
    // Re-render layer row
    const row = layersScrollList.querySelector(`[data-layer-id="${layerId}"]`);
    if (row) {
      const visBtn = row.querySelector(".btn-vis");
      if (visBtn) visBtn.textContent = hiddenLayerIds.has(layerId) ? "🚫" : "👁️";
    }
  }

  function hideLayer(layerId) {
    hiddenLayerIds.add(layerId);
    applyStatesToDomSvg();
    const row = layersScrollList.querySelector(`[data-layer-id="${layerId}"]`);
    if (row) {
      const visBtn = row.querySelector(".btn-vis");
      if (visBtn) visBtn.textContent = "🚫";
    }
  }

  // Toggle all visibility
  btnToggleAllVis.addEventListener("click", () => {
    const paths = svgCanvasContainer.querySelectorAll(".vector-path-element");
    const anyVisible = paths.length > hiddenLayerIds.size;
    if (anyVisible) {
      paths.forEach((p) => hiddenLayerIds.add(p.getAttribute("data-layer-id")));
    } else {
      hiddenLayerIds.clear();
      hiddenColors.clear();
    }
    applyStatesToDomSvg();
    // Refresh layers UI
    const currentCel = getCurrentCelData();
    if (currentCel) renderLayersList(currentCel.layers);
  });

  // Reset recolorings
  btnResetRecolors.addEventListener("click", () => {
    recolorMap = {};
    applyStatesToDomSvg();
    const currentCel = getCurrentCelData();
    if (currentCel) renderLayersList(currentCel.layers);
    showToast("Cores restauradas para os tons originais.", "info", 2000);
  });

  // Remove selected color from every frame (Anim8 feature)
  btnRemoveFromEveryFrame.addEventListener("click", () => {
    if (!selectedLayerColor) {
      showToast("Selecione uma forma primeiro para remover sua cor.", "warning");
      return;
    }
    hiddenColors.add(selectedLayerColor);
    applyStatesToDomSvg();
    const currentCel = getCurrentCelData();
    if (currentCel) renderLayersList(currentCel.layers);
    showToast(`Cor ${selectedLayerColor} removida de todos os ${flipbookResult.total_cels} cels!`, "success", 3000);
  });

  // ==========================================
  // LIVE COLOR RECOLORING (INSPECTOR)
  // ==========================================
  if (propColorPicker && propColorHex) {
    propColorPicker.addEventListener("input", (e) => {
      const newCol = e.target.value.toUpperCase();
      propColorHex.value = newCol;
      applyLiveColorChange(newCol);
    });

    propColorHex.addEventListener("change", (e) => {
      let val = e.target.value.trim().toUpperCase();
      if (!val.startsWith("#")) val = "#" + val;
      if (/^#[0-9A-F]{6}$/i.test(val)) {
        propColorPicker.value = val;
        applyLiveColorChange(val);
      }
    });
  }

  function applyLiveColorChange(newHex) {
    if (!selectedLayerColor) return;
    recolorMap[selectedLayerColor] = newHex;

    // Instantly update paths in the DOM SVG
    applyStatesToDomSvg();

    // Update swatch in layers list
    const matchingRows = layersScrollList.querySelectorAll(`[data-color="${selectedLayerColor}"]`);
    matchingRows.forEach((row) => {
      const swatch = row.querySelector(".layer-swatch");
      if (swatch) swatch.style.backgroundColor = newHex;
      const text = row.querySelector(".layer-name-text");
      if (text) {
        const celNum = row.dataset.layerId.split("_")[0].replace("cel", "");
        text.textContent = `Frame ${celNum} - ${newHex}`;
      }
    });
  }

  // ==========================================
  // TIMELINE & FLIPBOOK PLAYBACK (VIDEO MODE)
  // ==========================================
  function buildTimelineScrubber(cels) {
    if (!timelineTicks) return;
    timelineTicks.innerHTML = "";

    cels.forEach((cel, idx) => {
      const block = document.createElement("div");
      block.className = `timeline-cel-block ${idx === 0 ? "active" : ""}`;
      block.dataset.celIndex = cel.cel_index;
      block.textContent = cel.cel_index;

      block.addEventListener("click", () => {
        showCel(cel.cel_index);
      });

      timelineTicks.appendChild(block);
    });

    if (celTotalNum) celTotalNum.textContent = cels.length;
  }

  function showCel(celIndex) {
    if (!flipbookResult || !flipbookResult.cels) return;
    const cel = flipbookResult.cels.find((c) => c.cel_index === celIndex);
    if (!cel) return;

    activeCelIndex = celIndex;

    // Atualiza frame original exato correspondente a este cel
    if (imgOriginalCompare && cel.original_frame) {
      imgOriginalCompare.src = cel.original_frame;
    }

    // Render Cel's SVG
    renderSvgOnStage(cel.svg);
    renderLayersList(cel.layers);

    // Reaplica o modo de visualização atual (Vetor / Original / Comparar)
    applyViewMode();

    // Update Indicator
    if (celCurrentNum) celCurrentNum.textContent = `Cel ${celIndex}`;
    if (celTimeDisplay) {
      celTimeDisplay.textContent = `(${cel.timestamp.toFixed(3)}s - ${playbackFps} fps)`;
    }

    // Update Timeline Playhead
    const totalCels = flipbookResult.cels.length;
    const pct = ((celIndex - 1) / Math.max(1, totalCels - 1)) * 100;
    if (timelinePlayhead) timelinePlayhead.style.left = `${pct}%`;

    // Update tick active state
    const blocks = timelineTicks.querySelectorAll(".timeline-cel-block");
    blocks.forEach((b) => {
      b.classList.toggle("active", parseInt(b.dataset.celIndex) === celIndex);
    });
  }

  function getCurrentCelData() {
    if (currentMode === "video" && flipbookResult) {
      return flipbookResult.cels.find((c) => c.cel_index === activeCelIndex);
    }
    return currentResult;
  }

  // Play / Pause Controls
  btnPlayPause.addEventListener("click", () => {
    if (isPlaying) {
      pausePlayback();
    } else {
      startPlayback();
    }
  });

  function startPlayback() {
    if (!flipbookResult || flipbookResult.cels.length <= 1) return;
    isPlaying = true;
    btnPlayPause.innerHTML = `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>`;

    const intervalMs = Math.round(1000 / playbackFps);
    playbackTimer = setInterval(() => {
      let nextIndex = activeCelIndex + 1;
      if (nextIndex > flipbookResult.cels.length) {
        if (isLooping) {
          nextIndex = 1;
        } else {
          pausePlayback();
          return;
        }
      }
      showCel(nextIndex);
    }, intervalMs);
  }

  function pausePlayback() {
    isPlaying = false;
    if (playbackTimer) {
      clearInterval(playbackTimer);
      playbackTimer = null;
    }
    btnPlayPause.innerHTML = `<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>`;
  }

  btnStepPrev.addEventListener("click", () => {
    pausePlayback();
    if (activeCelIndex > 1) showCel(activeCelIndex - 1);
  });

  btnStepNext.addEventListener("click", () => {
    pausePlayback();
    if (flipbookResult && activeCelIndex < flipbookResult.cels.length) {
      showCel(activeCelIndex + 1);
    }
  });

  btnToggleLoop.addEventListener("click", () => {
    isLooping = !isLooping;
    btnToggleLoop.classList.toggle("active", isLooping);
  });

  fpsSelect.addEventListener("change", (e) => {
    playbackFps = parseInt(e.target.value) || 15;
    if (isPlaying) {
      pausePlayback();
      startPlayback();
    }
  });

  // Scrubber drag / click
  timelineScrubberTrack.addEventListener("click", (e) => {
    if (!flipbookResult || !flipbookResult.cels) return;
    const rect = timelineScrubberTrack.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    const targetCel = Math.round(ratio * (flipbookResult.cels.length - 1)) + 1;
    showCel(targetCel);
  });

  // ==========================================
  // ZOOM, PAN & VIEW MODES
  // ==========================================
  function updateTransform() {
    if (!canvasTransformLayer) return;
    canvasTransformLayer.style.transform = `translate(${panOffsetX}px, ${panOffsetY}px) scale(${zoomLevel})`;
    if (zoomText) zoomText.textContent = `${Math.round(zoomLevel * 100)}%`;
  }

  function zoomFit() {
    zoomLevel = 1.0;
    panOffsetX = 0;
    panOffsetY = 0;
    updateTransform();
  }

  btnZoomIn.addEventListener("click", () => {
    zoomLevel = Math.min(5.0, zoomLevel + 0.2);
    updateTransform();
  });

  btnZoomOut.addEventListener("click", () => {
    zoomLevel = Math.max(0.2, zoomLevel - 0.2);
    updateTransform();
  });

  btnZoomFit.addEventListener("click", zoomFit);

  // Mouse wheel zoom
  canvasViewport.addEventListener("wheel", (e) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.1 : -0.1;
    zoomLevel = Math.max(0.2, Math.min(5.0, zoomLevel + delta));
    updateTransform();
  }, { passive: false });

  // View modes (Vector, Original, Split)
  btnViewVector.addEventListener("click", () => setViewMode("vector"));
  btnViewOriginal.addEventListener("click", () => setViewMode("original"));
  btnViewSplit.addEventListener("click", () => setViewMode("split"));

  function setViewMode(mode) {
    currentViewMode = mode;
    btnViewVector.classList.toggle("active", mode === "vector");
    btnViewOriginal.classList.toggle("active", mode === "original");
    btnViewSplit.classList.toggle("active", mode === "split");
    applyViewMode();
  }

  function applyViewMode() {
    if (!imgOriginalCompare || !svgCanvasContainer) return;

    // Sincroniza source da imagem original correspondente ao cel/frame atual
    if (currentMode === "video" && flipbookResult) {
      const cel = getCurrentCelData();
      if (cel && cel.original_frame && imgOriginalCompare.src !== cel.original_frame) {
        imgOriginalCompare.src = cel.original_frame;
      }
    } else if (currentResult && currentResult.original_image) {
      if (imgOriginalCompare.src !== currentResult.original_image) {
        imgOriginalCompare.src = currentResult.original_image;
      }
    }

    if (currentViewMode === "vector") {
      imgOriginalCompare.classList.add("hidden");
      imgOriginalCompare.style.opacity = "0";
      svgCanvasContainer.style.opacity = "1";
      svgCanvasContainer.style.pointerEvents = "auto";
    } else if (currentViewMode === "original") {
      imgOriginalCompare.classList.remove("hidden");
      imgOriginalCompare.style.opacity = "1";
      svgCanvasContainer.style.opacity = "0";
      svgCanvasContainer.style.pointerEvents = "none";
    } else {
      // Modo Comparar (Split / Overlay): imagem original embaixo e vetor semi-transparente em cima
      imgOriginalCompare.classList.remove("hidden");
      imgOriginalCompare.style.opacity = "1";
      svgCanvasContainer.style.opacity = "0.55";
      svgCanvasContainer.style.pointerEvents = "auto";
    }
  }

  // ==========================================
  // EXPORT HANDLERS (SVG & ZIP)
  // ==========================================
  btnExportToggle.addEventListener("click", (e) => {
    exportMenu.classList.toggle("hidden");
    e.stopPropagation();
  });

  window.addEventListener("click", () => {
    if (exportMenu) exportMenu.classList.add("hidden");
  });

  dlSvgClean.addEventListener("click", async () => {
    const currentCel = getCurrentCelData();
    if (!currentCel || !currentCel.svg) {
      showToast("Nenhum arquivo vetorizado disponível para download.", "warning");
      return;
    }

    try {
      const response = await fetch("/api/export/svg", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          svg: currentCel.svg,
          recolors: recolorMap,
          filename: `anim8_vector_cel_${activeCelIndex}.svg`
        }),
      });

      if (!response.ok) throw new Error("Erro ao gerar SVG.");
      const blob = await response.blob();
      downloadBlob(blob, `anim8_vector_cel_${activeCelIndex}.svg`);
      showToast("Download do SVG concluído!", "success");
    } catch (err) {
      showToast(`Falha no download: ${err.message}`, "error");
    }
  });

  dlAllCelsZip.addEventListener("click", async () => {
    if (!flipbookResult || !flipbookResult.cels) {
      // If single image mode, export single SVG as fallback
      dlSvgClean.click();
      return;
    }

    showToast("Gerando arquivo ZIP com todos os cels e SVG animado...", "info", 3000);
    try {
      const response = await fetch("/api/export/zip", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cels: flipbookResult.cels,
          fps: playbackFps,
          recolors: recolorMap,
        }),
      });

      if (!response.ok) throw new Error("Erro ao gerar ZIP.");
      const blob = await response.blob();
      downloadBlob(blob, "anim8_flipbook_vector_cels.zip");
      showToast("Download do pacote ZIP concluído!", "success");
    } catch (err) {
      showToast(`Falha ao exportar ZIP: ${err.message}`, "error");
    }
  });

  function downloadBlob(blob, filename) {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  }

  // ==========================================
  // SLIDER VALUE UPDATERS & METRICS
  // ==========================================
  rangePrecision.addEventListener("input", (e) => valPrecision.textContent = e.target.value);
  if (rangeLayerDiff) rangeLayerDiff.addEventListener("input", (e) => valLayerDiff.textContent = e.target.value);
  if (rangeUnsharp) rangeUnsharp.addEventListener("input", (e) => valUnsharp.textContent = parseFloat(e.target.value).toFixed(1));
  rangeCorner.addEventListener("input", (e) => valCorner.textContent = `${e.target.value}°`);
  rangeSmooth.addEventListener("input", (e) => valSmooth.textContent = parseFloat(e.target.value).toFixed(1));
  rangeSpeckle.addEventListener("input", (e) => valSpeckle.textContent = `${e.target.value} px`);
  rangeBilateral.addEventListener("input", (e) => valBilateral.textContent = `${e.target.value} px`);

  // Presets Auto-sync to Sliders
  let cachedPresets = null;
  async function loadPresetsCatalog() {
    try {
      const res = await fetch("/api/presets");
      if (res.ok) {
        const data = await res.json();
        cachedPresets = data.presets;
      }
    } catch (e) {
      console.warn("Falha ao carregar catálogo de presets:", e);
    }
  }
  loadPresetsCatalog();

  if (presetSelect) {
    presetSelect.addEventListener("change", () => {
      const selected = presetSelect.value;
      if (cachedPresets && cachedPresets[selected]) {
        const p = cachedPresets[selected];
        if (rangePrecision && p.color_precision !== undefined) {
          rangePrecision.value = p.color_precision;
          valPrecision.textContent = p.color_precision;
        }
        if (rangeLayerDiff && p.layer_difference !== undefined) {
          rangeLayerDiff.value = p.layer_difference;
          valLayerDiff.textContent = p.layer_difference;
        }
        if (rangeUnsharp && p.unsharp_strength !== undefined) {
          rangeUnsharp.value = p.unsharp_strength;
          valUnsharp.textContent = parseFloat(p.unsharp_strength).toFixed(1);
        }
        if (rangeCorner && p.corner_threshold !== undefined) {
          rangeCorner.value = p.corner_threshold;
          valCorner.textContent = `${p.corner_threshold}°`;
        }
        if (rangeSmooth && p.length_threshold !== undefined) {
          rangeSmooth.value = p.length_threshold;
          valSmooth.textContent = parseFloat(p.length_threshold).toFixed(1);
        }
        if (rangeSpeckle && p.filter_speckle !== undefined) {
          rangeSpeckle.value = p.filter_speckle;
          valSpeckle.textContent = `${p.filter_speckle} px`;
        }
        if (rangeBilateral && p.bilateral_d !== undefined) {
          rangeBilateral.value = p.bilateral_d;
          valBilateral.textContent = `${p.bilateral_d} px`;
        }
      }
    });
  }

  function updateMetrics(durationMs, pathsCount, colorsCount) {
    if (metricDuration) metricDuration.textContent = `${durationMs || 0} ms`;
    if (metricPathsCount) metricPathsCount.textContent = `${pathsCount || 0} caminhos`;
    if (metricColorsCount) metricColorsCount.textContent = `${colorsCount || 0} cores`;
  }
});