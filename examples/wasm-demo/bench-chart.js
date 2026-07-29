/*
 * bench-chart.js — hand-rolled canvas plotting for the benchmark page.
 *
 * No charting library, no CDN: the demo ships as plain static files and adding
 * a script tag pointing at someone else's origin would be the only external
 * dependency on the whole site.
 *
 * Three views over the same rows:
 *
 *   time        log-log wall time per transform against n. Straight-ish.
 *   normalized  LINEAR ns / (n · log2 n). This is the one worth looking at:
 *               a textbook O(n log n) implementation is a flat line here, so
 *               every rise is a real cost the asymptotics hide — cache
 *               capacity misses, a strategy switch, a mixed-radix fallback.
 *               A log axis would flatten exactly the bumps we want to see.
 *   throughput  MB/s, parameterized by precision (see bytesPerElement).
 *
 * Series are distinguished by colour AND marker shape AND dash pattern. Colour
 * alone excludes roughly one man in twelve, and these series are frequently
 * one strategy apart, which is precisely when a reader most needs to tell them
 * apart. Colours come from the page's CSS custom properties so the chart and
 * the rest of the site cannot drift.
 */

(() => {
  "use strict";

  const MARKERS = ["circle", "square", "triangle", "diamond", "cross"];
  const DASHES = [[], [6, 4], [2, 3], [10, 4, 2, 4], [1, 4]];
  const RAMP_VARS = ["--phase-0", "--phase-180", "--phase-90", "--phase-270"];

  const PAD_TOP = 18;
  const PAD_RIGHT = 18;
  const PAD_BOTTOM = 34;
  const GUTTER_SLACK = 14;
  const HIT_RADIUS = 14;

  /**
   * Bytes moved per element, per precision.
   *
   * complex64 is 8 bytes (two float32), complex128 is 16. The transform is
   * out-of-place, so a pass reads n elements and writes n: 2 * n * bytes.
   * The previous code hardcoded `size * 16`, which is the correct complex64
   * figure and silently half the truth for complex128 — a constant that
   * happens to be right is the worst kind of bug to inherit.
   */
  function bytesPerElement(precision) {
    return precision === "complex128" ? 16 : 8;
  }

  function throughputMBs(row) {
    if (!row || !(row.medianNs > 0)) return 0;
    const bytesPerOp = 2 * row.size * bytesPerElement(row.precision);
    return (bytesPerOp * (1e9 / row.medianNs)) / 1e6;
  }

  function normalizedNs(row) {
    if (!row || !(row.medianNs > 0) || row.size < 2) return 0;
    return row.medianNs / (row.size * Math.log2(row.size));
  }

  function cssVar(name, fallback) {
    const value = getComputedStyle(document.documentElement)
      .getPropertyValue(name)
      .trim();
    return value || fallback;
  }

  function formatTimeNs(ns) {
    if (!(ns > 0)) return "0";
    if (ns < 1e3) return `${trim(ns)} ns`;
    if (ns < 1e6) return `${trim(ns / 1e3)} µs`;
    return `${trim(ns / 1e6)} ms`;
  }

  function trim(value) {
    if (value >= 100) return value.toFixed(0);
    if (value >= 10) return value.toFixed(1);
    return value.toFixed(2);
  }

  function formatSize(n) {
    if (n >= 1048576 && n % 1048576 === 0) return `${n / 1048576}M`;
    if (n >= 1024 && n % 1024 === 0) return `${n / 1024}k`;
    return String(n);
  }

  function niceLogTicks(lo, hi) {
    // One tick per decade, plus 2x and 5x subdivisions when the range is short.
    const ticks = [];
    const decLo = Math.floor(Math.log10(lo));
    const decHi = Math.ceil(Math.log10(hi));
    const dense = decHi - decLo <= 3;
    for (let d = decLo; d <= decHi; d += 1) {
      const base = Math.pow(10, d);
      const mults = dense ? [1, 2, 5] : [1];
      for (const m of mults) {
        const v = base * m;
        if (v >= lo * 0.999 && v <= hi * 1.001) ticks.push(v);
      }
    }
    return ticks.length ? ticks : [lo, hi];
  }

  function niceLinearTicks(lo, hi, count) {
    const span = hi - lo;
    if (!(span > 0)) return [lo];
    const raw = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
    const start = Math.ceil(lo / step) * step;
    const ticks = [];
    for (let v = start; v <= hi * 1.0001; v += step) ticks.push(v);
    return ticks.length ? ticks : [lo, hi];
  }

  function drawMarker(ctx, shape, x, y, r) {
    ctx.beginPath();
    switch (shape) {
      case "square":
        ctx.rect(x - r, y - r, r * 2, r * 2);
        break;
      case "triangle":
        ctx.moveTo(x, y - r * 1.15);
        ctx.lineTo(x + r, y + r * 0.85);
        ctx.lineTo(x - r, y + r * 0.85);
        ctx.closePath();
        break;
      case "diamond":
        ctx.moveTo(x, y - r * 1.2);
        ctx.lineTo(x + r * 1.2, y);
        ctx.lineTo(x, y + r * 1.2);
        ctx.lineTo(x - r * 1.2, y);
        ctx.closePath();
        break;
      case "cross":
        ctx.moveTo(x - r, y - r);
        ctx.lineTo(x + r, y + r);
        ctx.moveTo(x + r, y - r);
        ctx.lineTo(x - r, y + r);
        break;
      default:
        ctx.arc(x, y, r, 0, Math.PI * 2);
        break;
    }
  }

  /**
   * @param {{canvas: HTMLCanvasElement, legend: HTMLElement,
   *          tooltip: HTMLElement, modeButtons: NodeListOf<HTMLButtonElement>}} opts
   */
  function createBenchChart(opts) {
    const canvas = opts.canvas;
    const ctx = canvas.getContext("2d");
    const legendEl = opts.legend;
    const tooltipEl = opts.tooltip;

    let rows = [];
    let mode = "time";
    let series = [];
    /** Screen-space points of the last paint, for hit testing. */
    let hits = [];
    let cssW = 0;
    let cssH = 0;

    function seriesKey(row) {
      return `${row.precision} · ${row.strategyRequested}`;
    }

    function rebuildSeries() {
      const byKey = new Map();
      for (const row of rows) {
        if (row.error) continue;
        const key = seriesKey(row);
        if (!byKey.has(key)) {
          byKey.set(key, { key, rows: [], visible: true });
        }
        byKey.get(key).rows.push(row);
      }

      const previous = new Map(series.map((s) => [s.key, s.visible]));
      series = [...byKey.values()].map((s, i) => {
        s.rows.sort((a, b) => a.size - b.size);
        s.color = cssVar(RAMP_VARS[i % RAMP_VARS.length], "#ff5fa2");
        s.marker = MARKERS[i % MARKERS.length];
        s.dash = DASHES[i % DASHES.length];
        s.visible = previous.has(s.key) ? previous.get(s.key) : true;
        return s;
      });

      renderLegend();
    }

    function renderLegend() {
      if (!legendEl) return;
      legendEl.textContent = "";

      if (!series.length) {
        const hint = document.createElement("span");
        hint.className = "mono";
        hint.textContent = "no series yet";
        legendEl.appendChild(hint);
        return;
      }

      for (const s of series) {
        const button = document.createElement("button");
        button.type = "button";
        button.setAttribute("aria-pressed", String(s.visible));
        button.dataset.seriesKey = s.key;

        const swatch = document.createElement("canvas");
        swatch.width = 34;
        swatch.height = 12;
        swatch.setAttribute("aria-hidden", "true");
        swatch.style.verticalAlign = "middle";
        swatch.style.marginRight = "7px";
        const sctx = swatch.getContext("2d");
        sctx.strokeStyle = s.color;
        sctx.fillStyle = s.color;
        sctx.lineWidth = 1.5;
        sctx.setLineDash(s.dash);
        sctx.beginPath();
        sctx.moveTo(1, 6);
        sctx.lineTo(33, 6);
        sctx.stroke();
        sctx.setLineDash([]);
        drawMarker(sctx, s.marker, 17, 6, 3.2);
        if (s.marker === "cross") sctx.stroke();
        else sctx.fill();

        button.appendChild(swatch);
        button.appendChild(document.createTextNode(s.key));
        button.addEventListener("click", () => {
          s.visible = !s.visible;
          button.setAttribute("aria-pressed", String(s.visible));
          draw();
        });

        legendEl.appendChild(button);
      }
    }

    function valueOf(row) {
      if (mode === "normalized") return normalizedNs(row);
      if (mode === "throughput") return throughputMBs(row);
      return row.medianNs;
    }

    function axisLabel() {
      if (mode === "normalized") return "ns / (n · log₂ n)";
      if (mode === "throughput") return "MB/s";
      return "time per transform";
    }

    function formatValue(v) {
      if (mode === "normalized") return `${trim(v)} ns`;
      if (mode === "throughput") return `${trim(v)} MB/s`;
      return formatTimeNs(v);
    }

    function resize() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      cssW = Math.max(1, Math.round(rect.width));
      cssH = Math.max(1, Math.round(rect.height));
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
      // Set once, here. Everything below draws in CSS pixels and never
      // hand-multiplies a line width by dpr.
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function draw() {
      resize();

      const lume = cssVar("--lume", "#e8ecff");
      const dim = cssVar("--dim", "#8189b8");
      const rule = cssVar("--rule", "#262c4e");
      const void_ = cssVar("--void", "#0b0e1f");

      ctx.clearRect(0, 0, cssW, cssH);
      ctx.fillStyle = void_;
      ctx.fillRect(0, 0, cssW, cssH);

      ctx.font = '11px "JetBrains Mono", ui-monospace, monospace';
      ctx.textBaseline = "middle";

      const visible = series.filter((s) => s.visible && s.rows.length);
      const points = visible.flatMap((s) =>
        s.rows.filter((r) => valueOf(r) > 0 && r.size >= 2),
      );

      if (!points.length) {
        ctx.fillStyle = dim;
        ctx.textAlign = "center";
        ctx.fillText("no data — run a sweep", cssW / 2, cssH / 2);
        hits = [];
        return;
      }

      const xLo = Math.min(...points.map((r) => r.size));
      const xHi = Math.max(...points.map((r) => r.size));
      let yLo = Math.min(...points.map(valueOf));
      let yHi = Math.max(...points.map(valueOf));

      const yLog = mode === "time";
      if (yLog) {
        yLo = Math.pow(10, Math.floor(Math.log10(yLo)));
        yHi = Math.pow(10, Math.ceil(Math.log10(yHi)));
        if (yHi <= yLo) yHi = yLo * 10;
      } else {
        yLo = 0;
        yHi = yHi * 1.12 || 1;
      }

      const yTicks = yLog
        ? niceLogTicks(yLo, yHi)
        : niceLinearTicks(yLo, yHi, 5);

      // Size the left gutter to the widest label we are about to draw. A fixed
      // pad is how you clip the leading digit off "1024 µs" and never notice.
      let widest = 0;
      for (const t of yTicks) {
        widest = Math.max(widest, ctx.measureText(formatValue(t)).width);
      }
      const padLeft = Math.ceil(widest) + GUTTER_SLACK;

      const plotX = padLeft;
      const plotY = PAD_TOP;
      const plotW = Math.max(1, cssW - padLeft - PAD_RIGHT);
      const plotH = Math.max(1, cssH - PAD_TOP - PAD_BOTTOM);

      const lx0 = Math.log10(xLo);
      const lx1 = Math.log10(xHi);
      const xSpan = lx1 - lx0 || 1;

      const sx = (n) => plotX + ((Math.log10(n) - lx0) / xSpan) * plotW;
      const sy = (v) => {
        if (yLog) {
          const l0 = Math.log10(yLo);
          const l1 = Math.log10(yHi);
          return plotY + plotH - ((Math.log10(v) - l0) / (l1 - l0)) * plotH;
        }
        return plotY + plotH - ((v - yLo) / (yHi - yLo)) * plotH;
      };

      // Grid and y ticks.
      ctx.strokeStyle = rule;
      ctx.lineWidth = 1;
      ctx.setLineDash([]);
      ctx.textAlign = "right";
      ctx.fillStyle = dim;

      for (const t of yTicks) {
        const y = Math.round(sy(t)) + 0.5;
        if (y < plotY - 1 || y > plotY + plotH + 1) continue;
        ctx.beginPath();
        ctx.moveTo(plotX, y);
        ctx.lineTo(plotX + plotW, y);
        ctx.stroke();
        ctx.fillText(formatValue(t), plotX - 8, y);
      }

      // x ticks: every power of two present in the data range.
      ctx.textAlign = "center";
      const xTicks = [];
      for (let p = 1; p <= 1 << 20; p *= 2) {
        if (p >= xLo && p <= xHi) xTicks.push(p);
      }
      if (!xTicks.length) xTicks.push(xLo, xHi);
      const xStride = Math.ceil(xTicks.length / Math.max(2, Math.floor(plotW / 56)));

      xTicks.forEach((t, i) => {
        const x = Math.round(sx(t)) + 0.5;
        ctx.strokeStyle = rule;
        ctx.beginPath();
        ctx.moveTo(x, plotY);
        ctx.lineTo(x, plotY + plotH);
        ctx.stroke();
        if (i % xStride === 0) {
          ctx.fillStyle = dim;
          ctx.fillText(formatSize(t), x, plotY + plotH + 13);
        }
      });

      // Axis titles.
      ctx.fillStyle = dim;
      ctx.textAlign = "left";
      ctx.fillText(axisLabel(), plotX, plotY - 8 < 8 ? 8 : plotY - 8);
      ctx.textAlign = "right";
      ctx.fillText("n", plotX + plotW, plotY + plotH + 26);

      // Series.
      hits = [];
      for (const s of visible) {
        const pts = s.rows
          .filter((r) => valueOf(r) > 0 && r.size >= 2)
          .map((r) => ({ row: r, x: sx(r.size), y: sy(valueOf(r)) }));
        if (!pts.length) continue;

        ctx.strokeStyle = s.color;
        ctx.lineWidth = 1.6;
        ctx.setLineDash(s.dash);
        ctx.beginPath();
        pts.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
        ctx.stroke();
        ctx.setLineDash([]);

        for (const p of pts) {
          ctx.fillStyle = p.row.reliable === false ? void_ : s.color;
          ctx.strokeStyle = s.color;
          ctx.lineWidth = 1.4;
          drawMarker(ctx, s.marker, p.x, p.y, 3.6);
          if (s.marker === "cross") ctx.stroke();
          else {
            ctx.fill();
            if (p.row.reliable === false) ctx.stroke();
          }
          hits.push({ x: p.x, y: p.y, row: p.row, series: s });
        }
      }

      ctx.strokeStyle = rule;
      ctx.lineWidth = 1;
      ctx.strokeRect(plotX + 0.5, plotY + 0.5, plotW - 1, plotH - 1);
      void lume;
    }

    function hideTooltip() {
      if (tooltipEl) tooltipEl.hidden = true;
    }

    function onPointerMove(event) {
      if (!tooltipEl || !hits.length) return;
      const rect = canvas.getBoundingClientRect();
      const px = event.clientX - rect.left;
      const py = event.clientY - rect.top;

      let best = null;
      let bestD = HIT_RADIUS * HIT_RADIUS;
      for (const h of hits) {
        const dx = h.x - px;
        const dy = h.y - py;
        const d = dx * dx + dy * dy;
        if (d < bestD) {
          bestD = d;
          best = h;
        }
      }

      if (!best) {
        hideTooltip();
        return;
      }

      const row = best.row;
      tooltipEl.textContent = "";
      const lines = [
        `n = ${row.size}`,
        `${row.precision} · ${row.strategyRequested} → ${row.strategyResolved || "?"}`,
        `${row.algorithm || "?"}`,
        `${formatTimeNs(row.medianNs)} / transform`,
        `${trim(normalizedNs(row))} ns per n·log₂n`,
        `${trim(throughputMBs(row))} MB/s`,
        `±${(100 * (row.relStddev || 0)).toFixed(1)}% over ${(row.trialNs || []).length} trials`,
        row.reliable === false ? "below reliable threshold" : "",
      ].filter(Boolean);

      for (const line of lines) {
        const div = document.createElement("div");
        div.textContent = line;
        tooltipEl.appendChild(div);
      }

      tooltipEl.hidden = false;
      const tw = tooltipEl.offsetWidth;
      const th = tooltipEl.offsetHeight;
      let left = best.x + 14;
      if (left + tw > cssW) left = best.x - tw - 14;
      let top = best.y - th / 2;
      top = Math.max(0, Math.min(cssH - th, top));
      tooltipEl.style.left = `${Math.max(0, left)}px`;
      tooltipEl.style.top = `${top}px`;
    }

    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerleave", hideTooltip);

    // The tooltip is absolutely positioned inside the chart wrapper, so a
    // stale one left over from a wider viewport sticks out past the page and
    // gives the whole document a horizontal scrollbar. Hide it on any resize.
    const onResize = () => {
      hideTooltip();
      draw();
    };
    window.addEventListener("resize", onResize);

    // devicePixelRatio changes on zoom or a move between monitors; a
    // matchMedia listener is the only way to hear about it.
    let dprQuery = null;
    function watchDpr() {
      if (dprQuery) dprQuery.removeEventListener("change", onDprChange);
      dprQuery = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
      dprQuery.addEventListener("change", onDprChange);
    }
    function onDprChange() {
      watchDpr();
      draw();
    }
    watchDpr();

    if (opts.modeButtons) {
      opts.modeButtons.forEach((button) => {
        button.addEventListener("click", () => {
          mode = button.dataset.mode;
          opts.modeButtons.forEach((other) =>
            other.setAttribute("aria-pressed", String(other === button)),
          );
          hideTooltip();
          draw();
        });
      });
    }

    return {
      setRows(next) {
        rows = next.slice();
        rebuildSeries();
        hideTooltip();
        draw();
      },
      redraw: draw,
      get mode() {
        return mode;
      },
    };
  }

  globalThis.BenchChart = {
    create: createBenchChart,
    bytesPerElement,
    throughputMBs,
    normalizedNs,
    formatTimeNs,
  };
})();
