/* global React */
const { useState, useEffect, useRef, useMemo, useCallback } = React;

// NaijaBuddy serves three domains — the three datasets named in the brief.
// (The prior prototype also listed MovieLens / Amazon Books; those are not
// part of this system and have been removed.)

// ----- Sample personas, per platform (prefilled as chips in the forms) -----
const PERSONAS = {
  yelp: [
    "Casual diner in Tampa, loves outdoor seating",
    "Nigerian student in Tampa hunting spicy spots",
    "Saint Louis foodie, weekend brunch hopper",
  ],
  amazon: [
    "Electronics hobbyist (USB-C hubs, soldering kits)",
    "Lagos engineer kitting home workshop, tools + cables",
    "First-time office-supply buyer",
  ],
  goodreads: [
    "Romance fan, slow-burn YA",
    "Naija readathon participant, reads on okada commute",
    "Sci-fi enthusiast leaning Asian-author fiction",
  ],
};

const PLATFORMS = [
  { id: "yelp",      label: "Yelp" },
  { id: "amazon",    label: "Amazon" },
  { id: "goodreads", label: "Goodreads" },
];

// Task A's product input is now sourced from the live /api/items endpoint
// (see static/task-a.jsx). The previous SAMPLE_PRODUCTS / SAMPLE_PRODUCT
// hardcoded sample list has been removed — judges pick from the real
// catalogue instead of editing inlined demo data.

// ----- API layer. Calls the real /task_a /task_b agent endpoints. -----
// No mock fallback: if the agent errors, the UI surfaces a real error rather
// than silently showing fake output.
async function api(endpoint, body) {
  const t0 = performance.now();
  try {
    const r = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const elapsed = Math.round(performance.now() - t0);
    if (!r.ok) {
      let detail = "HTTP " + r.status;
      try { const j = await r.json(); if (j && j.detail) detail = j.detail; } catch (_) {}
      return { ok: false, error: detail };
    }
    const json = await r.json();
    recordLatency(endpoint, elapsed, !!body.naija_mode);
    return { ok: true, data: json };
  } catch (e) {
    return { ok: false, error: "Could not reach the agent — " + String(e) };
  }
}

// ----- Dataset cards (popover content) -----
const DATASETS = {
  yelp: {
    name: "Yelp Open Dataset",
    domain: "Local businesses — restaurants & services",
    size: "Densified 3-core subset · every user & item has ≥3 interactions",
    fields: "stars · review_text · category · city",
    notes: "Sampled from the public Yelp Open Dataset; geographic coverage skews to Tampa, St. Louis and similar US metros.",
  },
  amazon: {
    name: "Amazon Reviews",
    domain: "Electronics, home and office products",
    size: "Densified 3-core subset · every user & item has ≥3 interactions",
    fields: "rating · review_text · category · price",
    notes: "Sampled from the public McAuley/UCSD Amazon Reviews release.",
  },
  goodreads: {
    name: "Goodreads (UCSD)",
    domain: "Reader-driven book ratings and reviews",
    size: "Densified 3-core subset · every user & item has ≥3 interactions",
    fields: "rating · review_text · shelves · author",
    notes: "Sampled from the public UCSD Goodreads dump; skews toward English-language YA / fantasy / romance.",
  },
};

// ----- Latency bus (small pub/sub for any latency readout) -----
const latencyBus = (() => {
  const listeners = new Set();
  const samples = []; // {ts, ms, endpoint, naija}
  return {
    samples,
    subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); },
    emit() { listeners.forEach(fn => fn(samples)); },
  };
})();
function recordLatency(endpoint, ms, naija) {
  latencyBus.samples.push({ ts: Date.now(), ms, endpoint, naija });
  if (latencyBus.samples.length > 24) latencyBus.samples.shift();
  latencyBus.emit();
}

Object.assign(window, {
  PERSONAS, PLATFORMS, api,
  DATASETS, latencyBus, recordLatency,
});
