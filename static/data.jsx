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

// Sample products per platform (prefilled into the Task A form).
// Multiple variants per platform — the form picks one at random on mount /
// platform switch, and the "Shuffle" button cycles to the next.
const SAMPLE_PRODUCTS = {
  amazon: [
    {
      title: "Anker 7-in-1 USB-C Hub (PD 100W, 4K HDMI)",
      description: "Compact aluminium dock with HDMI, ethernet, SD card and three USB-A ports. Pass-through charging at 100W.",
      category: "Electronics › Computer Accessories",
      price: "44.99",
      average_rating: "4.4",
    },
    {
      title: "Hakko FX-888D Digital Soldering Station",
      description: "70W digital iron with thermal recovery, ceramic heater, password-locked presets. Replaceable tips. Hobbyist-grade durability.",
      category: "Tools › Soldering",
      price: "129.00",
      average_rating: "4.7",
    },
    {
      title: "JBL Flip 6 Portable Bluetooth Speaker",
      description: "IP67 rugged Bluetooth speaker with 12-hour playtime, PartyBoost pairing, racetrack-shaped passive radiators.",
      category: "Electronics › Speakers",
      price: "99.95",
      average_rating: "4.6",
    },
    {
      title: "Generic Cheap USB Hub (no brand)",
      description: "4-port USB 2.0 splitter from a no-name overseas seller. Plastic shell, unshielded cable. Mixed reviews on durability.",
      category: "Electronics › Hubs",
      price: "5.99",
      average_rating: "2.1",
    },
    {
      title: "Stanley FatMax 25ft Tape Measure",
      description: "Heavy-duty tape measure with 11-foot standout, BladeArmor coating on first 3 inches, magnetic hook tip.",
      category: "Tools › Measuring",
      price: "21.99",
      average_rating: "4.8",
    },
  ],
  goodreads: [
    {
      title: "Children of Blood and Bone — Tomi Adeyemi",
      description: "West African–inspired YA fantasy. Zélie must restore magic to her people before a ruthless prince eradicates her kind.",
      category: "Young Adult Fantasy",
      price: "—",
      average_rating: "4.1",
    },
    {
      title: "The Gilded Ones — Namina Forna",
      description: "Sierra Leonean–inspired YA fantasy. Sixteen-year-old Deka bleeds gold during a purity ritual — marking her as a demon — and is recruited into an army of outcast girls.",
      category: "Young Adult Fantasy",
      price: "—",
      average_rating: "4.0",
    },
    {
      title: "Pachinko — Min Jin Lee",
      description: "Four generations of a Korean family in 20th-century Japan. National Book Award finalist; sweeping family saga across war, occupation, exile.",
      category: "Historical Fiction",
      price: "—",
      average_rating: "4.4",
    },
    {
      title: "A Little Life — Hanya Yanagihara",
      description: "Four college friends in New York across thirty years. Centers on Jude, a man whose past keeps re-shaping his present. Divisive among readers.",
      category: "Contemporary Fiction",
      price: "—",
      average_rating: "4.3",
    },
    {
      title: "Forgotten Indie Sci-Fi (1997)",
      description: "Out-of-print pulp sci-fi novel with a small cult following on Reddit. Dated prose; uneven pacing; striking ending.",
      category: "Science Fiction",
      price: "—",
      average_rating: "3.2",
    },
  ],
  yelp: [
    {
      title: "Buya Ramen — Saint Louis, MO",
      description: "Counter-style ramen shop. Tonkotsu, miso, spicy chashu; small selection of buns and karaage.",
      category: "Ramen · Japanese",
      price: "$$",
      average_rating: "4.3",
    },
    {
      title: "Yellow Chilli — Victoria Island, Lagos",
      description: "Upscale Nigerian restaurant in V.I. Pepper soup, jollof rice, fried plantain, suya platter. White-tablecloth service.",
      category: "Nigerian · Fine Dining",
      price: "$$$",
      average_rating: "4.5",
    },
    {
      title: "Mama Cass — Surulere, Lagos",
      description: "Casual local chain. Jollof rice, egusi soup, dodo, peppered snail. Quick lunch favorite among office workers.",
      category: "Nigerian · Casual",
      price: "$",
      average_rating: "4.1",
    },
    {
      title: "Joe's Pizza — Bleecker Street, NYC",
      description: "Walk-up slice counter. Plain cheese and pepperoni slices on thin NY-style crust. Open until 4am.",
      category: "Pizza · American",
      price: "$",
      average_rating: "4.7",
    },
    {
      title: "Tourist-Trap Diner near Hotel Strip",
      description: "Bland mid-priced diner aimed at out-of-town visitors. Frozen ingredients, slow service, mostly negative reviews on TripAdvisor.",
      category: "American · Diner",
      price: "$$",
      average_rating: "2.4",
    },
  ],
};

// Back-compat: first variant of each platform.
const SAMPLE_PRODUCT = Object.fromEntries(
  Object.entries(SAMPLE_PRODUCTS).map(([k, v]) => [k, v[0]])
);

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
  PERSONAS, PLATFORMS, SAMPLE_PRODUCT, SAMPLE_PRODUCTS, api,
  DATASETS, latencyBus, recordLatency,
});
