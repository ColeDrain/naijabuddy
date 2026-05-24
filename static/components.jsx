/* global React, Icon */
const { useState: useStateC, useEffect: useEffectC, useRef: useRefC } = React;

// ---------- Stars ----------
function Stars({ value=0, size=22 }) {
  return (
    <span className="stars" aria-label={`${value} out of 5 stars`}>
      {[1,2,3,4,5].map(i => (
        <span key={i} className={i <= value ? "star-ochre" : "star-empty"}>
          {i <= value ? <Icon.Star size={size} /> : <Icon.StarOutline size={size} />}
        </span>
      ))}
    </span>
  );
}

// ---------- Naija badge corner ----------
function NaijaCorner() {
  return (
    <div className="naija-corner" aria-label="Naija mode active">
      <span className="flag" aria-hidden="true"><i></i><i></i><i></i></span>
      <span>Naija mode</span>
    </div>
  );
}

// ---------- Tooltip helper ----------
function Tooltip({ text, children }) {
  return (
    <span className="tt inline-flex items-center" tabIndex={0}>
      {children}
      <span className="tt-bubble">{text}</span>
    </span>
  );
}

// ---------- Skeleton blocks ----------
function Skeleton({ w="100%", h=14, className="" }) {
  return <div className={`skel ${className}`} style={{width:w, height:h}} />;
}

function SkeletonResultCard() {
  return (
    <div className="card p-7 fade-in" aria-busy="true" aria-live="polite">
      <div className="flex items-center gap-5 mb-5">
        <Skeleton w={156} h={26} />
        <Skeleton w={56} h={20} />
      </div>
      <div className="space-y-3">
        <Skeleton w="100%" h={14} />
        <Skeleton w="92%" h={14} />
        <Skeleton w="74%" h={14} />
      </div>
      <div className="mt-6 flex gap-2">
        <Skeleton w={210} h={22} />
      </div>
    </div>
  );
}

function SkeletonRankList() {
  return (
    <div className="space-y-3 fade-in" aria-busy="true" aria-live="polite">
      {[1,2,3,4,5].map(i => (
        <div className="card p-5 flex gap-5 items-start" key={i}>
          <Skeleton w={20} h={20} />
          <div className="flex-1 space-y-3">
            <Skeleton w="55%" h={16} />
            <Skeleton w="80%" h={12} />
          </div>
          <Skeleton w={48} h={20} />
        </div>
      ))}
    </div>
  );
}

// ---------- Elapsed timer + "Still thinking…" ----------
function ElapsedTimer({ active }) {
  const [t, setT] = React.useState(0);
  React.useEffect(() => {
    if (!active) { setT(0); return; }
    const start = performance.now();
    const id = setInterval(() => setT(Math.floor(performance.now() - start)), 80);
    return () => clearInterval(id);
  }, [active]);
  if (!active) return null;
  const sec = (t / 1000).toFixed(1);
  return (
    <div className="flex items-center gap-3 mono text-[12px] text-[#6B6F7A]">
      <span className="inline-flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald9 animate-pulse"></span>
        elapsed {sec}s
      </span>
      {t > 10000 && <span className="text-terracotta">Still thinking…</span>}
    </div>
  );
}

// ---------- Error Toast ----------
function ErrorToast({ message, onRetry, onDismiss }) {
  if (!message) return null;
  return (
    <div className="toast fade-in" role="alert">
      <div className="accent pt-0.5"><Icon.AlertTri size={18} /></div>
      <div className="flex-1">
        <div className="font-display font-semibold text-[14px]">Request failed</div>
        <div className="text-[13px] text-stone3 mt-0.5">{message}</div>
      </div>
      <div className="flex items-center gap-1">
        <button className="btn btn-link text-paper hover:text-ochre" onClick={onRetry}>
          <Icon.Refresh size={14}/> Retry
        </button>
        <button onClick={onDismiss} className="opacity-60 hover:opacity-100 p-1">
          <Icon.X size={14}/>
        </button>
      </div>
    </div>
  );
}

// ---------- Pill toggle (Naija Mode) ----------
function NaijaToggle({ on, onChange }) {
  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        role="switch"
        aria-checked={on}
        className="pill"
        data-on={on}
        onClick={() => onChange(!on)}
      >
        <span className="knob" aria-hidden="true"><i/></span>
        <span>Naija mode</span>
      </button>
      <Tooltip text="Switches model output style to Pidgin English & Nigerian cultural cues. Data sources remain unchanged.">
        <button type="button" className="w-6 h-6 grid place-items-center rounded-full text-stone4 hover:text-ink hover:bg-stone1" aria-label="What is Naija mode?">
          <Icon.Help size={14}/>
        </button>
      </Tooltip>
    </div>
  );
}

// ---------- Platform dropdown ----------
function PlatformSelect({ value, onChange }) {
  return (
    <div className="flex items-center gap-2">
      <label className="field-label" style={{fontSize:11}}>Platform</label>
      <div className="relative">
        <select
          className="select pr-9 mono text-[13px] py-2"
          style={{minWidth: 168}}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        >
          {window.PLATFORMS.map(p => (
            <option key={p.id} value={p.id}>{p.label}</option>
          ))}
        </select>
        <div className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-stone4">
          <Icon.Chevron size={14}/>
        </div>
      </div>
    </div>
  );
}

// ---------- Sample persona chips ----------
// Fetches the real synthesised personas from /api/users (built by
// generate_personas.py at Docker-build time, stored in the seeded SQLite),
// filters them to the current platform via the user.name prefix
// ("Yelp User ...", "Goodreads User ...", "Amazon User ..."), and shows
// six at random as clickable chips. Clicking loads the full persona text
// into the textarea. A "Shuffle different personas" link re-samples six
// new ones. Falls back to the small hardcoded set in data.jsx if the API
// is unreachable or returns nothing.
const _PLATFORM_LABEL = { yelp: "Yelp", amazon: "Amazon", goodreads: "Goodreads" };
const _PLATFORM_PREFIXES = ["Yelp User", "Amazon User", "Goodreads User"];

function PersonaChips({ platform, value, onPick }) {
  const [users, setUsers] = React.useState(null);   // null = not yet fetched
  const [sample, setSample] = React.useState([]);
  const [tick, setTick] = React.useState(0);        // bump to force re-sample
  // Pool: "platform" (default — picks from the active platform's Yelp/Amazon/
  // Goodreads users) or "nigerian" (the localized Nigerian personas seeded by
  // data_enricher.py — they have descriptive names like "Teni (Lagos Gen-Z
  // Influencer)" and do NOT carry a platform prefix, so we identify them as
  // "any user whose name doesn't match any platform prefix").
  const [pool, setPool] = React.useState("platform");

  // Fetch all users once on mount. Light payload (~6,000 rows × few fields).
  React.useEffect(() => {
    let alive = true;
    fetch("/api/users")
      .then((r) => (r.ok ? r.json() : []))
      .then((data) => { if (alive) setUsers(Array.isArray(data) ? data : []); })
      .catch(() => { if (alive) setUsers([]); });
    return () => { alive = false; };
  }, []);

  // Re-sample six personas whenever the platform, pool, or shuffle tick changes.
  React.useEffect(() => {
    if (users == null) return;
    let filtered;
    if (pool === "nigerian") {
      filtered = users.filter((u) => {
        const n = u.name || "";
        return !_PLATFORM_PREFIXES.some((p) => n.startsWith(p));
      });
    } else {
      const prefix = _PLATFORM_LABEL[platform] || "";
      filtered = users.filter((u) => (u.name || "").startsWith(prefix));
    }
    if (filtered.length === 0) {
      setSample([]);
      return;
    }
    // Fisher–Yates partial shuffle, take 6.
    const arr = filtered.slice();
    for (let i = arr.length - 1; i > 0 && i > arr.length - 7; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    setSample(arr.slice(-6));
  }, [users, platform, pool, tick]);

  // Pool toggle — shown above the chips in both loading and loaded states so
  // judges can switch source at any point. "Nigerian" surfaces the localized
  // personas that the platform-prefix filter would otherwise hide.
  const poolToggle = (
    <div className="flex gap-1.5 text-[11px]">
      <button type="button" className="chip"
              data-active={pool === "platform"}
              onClick={() => setPool("platform")}>
        {_PLATFORM_LABEL[platform] || "Platform"} users
      </button>
      <button type="button" className="chip"
              data-active={pool === "nigerian"}
              onClick={() => setPool("nigerian")}
              title="Localized Nigerian personas seeded over the bundled Nigerian catalogue">
        🇳🇬 Nigerian
      </button>
    </div>
  );

  // Still loading? Show a quiet placeholder rather than the stale hardcoded list.
  if (users == null) {
    return (
      <div className="space-y-2">
        {poolToggle}
        <div className="text-stone4 text-xs">loading personas…</div>
      </div>
    );
  }

  // Backend reachable but no personas for this filter — fall back to the
  // hardcoded sample list (only happens in local dev before the DB is seeded,
  // or if the Nigerian pool somehow ends up empty).
  if (sample.length === 0) {
    const list = pool === "nigerian"
      ? []  // no hardcoded Nigerian fallback — empty state shown below
      : ((window.PERSONAS || {})[platform] || []);
    if (list.length === 0) {
      return (
        <div className="space-y-2">
          {poolToggle}
          <div className="text-stone4 text-xs">no personas match this filter yet — the catalogue may still be seeding.</div>
        </div>
      );
    }
    return (
      <div className="space-y-2">
        {poolToggle}
        <div className="flex flex-wrap gap-2">
          {list.map((p) => (
            <button key={p} type="button" className="chip"
                    data-active={value === p} onClick={() => onPick(p)}>
              {p}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {poolToggle}
      <div className="flex flex-wrap gap-2">
        {sample.map((u) => (
          <button
            key={u.id}
            type="button"
            className="chip"
            data-active={value === u.persona}
            onClick={() => onPick(u.persona)}
            title={`${u.name} · avg ${
              u.user_mean_rating != null ? u.user_mean_rating.toFixed(1) + "★" : "—"
            }\n\n${u.persona || ""}`.slice(0, 400)}
          >
            {u.name}
          </button>
        ))}
      </div>
      <button
        type="button"
        className="text-[11px] text-stone4 hover:text-ink underline underline-offset-2"
        onClick={() => setTick((t) => t + 1)}
      >
        Shuffle different personas
      </button>
    </div>
  );
}

// ---------- Result card chrome (used by Task A) ----------
function ResultCardShell({ naija, children, label }) {
  return (
    <div className={`card p-7 relative ${naija ? "naija-edge" : ""}`}>
      {naija && <NaijaCorner />}
      {label && <div className="mono text-[11px] uppercase tracking-wider text-stone4 mb-3">{label}</div>}
      {children}
    </div>
  );
}

// ---------- Health pill ----------
function HealthPill() {
  // Cosmetic only — checks /health, falls back to "warm".
  const [state, setState] = React.useState("warm");
  React.useEffect(() => {
    let alive = true;
    fetch("/health").then(r => r.ok ? r.json() : null)
      .then(j => { if (alive && j) setState(j.model_warm ? "warm" : "cold"); })
      .catch(() => { /* keep default */ });
    return () => { alive = false; };
  }, []);
  return (
    <span className="tag" title="Backend health">
      <span className="dot" />
      <span>model {state}</span>
    </span>
  );
}

// ---------- Static product image (dummy placeholder) ----------
function ProductSlot({ width=120, height=120, className="", rounded=8 }) {
  return (
    <img
      src="placeholder.svg"
      alt=""
      width={width}
      height={height}
      className={className}
      style={{
        width: `${width}px`,
        height: `${height}px`,
        borderRadius: `${rounded}px`,
        border: "1px solid #E5E1D8",
        objectFit: "cover",
        flex: "none",
        display: "block",
      }}
    />
  );
}

// ---------- Latency histogram (mini SVG bars) ----------
function LatencyHistogram() {
  const [samples, setSamples] = React.useState(window.latencyBus.samples.slice());
  React.useEffect(() => {
    const off = window.latencyBus.subscribe((s) => setSamples(s.slice()));
    return off;
  }, []);
  // Always 24 columns, pad left with zeros.
  const cols = 24;
  const W = 120, H = 22, BW = (W - (cols-1)*2) / cols;
  const padded = [...Array(Math.max(0, cols - samples.length)).fill(null), ...samples.slice(-cols)];
  const max = Math.max(2500, ...samples.map(s => s.ms));
  const last = samples[samples.length - 1];
  const avg = samples.length ? Math.round(samples.reduce((s,x)=>s+x.ms,0)/samples.length) : null;
  return (
    <div className="flex items-center gap-3">
      <span className="mono text-[11px] text-stone4 uppercase tracking-wider">latency</span>
      <svg width={W} height={H} aria-label="recent request latencies">
        {padded.map((s, i) => {
          const h = s ? Math.max(2, Math.round((s.ms / max) * (H - 2))) : 2;
          const x = i * (BW + 2);
          const y = H - h;
          const fill = !s ? "#E5E1D8" : s.naija ? "#00563B" : "#1A1F2E";
          return <rect key={i} x={x} y={y} width={BW} height={h} fill={fill} rx="1" opacity={s ? 1 : 0.6} />;
        })}
      </svg>
      <span className="mono text-[11px] text-[#54596A]">
        {last ? `${last.ms} ms` : "—"}
        {avg && <span className="text-stone4"> · avg {avg}</span>}
      </span>
    </div>
  );
}

// ---------- Dataset popover (anchored next to platform select) ----------
function DatasetPopover({ platform }) {
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef(null);
  React.useEffect(() => {
    function onDoc(e) {
      if (!open) return;
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);
  const d = window.DATASETS[platform];
  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen(v => !v)}
        className="w-6 h-6 grid place-items-center rounded-full text-stone4 hover:text-ink hover:bg-stone1"
        aria-label={`About the ${platform} dataset`}
        aria-expanded={open}
      >
        <Icon.Help size={14}/>
      </button>
      {open && (
        <div className="absolute right-0 top-[calc(100%+8px)] z-30 w-[320px] card p-4 shadow-pop fade-in" role="dialog">
          <div className="flex items-baseline justify-between gap-3 mb-1.5">
            <div className="font-display font-bold text-[14.5px] tracking-tightish text-ink">{d.name}</div>
            <span className="mono text-[10.5px] text-stone4 uppercase tracking-wider">{platform}</span>
          </div>
          <div className="text-[12.5px] text-[#54596A] leading-[1.5] mb-3">{d.domain}</div>
          <dl className="text-[12px] space-y-1.5">
            <div className="flex gap-3">
              <dt className="mono text-stone4 uppercase tracking-wider w-[60px] flex-none text-[10.5px] pt-0.5">scale</dt>
              <dd className="text-ink">{d.size}</dd>
            </div>
            <div className="flex gap-3">
              <dt className="mono text-stone4 uppercase tracking-wider w-[60px] flex-none text-[10.5px] pt-0.5">fields</dt>
              <dd className="text-ink mono text-[11.5px]">{d.fields}</dd>
            </div>
          </dl>
          <div className="text-[12px] leading-[1.5] text-[#54596A] mt-3 pt-3 border-t border-stone2">
            {d.notes}
          </div>
        </div>
      )}
    </div>
  );
}

Object.assign(window, {
  Stars, NaijaCorner, Tooltip, Skeleton, SkeletonResultCard, SkeletonRankList,
  ElapsedTimer, ErrorToast, NaijaToggle, PlatformSelect, PersonaChips,
  ResultCardShell, HealthPill,
  DatasetPopover, LatencyHistogram, ProductSlot,
});
