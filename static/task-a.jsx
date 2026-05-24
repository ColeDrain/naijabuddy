/* global React, Icon */
// Task A — User Modeling: predict rating + review

// UI platform (lowercase) → DB domain (Capitalized, as seeded by data_enricher).
const _DOMAIN_BY_PLATFORM = { yelp: "Yelp", amazon: "Amazon", goodreads: "Goodreads" };

function TaskA({ platform, naija, personaA, setPersonaA, setProductA }) {
  const [items, setItems] = React.useState(null);         // array | null while fetching
  const [selectedId, setSelectedId] = React.useState(null);
  const [shuffleTick, setShuffleTick] = React.useState(0);
  const [loading, setLoading] = React.useState(false);
  const [result, setResult] = React.useState(null);
  const [error, setError] = React.useState(null);
  const [openExplain, setOpenExplain] = React.useState(false);
  const outputRef = React.useRef(null);

  // Re-fetch a fresh 12-item random sample whenever the platform or shuffle
  // bumps. /api/items?domain=X&limit=12 does the ORDER BY RANDOM() server-side
  // so the wire payload stays small (~12 rows × ~5 short fields) even though
  // the catalogue is ~90 K items.
  React.useEffect(() => {
    const apiDomain = _DOMAIN_BY_PLATFORM[platform];
    if (!apiDomain) { setItems([]); return; }
    let alive = true;
    setItems(null);
    fetch(`/api/items?domain=${encodeURIComponent(apiDomain)}&limit=12`)
      .then((r) => (r.ok ? r.json() : []))
      .then((data) => {
        if (!alive) return;
        const arr = Array.isArray(data) ? data : [];
        setItems(arr);
        // Auto-select the first item so judges can hit Generate immediately
        // without having to also click a card.
        setSelectedId(arr[0]?.id ?? null);
      })
      .catch(() => { if (alive) { setItems([]); setSelectedId(null); } });
    return () => { alive = false; };
  }, [platform, shuffleTick]);

  // Currently-selected item and the corresponding product payload sent to
  // /task_a. The catalogue table doesn't store price, so we leave that blank;
  // the agent only uses title/description/category/avg_rating substantively.
  const selectedItem = (items || []).find((it) => it.id === selectedId) || null;
  const product = selectedItem
    ? {
        title: selectedItem.name || "",
        description: selectedItem.description || "",
        category: selectedItem.category || "",
        price: "",
        average_rating:
          selectedItem.average_rating != null
            ? String(selectedItem.average_rating)
            : "",
      }
    : null;

  // Smooth-scroll to the output region whenever loading kicks off or a fresh
  // result lands — judges click "Generate" and the result is below the fold,
  // so bring it into view without making them hunt for it.
  React.useEffect(() => {
    if (!loading && !result && !error) return;
    const el = outputRef.current;
    if (!el) return;
    const top = el.getBoundingClientRect().top + window.scrollY - 80;
    window.scrollTo({ top, behavior: "smooth" });
  }, [loading, result, error]);

  // Lift the current product up to app.jsx so the Compare view (Neutral vs
  // Naija side-by-side) can fire its calls with the same product the user
  // sees here. Stringify-then-parse the selectedId in the dep array so the
  // effect only fires on real selection changes, not every re-render.
  React.useEffect(() => {
    if (typeof setProductA === "function") setProductA(product);
  }, [selectedId, items, setProductA]);  // eslint-disable-line react-hooks/exhaustive-deps

  async function submit() {
    if (!product) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setOpenExplain(false);
    const r = await window.api("/task_a", {
      user_persona: personaA,
      product,
      platform,
      naija_mode: naija,
    });
    setLoading(false);
    if (!r.ok) setError(r.error || "Network error");
    else setResult(r.data);
  }

  const canSubmit = personaA.trim().length > 4 && !!selectedItem && !loading;
  const showSpinner = items === null;

  return (
    <div className="fade-in">
      <SectionHeader
        eyebrow="Task A · User Modeling"
        title="Predict rating &amp; review"
        sub="Given a user persona and a product, the agent predicts a 1–5 star rating and writes a short review in the persona's voice."
      />

      <div className="space-y-7">
        {/* Persona */}
        <div className="space-y-2.5">
          <div className="flex items-baseline justify-between gap-3">
            <label className="field-label">User persona</label>
            <span className="field-help">multiline · pick a chip below to prefill</span>
          </div>
          <textarea
            className="textarea"
            placeholder="Describe the user — what they buy, what they like, where they live, how they shop…"
            value={personaA}
            onChange={(e) => setPersonaA(e.target.value)}
          />
          <div className="pt-1">
            <PersonaChips
              platform={platform}
              value={personaA}
              onPick={(p) => setPersonaA(p)}
            />
          </div>
        </div>

        {/* Product — picked from the live catalogue (no hardcoded sample data) */}
        <div className="space-y-3">
          <div className="flex items-baseline justify-between gap-3">
            <label className="field-label">Pick a product</label>
            <div className="flex items-center gap-3">
              <span className="field-help">
                {showSpinner
                  ? "loading…"
                  : `${(items || []).length} from the ${_DOMAIN_BY_PLATFORM[platform]} catalogue`}
              </span>
              <button
                type="button"
                className="btn btn-ghost btn-xs"
                onClick={() => setShuffleTick((t) => t + 1)}
                disabled={loading || showSpinner}
              >Shuffle</button>
            </div>
          </div>

          {/* Item chip grid */}
          {showSpinner ? (
            <div className="text-stone4 text-xs">loading catalogue items…</div>
          ) : (items || []).length === 0 ? (
            <div className="text-stone4 text-xs">no items found for this platform.</div>
          ) : (
            <div className="flex flex-wrap gap-2">
              {(items || []).map((it) => (
                <button
                  key={it.id}
                  type="button"
                  className="chip"
                  data-active={selectedId === it.id}
                  onClick={() => setSelectedId(it.id)}
                  title={`${it.name}\n${it.category || ""}\n\n${(it.description || "").slice(0, 240)}`}
                >
                  {it.name}
                </button>
              ))}
            </div>
          )}

          {/* Read-only details of the selected item */}
          {selectedItem && (
            <div className="flex gap-4 items-start pt-2">
              <ProductSlot width={140} height={140} rounded={8} />
              <div className="grid grid-cols-12 gap-3 flex-1">
                <div className="col-span-12">
                  <label className="input-label">Title</label>
                  <div className="input" style={{minHeight: 38, display:"flex", alignItems:"center"}}>
                    {selectedItem.name}
                  </div>
                </div>
                <div className="col-span-12">
                  <label className="input-label">Description</label>
                  <div className="textarea" style={{minHeight: 72, whiteSpace:"pre-wrap"}}>
                    {selectedItem.description || <span className="text-stone4">—</span>}
                  </div>
                </div>
                <div className="col-span-8">
                  <label className="input-label">Category</label>
                  <div className="input" style={{minHeight: 38, display:"flex", alignItems:"center"}}>
                    {selectedItem.category || <span className="text-stone4">—</span>}
                  </div>
                </div>
                <div className="col-span-4">
                  <label className="input-label">Avg rating</label>
                  <div className="input" style={{minHeight: 38, display:"flex", alignItems:"center"}}>
                    {selectedItem.average_rating != null
                      ? `${Number(selectedItem.average_rating).toFixed(1)} ★`
                      : <span className="text-stone4">—</span>}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Submit row */}
        <div className="flex items-center justify-between gap-4">
          <ElapsedTimer active={loading} />
          <div className="flex items-center gap-3">
            <button className="btn btn-ghost" onClick={() => {
              setPersonaA("");
              setShuffleTick((t) => t + 1);
              setResult(null); setError(null);
            }} disabled={loading}>Reset</button>
            <button className="btn btn-primary" onClick={submit} disabled={!canSubmit}>
              {loading ? "Generating…" : <>Generate prediction <Icon.ChevronRight size={14}/></>}
            </button>
          </div>
        </div>

        {/* Result / skeleton / error */}
        <div ref={outputRef}>
          {loading && <SkeletonResultCard />}
          {error && (
            <ErrorToast
              message={error}
              onRetry={submit}
              onDismiss={() => setError(null)}
            />
          )}
          {result && !loading && (
            <ResultA result={result} naija={naija} openExplain={openExplain} setOpenExplain={setOpenExplain} />
          )}
        </div>
      </div>
    </div>
  );
}

function ResultA({ result, naija, openExplain, setOpenExplain }) {
  return (
    <div className="fade-in space-y-4">
      <ResultCardShell naija={result.naija_mode} label="Predicted output">
        {/* Star row */}
        <div className="flex items-end gap-5 mb-4">
          <Stars value={Math.round(result.stars)} size={28} />
          <div className="font-display font-bold text-[28px] leading-none text-ink tracking-tighter2">
            {Number(result.stars).toFixed(2)}
            <span className="text-stone4 font-medium text-[18px] tracking-normal"> /5</span>
          </div>
        </div>

        {/* Review */}
        <div className="font-serif text-[17px] leading-[1.62] text-ink"
             dangerouslySetInnerHTML={renderReview(result.review, !!result.naija_mode)} />

        {/* Tag row */}
        <div className="mt-5 flex flex-wrap items-center gap-2.5">
          <span className="tag">
            <span className="dot"/>
            {result.model} · 3-term calibration · BGE retrieval
          </span>
          <span className="tag" style={{background:"transparent"}}>
            <span className="dot" style={{background:"#C75D3A"}} />
            {result.latency_ms} ms
          </span>
        </div>

        {/* Explain */}
        <div className="mt-5 pt-4 border-t border-stone2">
          <button
            className="btn btn-link inline-flex items-center gap-1.5 group"
            onClick={() => setOpenExplain(v => !v)}
            aria-expanded={openExplain}
          >
            <Icon.Sparkles size={14}/>
            <span>Why this rating?</span>
            <span className={`transition-transform ${openExplain ? "rotate-180" : ""}`}>
              <Icon.Chevron size={14}/>
            </span>
          </button>
          {openExplain && (
            <div className="mt-3 text-[14px] leading-[1.6] text-[#41475A] fade-in">
              {result.reasoning}
            </div>
          )}
        </div>
      </ResultCardShell>
    </div>
  );
}

// Reviews may contain <u>…</u> spans (Pidgin underline cues) we render as ochre underline.
function renderReview(text, naija) {
  if (!text) return { __html: "" };
  let html = text;
  if (naija) {
    html = html.replace(/<u>(.+?)<\/u>/g, '<span class="pidgin-mark">$1</span>');
  } else {
    html = html.replace(/<u>(.+?)<\/u>/g, '$1');
  }
  return { __html: html };
}

// ---------- Section header (shared) ----------
function SectionHeader({ eyebrow, title, sub }) {
  return (
    <div className="mb-8">
      <div className="mono text-[11px] uppercase tracking-wider text-stone4 mb-2">{eyebrow}</div>
      <h2 className="font-display font-bold text-[34px] leading-[1.1] tracking-tighter2 text-ink"
          dangerouslySetInnerHTML={{__html:title}} />
      {sub && <p className="text-[15px] leading-[1.55] text-[#54596A] mt-2 max-w-[58ch]">{sub}</p>}
    </div>
  );
}

window.TaskA = TaskA;
window.SectionHeader = SectionHeader;
window.renderReview = renderReview;
