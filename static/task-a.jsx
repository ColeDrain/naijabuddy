/* global React, Icon */
// Task A — User Modeling: predict rating + review

function TaskA({ platform, naija, personaA, setPersonaA }) {
  const pickRandomSample = (plat) => {
    const variants = window.SAMPLE_PRODUCTS?.[plat] || [window.SAMPLE_PRODUCT[plat]];
    const idx = Math.floor(Math.random() * variants.length);
    return { variant: variants[idx], idx };
  };
  const initSample = pickRandomSample(platform);
  const [product, setProduct] = React.useState(initSample.variant);
  const [sampleIdx, setSampleIdx] = React.useState(initSample.idx);
  const [loading, setLoading] = React.useState(false);
  const [result, setResult] = React.useState(null);
  const [error, setError] = React.useState(null);
  const [openExplain, setOpenExplain] = React.useState(false);
  const outputRef = React.useRef(null);

  React.useEffect(() => {
    const { variant, idx } = pickRandomSample(platform);
    setProduct(variant);
    setSampleIdx(idx);
  }, [platform]);

  // Cycle to a different sample variant for the current platform.
  const shuffleSample = () => {
    const variants = window.SAMPLE_PRODUCTS?.[platform] || [window.SAMPLE_PRODUCT[platform]];
    if (variants.length <= 1) return;
    let next = sampleIdx;
    while (next === sampleIdx) next = Math.floor(Math.random() * variants.length);
    setSampleIdx(next);
    setProduct(variants[next]);
  };

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

  const setF = (k) => (e) => setProduct(p => ({...p, [k]: e.target.value }));

  async function submit() {
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

  const canSubmit = personaA.trim().length > 4 && product.title.trim().length > 0 && !loading;

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
            <span className="field-help">multiline · prefill with a sample below</span>
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

        {/* Product */}
        <div className="space-y-3">
          <div className="flex items-baseline justify-between gap-3">
            <label className="field-label">Product details</label>
            <div className="flex items-center gap-3">
              <span className="field-help">sample {sampleIdx + 1}/{(window.SAMPLE_PRODUCTS?.[platform] || []).length || 1} for <span className="mono">{platform}</span></span>
              <button
                type="button"
                className="btn btn-ghost btn-xs"
                onClick={shuffleSample}
                disabled={loading || (window.SAMPLE_PRODUCTS?.[platform]?.length || 1) <= 1}
              >Shuffle</button>
            </div>
          </div>
          <div className="flex gap-4 items-start">
            <ProductSlot
              width={140}
              height={140}
              rounded={8}
            />
            <div className="grid grid-cols-12 gap-3 flex-1">
              <div className="col-span-12">
                <label className="input-label">Title</label>
                <input className="input" placeholder="e.g. Anker 7-in-1 USB-C Hub" value={product.title} onChange={setF("title")} />
              </div>
              <div className="col-span-12">
                <label className="input-label">Description</label>
                <textarea className="textarea" style={{minHeight: 72}} placeholder="One or two sentences describing the product" value={product.description} onChange={setF("description")} />
              </div>
              <div className="col-span-6">
                <label className="input-label">Category</label>
                <input className="input" placeholder="e.g. Electronics › Hubs" value={product.category} onChange={setF("category")} />
              </div>
              <div className="col-span-3">
                <label className="input-label">Price</label>
                <input className="input" placeholder="USD" value={product.price} onChange={setF("price")} />
              </div>
              <div className="col-span-3">
                <label className="input-label">Avg rating</label>
                <input className="input" placeholder="0–5" value={product.average_rating} onChange={setF("average_rating")} />
              </div>
            </div>
          </div>
        </div>

        {/* Submit row */}
        <div className="flex items-center justify-between gap-4">
          <ElapsedTimer active={loading} />
          <div className="flex items-center gap-3">
            <button className="btn btn-ghost" onClick={() => {
              setPersonaA("");
              const { variant, idx } = pickRandomSample(platform);
              setProduct(variant); setSampleIdx(idx);
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
