/* global React, Icon */
// Task B — Recommendations (top 5)

function TaskB({ platform, naija, personaB, setPersonaB }) {
  const [loading, setLoading] = React.useState(false);
  const [result, setResult] = React.useState(null);
  const [error, setError] = React.useState(null);
  const outputRef = React.useRef(null);

  React.useEffect(() => {
    if (!loading && !result && !error) return;
    const el = outputRef.current;
    if (!el) return;
    const top = el.getBoundingClientRect().top + window.scrollY - 80;
    window.scrollTo({ top, behavior: "smooth" });
  }, [loading, result, error]);

  async function submit() {
    setLoading(true);
    setError(null);
    setResult(null);
    const r = await window.api("/task_b", {
      user_persona: personaB,
      platform,
      naija_mode: naija,
    });
    setLoading(false);
    if (!r.ok) setError(r.error || "Network error");
    else setResult(r.data);
  }

  const canSubmit = personaB.trim().length > 4 && !loading;

  return (
    <div className="fade-in">
      <SectionHeader
        eyebrow="Task B · Recommendation"
        title="Rank the top 5 items"
        sub="Given a user persona, the agent ranks five items from the selected platform's catalogue and explains each pick."
      />

      <div className="space-y-7">
        <div className="space-y-2.5">
          <div className="flex items-baseline justify-between gap-3">
            <label className="field-label">User persona</label>
            <span className="field-help">multiline · prefill with a sample below</span>
          </div>
          <textarea
            className="textarea"
            placeholder="Describe the user — what they enjoy, where they live, what they've rated highly…"
            value={personaB}
            onChange={(e) => setPersonaB(e.target.value)}
          />
          <div className="pt-1">
            <PersonaChips
              platform={platform}
              value={personaB}
              onPick={(p) => setPersonaB(p)}
            />
          </div>
        </div>

        <div className="flex items-center justify-between gap-4">
          <ElapsedTimer active={loading} />
          <div className="flex items-center gap-3">
            <button className="btn btn-ghost" onClick={() => {
              setPersonaB(""); setResult(null); setError(null);
            }} disabled={loading}>Reset</button>
            <button className="btn btn-primary" onClick={submit} disabled={!canSubmit}>
              {loading ? "Ranking…" : <>Generate recommendations <Icon.ChevronRight size={14}/></>}
            </button>
          </div>
        </div>

        <div ref={outputRef}>
          {loading && <SkeletonRankList />}
          {error && <ErrorToast message={error} onRetry={submit} onDismiss={() => setError(null)} />}
          {result && !loading && <RankedList result={result} naija={naija} platform={platform} />}
        </div>
      </div>
    </div>
  );
}

function RankedList({ result, naija, platform }) {
  return (
    <div className="space-y-3 fade-in">
      <div className="flex items-center justify-between mb-1">
        <div className="mono text-[11px] uppercase tracking-wider text-stone4">Ranked output · top 5</div>
        <div className="flex items-center gap-2">
          <span className="tag" style={{background:"transparent"}}>
            <span className="dot" style={{background:"#C75D3A"}} />
            {result.latency_ms} ms
          </span>
          <span className="tag">
            <span className="dot"/>
            {result.model}
          </span>
        </div>
      </div>
      {result.ranked.map((it) => (
        <RankCard key={it.item_id} item={it} naija={result.naija_mode} platform={platform} />
      ))}
    </div>
  );
}

function RankCard({ item, naija, platform }) {
  return (
    <div className={`card p-5 relative flex items-start gap-5 ${naija ? "naija-edge" : ""}`}>
      {naija && <NaijaCorner />}
      <div className="rank pt-0.5" aria-label={`Rank ${item.rank}`}>{String(item.rank).padStart(2,"0")}</div>
      <window.ProductSlot
        width={64}
        height={64}
        rounded={6}
      />
      <div className="flex-1 min-w-0">
        <div className="flex flex-wrap items-baseline gap-2.5 pr-12">
          <div className="font-display font-semibold text-[17px] leading-tight tracking-tightish text-ink">
            {item.title}
          </div>
          <span className="cat-badge">{item.category}</span>
        </div>
        <div className="text-[14px] leading-[1.55] text-[#41475A] mt-2 max-w-[64ch]"
             dangerouslySetInnerHTML={renderReason(item.reason, naija)} />
        <div className="mt-2.5 mono text-[11px] text-stone4">item_id <span className="text-ink/70">{item.item_id}</span></div>
      </div>
      <Tooltip text="Semantic match between the persona and this item — 0–100, cosine similarity from BGE retrieval.">
        <div className="score cursor-default">
          match&nbsp;
          <span className="text-ink font-semibold">{item.collab_score}</span>
        </div>
      </Tooltip>
    </div>
  );
}

function renderReason(text, naija) {
  if (!naija) return { __html: text };
  // Apply ochre underline to a short phrase. Simple heuristic: underline the first 4-6 word run.
  const phrases = [
    /(go match your bukka vibes)/i,
    /(same kind cables you dey buy)/i,
    /(dey rate am well)/i,
    /(go enter for you sharp sharp)/i,
    /(head go scatter)/i,
    /(plenty pepper)/i,
  ];
  for (const r of phrases) {
    if (r.test(text)) {
      return { __html: text.replace(r, '<span class="pidgin-mark">$1</span>') };
    }
  }
  return { __html: text };
}

window.TaskB = TaskB;
