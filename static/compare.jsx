/* global React, Icon */
// Stretch — side-by-side comparison view. Fires both Naija OFF and Naija ON calls in parallel.

function CompareView({ task, platform, persona, product, onClose }) {
  const [loading, setLoading] = React.useState(true);
  const [neutral, setNeutral] = React.useState(null);
  const [naija, setNaija] = React.useState(null);
  const [error, setError] = React.useState(null);

  async function run() {
    setLoading(true); setNeutral(null); setNaija(null); setError(null);
    const endpoint = task === "a" ? "/task_a" : "/task_b";
    const baseBody = task === "a"
      ? { user_persona: persona, product, platform }
      : { user_persona: persona, platform };
    const [a, b] = await Promise.all([
      window.api(endpoint, {...baseBody, naija_mode: false}),
      window.api(endpoint, {...baseBody, naija_mode: true}),
    ]);
    setLoading(false);
    if (!a.ok || !b.ok) setError("One of the calls failed.");
    else { setNeutral(a.data); setNaija(b.data); }
  }

  React.useEffect(() => { run(); /* eslint-disable-line */ }, []);

  return (
    <div className="fixed inset-0 z-40 bg-ink/30 backdrop-blur-[2px] flex items-start justify-center overflow-auto" onClick={onClose}>
      <div className="w-full max-w-[1100px] mx-auto my-10 px-6" onClick={e => e.stopPropagation()}>
        <div className="card p-7 fade-in">
          <div className="flex items-start justify-between gap-4 mb-6">
            <div>
              <div className="mono text-[11px] uppercase tracking-wider text-stone4 mb-1.5">Side-by-side</div>
              <h2 className="font-display font-bold text-[26px] tracking-tighter2">Neutral vs. Naija mode</h2>
              <p className="text-[14px] text-[#54596A] mt-1.5">
                Same persona &amp; {task === "a" ? "product" : "platform"} — only the output style overlay changes.
              </p>
            </div>
            <button className="btn btn-ghost" onClick={onClose}><Icon.X size={14}/> Close</button>
          </div>

          {error && <ErrorToast message={error} onRetry={run} onDismiss={() => setError(null)} />}

          <div className="compare-grid">
            <CompareCol label="Neutral output" naija={false} loading={loading} data={neutral} task={task} />
            <CompareCol label="Naija mode output" naija={true} loading={loading} data={naija} task={task} />
          </div>
        </div>
      </div>
    </div>
  );
}

function CompareCol({ label, naija, loading, data, task }) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <div className="mono text-[11px] uppercase tracking-wider text-stone4">{label}</div>
        {naija && (
          <span className="inline-flex items-center gap-1 text-emerald9 text-[12px] font-semibold">
            <span className="w-2 h-2 rounded-full bg-emerald9"></span> on
          </span>
        )}
      </div>
      {loading && (task === "a" ? <SkeletonResultCard /> : <SkeletonRankList />)}
      {!loading && data && task === "a" && <MiniResultA data={data} />}
      {!loading && data && task === "b" && (
        <div className="space-y-2.5">
          {data.ranked.slice(0,5).map(it => <MiniRankCard key={it.item_id} item={it} naija={data.naija_mode} />)}
        </div>
      )}
    </div>
  );
}

function MiniResultA({ data }) {
  return (
    <div className={`card p-5 relative ${data.naija_mode ? "naija-edge" : ""}`}>
      {data.naija_mode && <NaijaCorner />}
      <div className="flex items-end gap-3 mb-3">
        <Stars value={Math.round(data.stars)} size={22} />
        <div className="font-display font-bold text-[20px] leading-none tracking-tighter2">
          {Number(data.stars).toFixed(2)}<span className="text-stone4 font-medium text-[14px]"> /5</span>
        </div>
      </div>
      <div className="font-serif text-[15px] leading-[1.6] text-ink"
        dangerouslySetInnerHTML={window.renderReview(data.review, !!data.naija_mode)} />
      <div className="mt-4 mono text-[11px] text-stone4">{data.latency_ms} ms · {data.model}</div>
    </div>
  );
}

function MiniRankCard({ item, naija }) {
  return (
    <div className={`card p-4 relative flex items-start gap-4 ${naija ? "naija-edge" : ""}`}>
      {naija && <div className="naija-corner" style={{transform:"scale(.85)", transformOrigin:"top right"}}><span className="flag"><i></i><i></i><i></i></span><span>NG</span></div>}
      <div className="rank text-[18px]">{String(item.rank).padStart(2,"0")}</div>
      <div className="flex-1 min-w-0 pr-10">
        <div className="font-display font-semibold text-[14.5px] tracking-tightish leading-tight">{item.title}</div>
        <div className="mt-1"><span className="cat-badge">{item.category}</span></div>
        <div className="text-[13px] text-[#41475A] mt-1.5">{item.reason}</div>
      </div>
      <div className="score">{item.collab_score}</div>
    </div>
  );
}

window.CompareView = CompareView;
