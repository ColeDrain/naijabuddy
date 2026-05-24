/* global React, ReactDOM, Icon, TaskA, TaskB, CompareView */

// ----- Tweaks (persisted) -----
const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "teamName": "Team NaijaBuddy",
  "showHeaderPattern": true,
  "headerAccent": "emerald"
}/*EDITMODE-END*/;

function Header({ tweaks, naija, setNaija, platform, setPlatform }) {
  return (
    <header className="border-b border-stone2 bg-paper">
      <div
        className={`relative ${tweaks.showHeaderPattern ? "adire-band" : ""}`}
        style={{
          backgroundColor: tweaks.showHeaderPattern ? "rgba(0,86,59,0.03)" : "transparent",
        }}
      >
        <div className="max-w-[1080px] mx-auto px-8 py-6 flex items-center justify-between gap-x-6 gap-y-4 flex-wrap relative">
          <div className="flex items-center gap-3.5 min-w-0 flex-shrink">
            <LogoMark accent={tweaks.headerAccent}/>
            <div className="min-w-0">
              <div className="font-display font-bold text-[18px] leading-[1.05] tracking-tightish text-ink whitespace-nowrap">
                DSN<span className="text-terracotta">×</span>BCT LLM Agent Challenge
              </div>
              <div className="text-[12.5px] text-[#6B6F7A] mt-0.5 truncate">
                {tweaks.teamName} <span className="text-stone3">·</span> DSN<span className="text-terracotta">×</span>BCT 2026 submission
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4 flex-shrink-0 flex-wrap gap-y-2">
            <NaijaToggle on={naija} onChange={setNaija}/>
            <div className="flex items-center gap-1 flex-shrink-0">
              <PlatformSelect value={platform} onChange={setPlatform}/>
              <DatasetPopover platform={platform} />
            </div>
            <GithubLink />
          </div>
        </div>
      </div>
    </header>
  );
}

function LogoMark({ accent="emerald" }) {
  // Compact geometric mark — chevron + diamond, no figurative motifs.
  const c1 = accent === "terracotta" ? "#C75D3A" : "#00563B";
  return (
    <div className="w-9 h-9 rounded-md grid place-items-center bg-ink text-paper relative overflow-hidden">
      <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 3 L21 12 L12 21 L3 12 Z" stroke={c1}/>
        <path d="M12 8 L16 12 L12 16 L8 12 Z" />
      </svg>
    </div>
  );
}

function GithubLink() {
  const url = "https://github.com/your-team/dsn-bct-llm-agent";
  return (
    <Tooltip text={url}>
      <a
        href={url}
        target="_blank"
        rel="noreferrer noopener"
        aria-label="Project repository on GitHub"
        className="w-9 h-9 grid place-items-center rounded-md text-ink hover:bg-stone1 transition-colors"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
          <path d="M12 .5a11.5 11.5 0 0 0-3.64 22.42c.58.1.79-.25.79-.56v-2.02c-3.2.7-3.88-1.36-3.88-1.36-.52-1.32-1.28-1.67-1.28-1.67-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.2 1.77 1.2 1.03 1.77 2.7 1.26 3.36.96.1-.74.4-1.26.73-1.55-2.56-.29-5.25-1.28-5.25-5.69 0-1.26.45-2.28 1.18-3.09-.12-.29-.51-1.46.11-3.04 0 0 .97-.31 3.18 1.18a11.04 11.04 0 0 1 5.78 0c2.21-1.5 3.18-1.18 3.18-1.18.63 1.58.24 2.75.12 3.04.74.81 1.18 1.83 1.18 3.09 0 4.42-2.7 5.4-5.27 5.68.41.35.78 1.05.78 2.12v3.14c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5z"/>
        </svg>
      </a>
    </Tooltip>
  );
}

function Tabs({ active, onChange, onCompare, compareDisabled, compareHint }) {
  return (
    <div className="border-b border-stone2 bg-paper sticky top-0 z-20">
      <div className="max-w-[1080px] mx-auto px-8 flex items-center justify-between">
        <div className="flex items-center gap-8">
          <button data-active={active === "a"} className="tab-btn" onClick={() => onChange("a")}>
            Task A · User Modeling
          </button>
          <button data-active={active === "b"} className="tab-btn" onClick={() => onChange("b")}>
            Task B · Recommendation
          </button>
        </div>
        <div className="py-2.5">
          <Tooltip text={compareHint}>
            <button className="btn btn-ghost text-[13px] py-2" onClick={onCompare} disabled={compareDisabled}>
              <Icon.Split size={13}/>
              Compare neutral vs Naija
            </button>
          </Tooltip>
        </div>
      </div>
    </div>
  );
}

function Footer() {
  return (
    <footer className="border-t border-stone2 mt-20">
      <div className="max-w-[1080px] mx-auto px-8 py-7 flex items-center justify-center text-[12.5px]">
        <div className="mono text-[11px] text-stone4">
          /task_a · /task_b · /health
        </div>
      </div>
    </footer>
  );
}

function App() {
  const [tweaks, setTweak] = window.useTweaks(TWEAK_DEFAULTS);

  const [tab, setTab] = React.useState("a");
  const [naija, setNaija] = React.useState(false);
  const [platform, setPlatform] = React.useState("amazon");

  // Persist persona text per-tab so it survives tab switches but reset on platform change
  const [personaA, setPersonaA] = React.useState("");
  const [personaB, setPersonaB] = React.useState("");
  const [productA, setProductA] = React.useState(window.SAMPLE_PRODUCT.amazon);
  React.useEffect(() => { setPersonaA(""); setPersonaB(""); }, [platform]);

  const [compareOpen, setCompareOpen] = React.useState(false);
  const activePersona = tab === "a" ? personaA : personaB;
  const compareDisabled = activePersona.trim().length < 5;
  const compareHint = compareDisabled
    ? "Fill in the persona first, then run a side-by-side."
    : "Fires both calls in parallel — neutral and Naija mode side by side.";

  return (
    <>
      <Header
        tweaks={tweaks}
        naija={naija} setNaija={setNaija}
        platform={platform} setPlatform={setPlatform}
      />
      <Tabs
        active={tab} onChange={setTab}
        onCompare={() => setCompareOpen(true)}
        compareDisabled={compareDisabled}
        compareHint={compareHint}
      />

      <main className="max-w-[880px] mx-auto px-8 pt-12 pb-16">
        {tab === "a" && (
          <TaskA
            platform={platform}
            naija={naija}
            personaA={personaA}
            setPersonaA={setPersonaA}
          />
        )}
        {tab === "b" && (
          <TaskB
            platform={platform}
            naija={naija}
            personaB={personaB}
            setPersonaB={setPersonaB}
          />
        )}
      </main>

      <Footer/>

      {compareOpen && (
        <CompareView
          task={tab}
          platform={platform}
          persona={activePersona}
          product={tab === "a" ? productA : null}
          onClose={() => setCompareOpen(false)}
        />
      )}

      <window.TweaksPanel title="Tweaks">
        <window.TweakSection label="Identity">
          <window.TweakText label="Team name" value={tweaks.teamName} onChange={(v) => setTweak("teamName", v)} />
        </window.TweakSection>
        <window.TweakSection label="Header">
          <window.TweakToggle label="Adire watermark band" value={tweaks.showHeaderPattern} onChange={(v) => setTweak("showHeaderPattern", v)} />
          <window.TweakRadio label="Logo accent" value={tweaks.headerAccent} options={["emerald","terracotta"]} onChange={(v) => setTweak("headerAccent", v)} />
        </window.TweakSection>
        <window.TweakSection label="Demo state">
          <window.TweakButton
            label={`Toggle Naija mode (${naija ? "ON" : "OFF"})`}
            onClick={() => setNaija(n => !n)}
          />
          <window.TweakSelect label="Platform" value={platform} onChange={setPlatform}
            options={window.PLATFORMS.map(p => ({value: p.id, label: p.label}))} />
        </window.TweakSection>
      </window.TweaksPanel>
    </>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
