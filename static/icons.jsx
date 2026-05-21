/* global React */
// Tiny line-icon set (lucide-style strokes, 1.6px). Charcoal by default.

const S = ({ size=16, stroke=1.6, children, className="" }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round"
    className={className} aria-hidden="true">{children}</svg>
);

const Icon = {
  Sparkles: (p) => <S {...p}><path d="M12 3v4"/><path d="M12 17v4"/><path d="M3 12h4"/><path d="M17 12h4"/><path d="M5.6 5.6l2.8 2.8"/><path d="M15.6 15.6l2.8 2.8"/><path d="M5.6 18.4l2.8-2.8"/><path d="M15.6 8.4l2.8-2.8"/></S>,
  Chevron: (p) => <S {...p}><path d="M6 9l6 6 6-6"/></S>,
  ChevronRight: (p) => <S {...p}><path d="M9 6l6 6-6 6"/></S>,
  Help:     (p) => <S {...p}><circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 0 1 5 0c0 1.6-2.5 2.2-2.5 4"/><path d="M12 17h.01"/></S>,
  Refresh:  (p) => <S {...p}><path d="M3 12a9 9 0 0 1 15.5-6.3L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-15.5 6.3L3 16"/><path d="M3 21v-5h5"/></S>,
  X:        (p) => <S {...p}><path d="M6 6l12 12"/><path d="M18 6L6 18"/></S>,
  Star:     (p) => <S {...p}><path d="M12 3l2.7 5.6 6.2.9-4.5 4.3 1.1 6.1L12 17l-5.5 2.9 1.1-6.1L3 9.5l6.2-.9z" fill="currentColor" stroke="currentColor"/></S>,
  StarOutline: (p) => <S {...p}><path d="M12 3l2.7 5.6 6.2.9-4.5 4.3 1.1 6.1L12 17l-5.5 2.9 1.1-6.1L3 9.5l6.2-.9z"/></S>,
  Bolt:     (p) => <S {...p}><path d="M13 3L4 14h7l-1 7 9-11h-7z"/></S>,
  AlertTri: (p) => <S {...p}><path d="M12 4l9.5 16.5h-19z"/><path d="M12 10v5"/><path d="M12 18h.01"/></S>,
  Split:    (p) => <S {...p}><path d="M16 3h5v5"/><path d="M4 20l16-16"/><path d="M21 16v5h-5"/><path d="M15 15l6 6"/><path d="M4 4l5 5"/></S>,
  Tag:      (p) => <S {...p}><path d="M20.5 12.5l-8 8a2 2 0 0 1-2.8 0L3 13.8V4h9.8l7.7 7.7a2 2 0 0 1 0 2.8z"/><circle cx="7.5" cy="7.5" r="1.2" fill="currentColor"/></S>,
  Search:   (p) => <S {...p}><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></S>,
  Logo:     (p) => (
    <S {...p}>
      <path d="M12 3l9 5v8l-9 5-9-5V8z" />
      <path d="M12 3v18" />
      <path d="M3 8l9 5 9-5" />
    </S>
  ),
  Eye:      (p) => <S {...p}><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></S>,
};

window.Icon = Icon;
