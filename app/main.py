"""Instinct+ : predict a video's reach from its title and thumbnail.

    streamlit run app/main.py

Tabs: Score (one title + thumbnail, with Claude suggestions), Compare (A/B),
Batch (a CSV of titles), Model card (how the models were trained and how
well they did on held-out channels).
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.charts import driver_bars, gauge, importance_bars  # noqa: E402
from config.schema import CONTENT_TYPES, DATA_DIR  # noqa: E402
from src.generation.titles import DEFAULT_MODEL, claude_available, improve_title  # noqa: E402
from src.scoring.registry import get_scorer, reset  # noqa: E402
from src.scoring.suggestions import THUMBNAIL_LABELS, TITLE_LABELS  # noqa: E402

st.set_page_config(page_title="Instinct+", page_icon="🎯", layout="wide")

ACCENT = "#FF6A3D"
st.markdown(
    f"""
<style>
.block-container {{padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1180px;}}
.stAppDeployButton, #MainMenu, footer {{display: none;}}
[data-testid="stHeader"] {{background: transparent;}}
.stTabs [data-baseweb="tab-list"] {{gap: 2px; border-bottom: 1px solid rgba(128,128,128,.25);}}
.stTabs [data-baseweb="tab"] {{height: 46px; padding: 0 20px; font-size: .95rem; font-weight: 500; letter-spacing: .01em;}}
.stTabs [data-baseweb="tab-highlight"] {{background-color: {ACCENT}; height: 3px;}}
.stTabs [data-baseweb="tab-border"] {{display: none;}}
.stTabs [aria-selected="true"] p {{color: {ACCENT};}}
.stTabs [data-baseweb="tab-panel"] {{padding-top: 1.2rem;}}
.section-label {{font-size: .76rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; opacity: .6; margin: 1.1rem 0 .15rem 0;}}
.section-sub {{font-size: .88rem; opacity: .7; margin: 0 0 .5rem 0;}}
.brand {{font-size: 1.55rem; font-weight: 700; letter-spacing: -.02em; margin: 0 0 .1rem 0;}}
.brand-sub {{font-size: .85rem; opacity: .65; margin: 0 0 1rem 0;}}
h1, h2, h3 {{font-weight: 600; letter-spacing: -.01em;}}
[data-testid="stMetricValue"] {{font-size: 1.5rem;}}
[data-testid="stMetricLabel"] p {{font-size: .78rem; letter-spacing: .06em; text-transform: uppercase; opacity: .7;}}
[data-testid="stExpander"] summary p {{font-weight: 600; font-size: .92rem;}}
div[data-testid="stAlert"] p {{font-size: .9rem;}}
[data-testid="stCaptionContainer"] p {{font-size: .82rem;}}
[data-testid="stVerticalBlockBorderWrapper"] {{border-radius: 16px; border: 1px solid rgba(128,128,128,.22); background: rgba(128,128,128,.06); padding: 4px 6px;}}
.card {{border-radius: 16px; padding: 18px 20px 16px 20px; border: 1px solid rgba(255,255,255,.06);}}
.card-head {{display:flex; align-items:center; gap:8px; font-size:.74rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; opacity:.85;}}
.card-head .dot {{width:10px; height:10px; border-radius:50%; display:inline-block;}}
.gauge-wrap {{display:flex; flex-direction:column; align-items:center; margin-top:10px;}}
.gauge-wrap {{width:100%;}}
.gauge {{width:100%; max-width:220px; aspect-ratio:2/1; border-radius:999px 999px 0 0; position:relative; overflow:hidden;}}
.gauge::after {{content:""; position:absolute; left:13%; top:26%; width:74%; height:148%; border-radius:50%; background:var(--card-bg);}}
.gauge-num {{position:absolute; left:0; right:0; bottom:0; text-align:center; font-size:2.4rem; font-weight:700; line-height:1; z-index:2; letter-spacing:-.02em;}}
.gauge-ticks {{display:flex; justify-content:space-between; width:100%; max-width:220px; font-size:.7rem; opacity:.55; margin-top:4px;}}
.verdict {{margin-top:10px; font-size:.95rem; font-weight:600;}}
.verdict small {{font-weight:400; opacity:.7;}}
.panel-head {{display:flex; align-items:center; gap:10px; margin: 2px 0 8px 0;}}
.panel-head .bar {{width:6px; height:22px; border-radius:3px;}}
.panel-head span.t {{font-size:.8rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase;}}
.panel-head span.s {{font-size:.82rem; opacity:.65;}}
</style>
""",
    unsafe_allow_html=True,
)

CARD = {
    "title": {"accent": "#FF6A3D", "bg": "#2A1D1A"},
    "thumbnail": {"accent": "#FFC857", "bg": "#2A2419"},
    "overall": {"accent": "#5AA9FF", "bg": "#182236"},
}


def band(score: float):
    if score < 35:
        return "Weak", "#E05A5A"
    if score < 65:
        return "Average", "#E3A008"
    return "Strong", "#2E9E5B"


def score_card(key: str, label: str, score, sub: str = ""):
    """A self-contained dashboard tile: colour-coded block, arc gauge, big number, verdict."""
    c = CARD[key]
    if score is None:
        st.markdown(
            f"<div class='card' style='background:{c['bg']};--card-bg:{c['bg']};min-height:232px'>"
            f"<div class='card-head'><span class='dot' style='background:{c['accent']}'></span>{label}</div>"
            f"<div style='opacity:.55;font-size:.95rem;margin-top:70px;text-align:center'>{sub or 'nothing to score yet'}</div></div>",
            unsafe_allow_html=True,
        )
        return
    verdict, vcol = band(score)
    pct = max(0.0, min(100.0, float(score))) / 2.0  # half circle
    st.markdown(
        f"<div class='card' style='background:{c['bg']};--card-bg:{c['bg']}'>"
        f"<div class='card-head'><span class='dot' style='background:{c['accent']}'></span>{label}</div>"
        f"<div class='gauge-wrap'><div class='gauge' style='background:conic-gradient(from 270deg, {c['accent']} 0 {pct}%, rgba(255,255,255,.10) {pct}% 50%, transparent 50%)'>"
        f"<div class='gauge-num' style='color:{c['accent']}'>{score:.0f}</div></div>"
        f"<div class='gauge-ticks'><span>0</span><span>50</span><span>100</span></div></div>"
        f"<div class='verdict' style='color:{vcol}'>{verdict} <small>· {sub}</small></div></div>",
        unsafe_allow_html=True,
    )


def panel_head(key: str, title: str, sub: str = ""):
    c = CARD.get(key, {"accent": ACCENT})
    st.markdown(
        f"<div class='panel-head'><span class='bar' style='background:{c['accent']}'></span>"
        f"<span class='t'>{title}</span><span class='s'>{sub}</span></div>",
        unsafe_allow_html=True,
    )


def section(title: str, sub: str = None):
    """A small, quiet section label instead of a heavy subheader."""
    st.markdown(f"<p class='section-label'>{title}</p>", unsafe_allow_html=True)
    if sub:
        st.markdown(f"<p class='section-sub'>{sub}</p>", unsafe_allow_html=True)


@st.cache_resource
def scorer():
    return get_scorer()


def _manifest_files():
    return sorted((DATA_DIR / "channel_demo").glob("*/manifest.csv"))


@st.cache_data
def _known_videos(stamp: tuple) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in _manifest_files()]
    if not frames:
        return pd.DataFrame(columns=["channel", "video_id", "title", "views", "md5", "filename", "path"])
    return pd.concat(frames, ignore_index=True)


def known_videos() -> pd.DataFrame:
    """Videos checked with scripts/channel_check.py: title, views, kind and the
    md5 of each thumbnail, so an uploaded thumbnail can be recognised and its
    original title filled in. Re-read whenever a manifest changes."""
    return _known_videos(tuple((str(f), f.stat().st_mtime) for f in _manifest_files()))


def decode_image(data: bytes):
    import cv2

    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None


def annotated(img_bgr):
    import cv2

    from src.thumbnail.features import annotate

    return cv2.cvtColor(annotate(img_bgr), cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ sidebar --

with st.sidebar:
    st.markdown("<p class='brand'>Instinct+</p><p class='brand-sub'>Reach prediction for titles and thumbnails</p>", unsafe_allow_html=True)
    content_type = st.selectbox("Content type", CONTENT_TYPES, index=0)
    subs_in = st.number_input("Your subscriber count (optional)", min_value=0, value=0, step=1000,
                              help="Views per subscriber falls as channels grow. The score is corrected for size so titles compare fairly; leave 0 if unknown.")
    subscriber_count = float(subs_in) if subs_in > 0 else None
    use_claude = st.toggle("Use Claude for title suggestions", value=claude_available(),
                           help=f"Needs ANTHROPIC_API_KEY in .env. Model: {DEFAULT_MODEL}")
    if use_claude and not claude_available():
        st.warning("No ANTHROPIC_API_KEY found; suggestions will use rule-based rewrites.")

    st.divider()
    S = scorer()
    section("Models")
    for part, backend, card in (("Title", S.status.title_backend, S.status.title_card),
                                ("Thumbnail", S.status.thumbnail_backend, S.status.thumbnail_card)):
        if backend == "model" and card:
            st.markdown(f"**{part}** · ρ {card['spearman']:.2f} on {card['channels']} held-out channels")
        else:
            st.markdown(f"**{part}**: :red[placeholder] (random numbers)")
    if S.status.data_source == "synthetic":
        st.markdown(":orange[Synthetic training data: a demo of the pipeline.]")
    elif S.status.data_source == "collected":
        st.markdown(":green[Trained on collected YouTube data.]")
    if st.button("Reload models"):
        reset()
        scorer.clear()
        st.rerun()

if not S.status.all_real:
    st.error("One or both models are placeholders. Train them: `python -m src.data.synthetic` (demo) or `python -m src.collection all` then `python -m src.modelling.train`.")
elif S.status.synthetic:
    st.warning("Demo models, trained on synthetic data. Scores show the pipeline working, not YouTube behaviour yet.", icon="🧪")

tab_score, tab_compare, tab_batch, tab_card = st.tabs(["Score", "Compare", "Batch", "Model card"])

# ------------------------------------------------------------------ score --

with tab_score:
    if "title_input" not in st.session_state:
        st.session_state.title_input = ""
    # a chosen suggestion is staged by the Use button and applied here, before
    # the widget exists on this run (Streamlit forbids setting it afterwards)
    if "pending_title" in st.session_state:
        st.session_state.title_input = st.session_state.pop("pending_title")
    known = known_videos()
    stored_thumb = None  # bytes of a stored thumbnail when a video is picked from the menu
    if len(known):
        if "kind" not in known.columns:
            known["kind"] = "long"
        longs = known[known["kind"].isin(["long", "live"])]
        shorts = known[known["kind"] == "short"]
        channels = ", ".join(known["channel"].unique())
        n_live = int((known["kind"] == "live").sum())
        live_txt = f" (incl. {n_live} streams)" if n_live else ""
        with st.expander(f"{channels}  ·  {len(longs)} long-form{live_txt}  ·  {len(shorts)} Shorts", expanded=True):
            auto_fill = st.toggle("Fill in the original title when a known thumbnail is uploaded", value=True, key="auto_fill")
            # a picker reset staged on the previous run, applied before the widgets exist
            if "reset_picker" in st.session_state:
                st.session_state[st.session_state.pop("reset_picker")] = "(none)"

            def fmt(r):
                views = f"{int(r.views):,} views" if pd.notna(r.views) else "views unknown"
                dur = f", {int(r.duration_seconds) // 60}:{int(r.duration_seconds) % 60:02d}" if pd.notna(r.duration_seconds) and r.kind != "short" else ""
                live = "  ·  live stream" if r.kind == "live" else ""
                return f"{r.title}  ·  {views}{dur}{live}"

            c_long, c_short = st.columns(2)
            with c_long:
                section("Long-form")
                long_opts = ["(none)"] + [fmt(r) for r in longs.itertuples()]
                picked_long = st.selectbox("Pick a long-form video", long_opts, key="picked_long", label_visibility="collapsed")
            with c_short:
                section("Shorts")
                short_opts = ["(none)"] + [fmt(r) for r in shorts.itertuples()]
                picked_short = st.selectbox("Pick a Short", short_opts, key="picked_short", label_visibility="collapsed")

            picked_row = None
            for key, opts, frame in (("picked_long", long_opts, longs), ("picked_short", short_opts, shorts)):
                val = st.session_state.get(key, "(none)")
                if val != "(none)" and st.session_state.get("last_picked") != val:
                    picked_row = frame.iloc[opts.index(val) - 1]
                    st.session_state.last_picked = val
                    st.session_state.pending_title = picked_row.title
                    # clear the other picker so only one video is active
                    st.session_state.reset_picker = "picked_short" if key == "picked_long" else "picked_long"
                    st.rerun()
            current = st.session_state.get("last_picked")
            if current and current != "(none)":
                for opts, frame in ((long_opts, longs), (short_opts, shorts)):
                    if current in opts:
                        row = frame.iloc[opts.index(current) - 1]
                        if Path(row.path).exists():
                            stored_thumb = Path(row.path).read_bytes()
                        if row.kind == "short":
                            st.info("This is a Short. Shorts are excluded from the training data by design: they are served by autoplay in a vertical feed, so the title and thumbnail rarely decide the view. The score below is the long-form model applied out of its domain.")
    else:
        auto_fill = False

    # the uploader is created before the title box so a recognised thumbnail
    # can fill the title in on this same run
    col_in, col_img = st.columns([3, 2])
    with col_img:
        upload = st.file_uploader("Thumbnail", type=["jpg", "jpeg", "png", "webp"], help="JPEG or PNG, ideally 1280x720")
        if upload and auto_fill:
            digest = hashlib.md5(upload.getvalue()).hexdigest()
            match = known[known["md5"] == digest]
            if len(match) and st.session_state.get("autofilled_md5") != digest:
                st.session_state.autofilled_md5 = digest
                st.session_state.pending_title = match.iloc[0].title
                st.rerun()
            if len(match):
                m0 = match.iloc[0]
                kind = {"long": "long-form video", "live": "live stream", "short": "Short"}.get(m0.get("kind", "long"), "video")
                views = f"{int(m0.views):,} views" if pd.notna(m0.views) else "views unknown"
                st.caption(f"Recognised {kind} from {m0.channel}, {views}. Title filled in from the original.")
                if m0.get("kind") == "short":
                    st.info("This is a Short: outside what the models were built for. Autoplay feeds decide most Shorts views, not the packaging.")
    with col_in:
        title = st.text_area("Title", key="title_input", height=90, placeholder="Paste a draft title…")

    thumb_bytes = upload.getvalue() if upload else stored_thumb
    img = decode_image(thumb_bytes) if thumb_bytes else None
    title_result = S.score_title(title, content_type, subscriber_count) if title.strip() else None
    thumb_result = S.score_thumbnail_bytes(thumb_bytes, content_type, subscriber_count) if thumb_bytes else None
    if stored_thumb and not upload:
        st.caption("Thumbnail loaded from the channel check; upload one to override it.")
    overall = S.combine(title_result, thumb_result) if (title_result or thumb_result) else None

    if overall is None:
        st.info("Enter a title and/or upload a thumbnail to get a score.")
    else:
        g1, g2, g3 = st.columns(3)
        with g1:
            score_card("title", "Title", title_result.score if title_result else None,
                       f"{len(title_result.drivers)} drivers" if title_result else "enter a title")
        with g2:
            has_thumb = thumb_result and not thumb_result.meta.get("image_missing")
            score_card("thumbnail", "Thumbnail", thumb_result.score if has_thumb else None,
                       f"{int(thumb_result.features.get('face_count', 0))} face(s), {thumb_result.features.get('text_area', 0):.0%} text" if has_thumb else "upload one")
        with g3:
            parts = overall.meta.get("parts", [])
            score_card("overall", "Overall", overall.score if parts else None,
                       "title + thumbnail" if len(parts) == 2 else f"{parts[0]} only" if parts else "")
        w = overall.meta.get("weights", {})
        if len(w) == 2:
            st.caption(f"Overall = title {w['title']:.0%} + thumbnail {w['thumbnail']:.0%}, weighted by each model's validated skill. 70 means better than 70% of this category.")
        if title_result and title_result.meta.get("context_points"):
            ctx = title_result.meta["context_points"].get("log_subscribers")
            if ctx is not None and subscriber_count:
                st.caption(f"Channel-size effect held out of the score: {ctx:+.0f} points for a channel of {int(subscriber_count):,} subscribers.")

        c1, c2 = st.columns(2)
        with c1, st.container(border=True):
            panel_head("title", "Title", "what moved the score")
            if title_result:
                if title_result.drivers:
                    st.plotly_chart(driver_bars(title_result.drivers, TITLE_LABELS), key="d_title")
                for n in title_result.notes:
                    st.markdown(f"- {n}")
                with st.expander("All title features"):
                    st.dataframe(pd.Series(title_result.features, name="value").to_frame(), height=300)
            else:
                st.caption("No title yet.")
        with c2, st.container(border=True):
            panel_head("thumbnail", "Thumbnail", "what the model saw")
            if thumb_result and img is not None and not thumb_result.meta.get("image_missing"):
                st.image(annotated(img), caption="Faces in green, text in orange", width="stretch")
                if thumb_result.drivers:
                    st.plotly_chart(driver_bars(thumb_result.drivers, THUMBNAIL_LABELS), key="d_thumb")
                for n in thumb_result.notes:
                    st.markdown(f"- {n}")
                with st.expander("All thumbnail features"):
                    st.dataframe(pd.Series(thumb_result.features, name="value").to_frame(), height=300)
            elif thumb_result:
                for n in thumb_result.notes:
                    st.markdown(f"- {n}")
            else:
                st.caption("No thumbnail yet.")

        if title_result:
            with st.container(border=True):
                panel_head("overall", "Better titles", "generated, scored by the same model, ranked")
                if st.button("Suggest titles", type="primary"):
                    with st.spinner("Generating and scoring alternatives..."):
                        st.session_state.suggestions = improve_title(title, content_type, S, subscriber_count, n=6, use_claude=use_claude)
                sugg = st.session_state.get("suggestions")
                if sugg and sugg["original"].title == title:
                    if sugg["error"]:
                        st.warning(sugg["error"])
                    st.caption(f"Source: {'Claude · ' + sugg['model'] if sugg['source'] == 'claude' else 'rule-based rewrites'}")
                    for i, c in enumerate(sugg["candidates"]):
                        a, b, c3 = st.columns([6, 1, 1])
                        with a:
                            st.markdown(f"**{c.title}**  \n<small>{c.rationale}</small>", unsafe_allow_html=True)
                        with b:
                            st.metric("score", f"{c.score:.0f}", f"{c.delta:+.0f}")
                        with c3:
                            if st.button("Use", key=f"use_{i}"):
                                st.session_state.pending_title = c.title
                                st.session_state.pop("suggestions", None)
                                st.rerun()

# ---------------------------------------------------------------- compare --

with tab_compare:
    section("Compare two drafts", "Same model, same category, same channel size.")
    ca, cb = st.columns(2)
    variants = {}
    for name, col in (("A", ca), ("B", cb)):
        with col:
            section(f"Variant {name}")
            t = st.text_input(f"Title {name}", key=f"cmp_title_{name}")
            u = st.file_uploader(f"Thumbnail {name}", type=["jpg", "jpeg", "png", "webp"], key=f"cmp_img_{name}")
            tr = S.score_title(t, content_type, subscriber_count) if t.strip() else None
            ir = S.score_thumbnail_bytes(u.getvalue(), content_type, subscriber_count) if u else None
            ov = S.combine(tr, ir) if (tr or ir) else None
            variants[name] = (tr, ir, ov)
            if ov:
                m1, m2, m3 = st.columns(3)
                m1.metric("Title", f"{tr.score:.0f}" if tr else "–")
                m2.metric("Thumbnail", f"{ir.score:.0f}" if ir and not ir.meta.get('image_missing') else "–")
                m3.metric("Overall", f"{ov.score:.0f}")
                if u and not (ir and ir.meta.get("image_missing")):
                    st.image(annotated(decode_image(u.getvalue())), width="stretch")
    a, b = variants["A"][2], variants["B"][2]
    if a and b:
        diff = a.score - b.score
        winner = "A" if diff > 0 else "B"
        st.success(f"Variant **{winner}** leads by **{abs(diff):.0f}** points overall." if abs(diff) >= 1 else "Too close to call: under a point apart.")
        ta, tb = variants["A"][0], variants["B"][0]
        if ta and tb:
            keys = sorted(set(ta.drivers) | set(tb.drivers), key=lambda k: -abs(ta.drivers.get(k, 0) - tb.drivers.get(k, 0)))
            rows = [{"feature": TITLE_LABELS.get(k, k), "A": ta.drivers.get(k, 0.0), "B": tb.drivers.get(k, 0.0), "A − B": ta.drivers.get(k, 0.0) - tb.drivers.get(k, 0.0)} for k in keys[:10]]
            section("What separates the titles", "Driver difference in points.")
            st.dataframe(pd.DataFrame(rows).round(0).astype({"A": int, "B": int, "A − B": int}), hide_index=True)

# ------------------------------------------------------------------ batch --

with tab_batch:
    section("Score many titles", "One per line, or a CSV with a title column (optional content_type, subscriber_count).")
    pasted = st.text_area("Titles, one per line", height=140, key="batch_text")
    csv_up = st.file_uploader("...or a CSV", type=["csv"], key="batch_csv")
    frame = None
    if csv_up:
        frame = pd.read_csv(csv_up)
        if "title" not in frame.columns:
            st.error("The CSV needs a `title` column.")
            frame = None
    elif pasted.strip():
        frame = pd.DataFrame({"title": [l.strip() for l in pasted.splitlines() if l.strip()]})
    if frame is not None and len(frame):
        if "content_type" not in frame.columns:
            frame["content_type"] = content_type
        if "subscriber_count" not in frame.columns:
            frame["subscriber_count"] = subscriber_count
        out = []
        for _, r in frame.iterrows():
            subs = r["subscriber_count"] if pd.notna(r["subscriber_count"]) and r["subscriber_count"] else None
            res = S.score_title(str(r["title"]), str(r["content_type"]), subs)
            top = sorted(res.drivers.items(), key=lambda kv: -abs(kv[1]))[:3]
            out.append({"title": r["title"], "content_type": r["content_type"], "score": int(round(res.score)),
                        "top drivers": ", ".join(f"{TITLE_LABELS.get(k, k)} {v:+.0f}" for k, v in top),
                        "first note": res.notes[0] if res.notes else ""})
        result = pd.DataFrame(out).sort_values("score", ascending=False)
        st.dataframe(result, hide_index=True, height=min(600, 40 + 36 * len(result)))
        st.download_button("Download scored CSV", result.to_csv(index=False).encode("utf-8"), "instinct_scores.csv", "text/csv")

# ------------------------------------------------------------- model card --

with tab_card:
    for part, card, labels in (("Title model", S.status.title_card, TITLE_LABELS), ("Thumbnail model", S.status.thumbnail_card, THUMBNAIL_LABELS)):
        section(part)
        if not card:
            st.caption("Not trained.")
            continue
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Spearman ρ", f"{card['spearman']:.2f}", help="rank correlation between predicted and true reach on held-out channels")
        m2.metric("R²", f"{card['r2']:.2f}")
        m3.metric("MAE vs baseline", f"{card['mae']:.2f}", f"{card['mae'] - card['baseline_mae']:+.2f}", delta_color="inverse")
        m4.metric("Videos", f"{card['rows']:,}", help=f"from {card['channels']} channels, {card['folds']}-fold grouped CV")
        st.caption(f"Data: **{card['data_source']}** · trained {card['trained_at']} · backend {card['backend']} · features {card['feature_version']} · data hash {card['data_hash']}")
        per = pd.DataFrame(card["per_content_type"]).T
        if len(per):
            st.dataframe(per.round(2))
        st.plotly_chart(importance_bars(card["importance"], labels), key=f"imp_{part}")
        for n in card.get("notes", []):
            st.warning(n)
    section("How to read a score")
    st.markdown(
        "The model predicts age-adjusted log(views ÷ subscribers). The score is the percentile of that prediction among "
        "the model's own out-of-fold predictions for the same content type, with the channel-size contribution removed. "
        "Drivers are SHAP contributions in score points. Details: `docs/validation_protocol.md`."
    )
