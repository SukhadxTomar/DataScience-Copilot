#!/usr/bin/env python
"""Render INTERVIEW_GUIDE_COMBINED.md -> DataScience_Copilot_Interview_Guide.pdf.

Backends, best first:
  1. `markdown` library -> HTML   (nicer output, used if installed)
  2. built-in minimal md->HTML     (fallback, no dependencies)
Rendering is done with PyMuPDF's Story engine, which is confirmed installed,
so this always produces a PDF with no extra installs.
"""
import os
import re
import html

HERE = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(HERE, "INTERVIEW_GUIDE_COMBINED.md")
PDF_PATH = os.path.join(HERE, "DataScience_Copilot_Interview_Guide.pdf")

CSS = """
body { font-family: sans-serif; color: #111; }
h1 { font-size: 21px; color: #14213d; margin: 20px 0 6px; }
h2 { font-size: 15px; color: #1b263b; margin: 16px 0 4px;
     border-bottom: 1px solid #cbd5e1; padding-bottom: 2px; }
h3 { font-size: 13px; color: #0f3460; margin: 13px 0 3px; }
p  { font-size: 10.5px; line-height: 1.5; margin: 5px 0; }
li { font-size: 10.5px; line-height: 1.45; margin: 3px 0; }
blockquote { font-size: 10px; color: #475569; border-left: 3px solid #94a3b8;
     margin: 6px 0; padding: 3px 9px; background: #f1f5f9; }
code { font-family: monospace; background: #eef2f7; font-size: 9.5px; padding: 0 2px; }
hr { border: none; border-top: 1px solid #cbd5e1; margin: 10px 0; }
"""


def inline(t):
    """Inline markdown -> HTML (bold, code, italic), on escaped text."""
    t = html.escape(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+?)`", r"<code>\1</code>", t)
    t = re.sub(r"(?<![\*\w])\*([^*]+?)\*(?![\*\w])", r"<i>\1</i>", t)
    return t


def minimal_md_to_html(md):
    out, quote, mode = [], [], None  # mode: None | 'ul' | 'ol'

    def flush_quote():
        if quote:
            out.append("<blockquote>" + "<br/>".join(quote) + "</blockquote>")
            quote.clear()

    def close_list():
        nonlocal mode
        if mode:
            out.append(f"</{mode}>")
            mode = None

    def close_all():
        flush_quote()
        close_list()

    for raw in md.split("\n"):
        stripped = raw.strip()
        if not stripped:
            close_all()
            continue
        m = re.match(r"(#{1,6})\s+(.*)", stripped)
        if m:
            close_all()
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            continue
        if re.match(r"^-{3,}$", stripped):
            close_all()
            out.append("<hr/>")
            continue
        if stripped.startswith(">"):
            close_list()
            quote.append(inline(stripped.lstrip(">").strip()))
            continue
        flush_quote()
        indented = raw[:1] in (" ", "\t")
        mo = re.match(r"^\d+\.\s+(.*)", raw)
        mu = re.match(r"^[-*]\s+(.*)", raw)
        if mo and not indented:
            if mode != "ol":
                close_list()
                out.append("<ol>")
                mode = "ol"
            out.append(f"<li>{inline(mo.group(1))}</li>")
            continue
        if mu and not indented:
            if mode != "ul":
                close_list()
                out.append("<ul>")
                mode = "ul"
            out.append(f"<li>{inline(mu.group(1))}</li>")
            continue
        if indented and mode and out and out[-1].endswith("</li>"):
            out[-1] = out[-1][:-5] + "<br/>" + inline(stripped) + "</li>"
            continue
        close_list()
        out.append(f"<p>{inline(stripped)}</p>")
    close_all()
    return "\n".join(out)


def build_html(md):
    try:
        import markdown
        body = markdown.markdown(md, extensions=["extra", "sane_lists"])
        used = "markdown-lib"
    except Exception:
        body = minimal_md_to_html(md)
        used = "builtin-converter"
    return f"<div>{body}</div>", used


def render(html_body):
    import pymupdf

    story = pymupdf.Story(html=html_body, user_css=CSS)
    writer = pymupdf.DocumentWriter(PDF_PATH)
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (54, 54, -54, -54)
    pages, more = 0, 1
    while more:
        dev = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
        pages += 1
    writer.close()
    return pages


if __name__ == "__main__":
    with open(MD_PATH, encoding="utf-8") as f:
        md = f.read()
    body, used = build_html(md)
    pages = render(body)
    print(f"OK -> {os.path.basename(PDF_PATH)}  ({pages} pages, via {used})")
