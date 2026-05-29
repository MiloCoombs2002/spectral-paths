"""Render a Markdown report to PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

from markdown_pdf import MarkdownPdf, Section


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to the input Markdown file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to the output PDF. Defaults to the Markdown path with a .pdf suffix.",
    )
    parser.add_argument(
        "--paper-size",
        default="A4",
        help="Paper size passed through to markdown-pdf. Default: A4.",
    )
    return parser


def render_pdf(markdown_path: Path, pdf_path: Path, paper_size: str) -> None:
    """Render the Markdown file to a PDF with local asset support."""
    markdown_text = markdown_path.read_text(encoding="utf-8")
    pdf = MarkdownPdf(toc_level=2)
    pdf.meta["title"] = markdown_path.stem.replace("_", " ")
    pdf.add_section(
        Section(
            markdown_text,
            toc=False,
            root=str(markdown_path.parent),
            paper_size=paper_size,
        )
    )
    pdf.save(str(pdf_path))


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    markdown_path = args.input.resolve()
    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

    pdf_path = (
        args.output.resolve()
        if args.output is not None
        else markdown_path.with_suffix(".pdf").resolve()
    )
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    render_pdf(markdown_path, pdf_path, paper_size=args.paper_size)
    print(f"Wrote PDF to {pdf_path}")


if __name__ == "__main__":
    main()
