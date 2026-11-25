# Publication review package

Status: prepared for owner review; website and manuscript publication have not occurred.

- Website source: `JDAppleseed/EEG_HAND_WEBSITE`, branch `main`.
- Proposed post: copy the adjacent MDX file to `content/updates/` after review.
- Keep the featured March model and existing poster unchanged; new runs are unpromoted.
- After applying the post, run the website's `check:runs`, `generate:viewer-summaries`, `generate:seo`, and build commands; inspect the rendered update before Firebase deployment.
- Local manuscript: `paper/research_paper.tex`, with generated `finalization_20261125.tex` and compiled PDF. The November comparison is primary; September ensembles remain explicitly archived.
- The manuscript remains a draft. Author confirmation of the actual acquisition protocol, ethics statements, and cohort descriptions is still needed before external submission. This cleanup does not invent those details.
- GitHub publication includes source, baseline presets, aggregate data/provenance, cleanup review, and rewrite requirements. New participant-level data and model outputs remain local.

No website deployment, manuscript upload, trimming, or separate rewrite is included in the GitHub push.

## Requested release order

1. Merge research PR #24, including the restored full-corpus evaluator and
   preserved historical headline registry.
2. Review the website branch `update/november-evaluation-context-20261125` in
   `/Users/jonathandavanzo/ISEF Robot Hand Code/EEG_HAND_WEBSITE`. It adds the dated
   comparison, five downloadable full-corpus reports, and a results-page link;
   the existing March/May headline metric bundles stay unchanged. Website release
   comes first. Preparation/building is not Firebase deployment.
3. After the website update is finalized, align the private manuscript: preserve
   original headline values with original denominators, add the full-corpus table
   as development-fit evidence, retain the November holdout table separately, and
   distinguish classifier accuracy from gated command precision/event hit rate.
   Cite the website release and merged research commit. Rebuild/render the PDF
   after those edits; do not treat current preparation as manuscript publication.
