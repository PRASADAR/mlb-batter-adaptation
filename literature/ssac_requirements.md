# SSAC 2027 submission requirements

Checked 2026-09-21 against the [official MIT Sloan Sports Analytics Conference Research Paper Competition page](https://www.sloansportsconference.com/research-paper-competition). The table below distinguishes published rules from our production choices. Recheck the portal before submission because the web page can change.

## Verified official requirements

| Item | Published requirement |
|---|---|
| Abstract deadline | October 1, 2026, 11:59 p.m. Eastern Time. |
| Length | Fewer than 500 words, including title and body. |
| Sections | Introduction, Methods, Results, Conclusion. |
| Evidence | Actual results with relevant statistics, not promised analyses. |
| Visuals | At most two figures/tables combined. |
| Review criteria | Novelty, academic rigor/validity, reproducibility, application; the page also emphasizes impact. |
| Open research | Link to an open repository containing data used in the research. Model code is encouraged. |
| Track | Baseball. |
| Subsequent deadline | If invited, full manuscript due December 4, 2026, 11:59 p.m. Eastern. |

The page's introductory deadline says “EST,” while its timeline says Eastern Time. Use `America/New_York` for planning and submit early. The open-source section retains an older “SSAC 2025” label, but the current SSAC27 introduction explicitly repeats the open-source requirement.

## Project choices, not verified competition mandates

Use US Letter, professional typography, an author placeholder until authorship is supplied, exact Markdown-to-PDF text agreement, deterministic builds, dependency pins, a checksummed analytic-data snapshot, and reproducible notebooks. These choices satisfy the user brief; the inspected official page does not specify a mandatory font, page size, author/anonymity policy, PDF file size, or figure-caption word-count treatment.

For conservative counting, include title, section headings, all body text, and any captions or table text in the project's reported abstract word count. Aim comfortably below 500 rather than depending on portal tokenization. Do not fill numerical claims before the analysis has run.

An unpublished local repository is not a live submission link. A downloader alone is insufficient for this project's data-availability target: include the final analytic data or link a durable downloadable artifact with version/checksum and provenance. Public accessibility and permission to redistribute are separate questions; retain upstream licenses and attribution.

The exact current upload fields, submission-format constraints, and submission success have not been verified. This repository can be prepared for submission without implying it has been submitted or accepted. Publication should be recorded only after the repository URL and data links are reachable.
