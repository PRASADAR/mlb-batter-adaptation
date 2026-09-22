# Publish the existing GitHub Desktop repository

This project is in the local `mlb-batter-adaptation` repository opened in GitHub Desktop. Its finished commit contains the frozen public data, source manifests, checksums, executed notebook, figures, posterior draws, tests, pinned environment, and SSAC abstract PDF. The code is MIT licensed; MLB source data retains its own rights.

GitHub Desktop currently shows **Publish repository**, which means this repository has no remote yet. Use that control after reviewing the commit. Set the repository name to `mlb-batter-adaptation` and make it public to satisfy the conference's open-source rule. GitHub Desktop can then push subsequent changes to the same repository. Do not create a second `deja-swing` repository.

After publishing, verify the repository URL in a browser and check that the raw Parquet partitions and final analytic data can be downloaded. The largest tracked file is under GitHub's 100 MiB single-file limit, although the complete repository is about 1.3 GB. Insert the verified public URL in the SSAC submission form. The [official competition rules](https://www.sloansportsconference.com/research-paper-competition) require the abstract to be shorter than 500 words, to use Introduction, Methods, Results, and Conclusion sections, and to link an open repository with the data used in the research.

`paper/abstract.md` is the editable source, and `make abstract pdf` regenerates its one-page exact-text PDF. The submitted abstract intentionally omits an author placeholder; provide author details in the portal or add them to the generator if the portal explicitly requires them inside the PDF. The PDF is locally validated for page size, section headings, word count, and exact text match.

Before submission, check the public repository link, the final PDF, and the portal's live upload fields. The 2026 snapshot ends September 20. The percentile search is exploratory, and its positive swing-length relationships do not establish persistent adaptation skill. No submission or acceptance is claimed by this repository.
