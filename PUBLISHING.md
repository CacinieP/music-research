# Build and publish the reviewed notes

The source of truth is the Markdown on `master`. The reading site and GitHub Wiki are generated from it. Do not edit the generated versions independently.

## Local checks

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-site.txt
npm ci --ignore-scripts
bash scripts/build.sh
```

The build checks internal links, section anchors, source-to-HTML formula completeness, and MathJax parsing before exporting the Wiki. CPU examples use a separate Python 3.13 environment. Direct dependencies are pinned in `requirements-examples.txt`; [requirements-examples-lock.txt](requirements-examples-lock.txt) records all installed versions from the macOS arm64 review environment (not a cross-platform lock):

```bash
python3.13 -m venv .venv-examples
source .venv-examples/bin/activate
python -m pip install -r requirements-examples.txt
python -m unittest discover -s scripts -p 'test_examples.py'
```

Model examples in the cookbook have upstream API checks, not full GPU/weight execution. No measured model benchmark is claimed.

## Publish

After all content checks pass and publication is authorized:

1. Merge the reviewed source changes to `master`.
2. Rebuild from that source revision.
3. Publish the `site/` contents with `.nojekyll` to the `gh-pages` branch. Configure GitHub Pages to serve that branch at `/`.
4. Initialize the GitHub Wiki homepage in GitHub's UI if necessary, clone its `.wiki.git` repository and synchronize `.wiki-docs/` into it.
5. Verify the live site and Wiki, including a formula-rich article and cross-page navigation. Compare deployed files against the built files and record the source revision.

This repository does not add a scheduled or automatically dispatched CI workflow. Hosting may run GitHub's own Pages deployment when the publication branch is updated.
