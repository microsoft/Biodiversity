---
description: "How to contribute to the Microsoft AI for Good Lab biodiversity projects: where to open issues and pull requests, and how to build this documentation site."
tags:
  - contribute
  - open source conservation AI
  - pull request guidelines
  - Microsoft Biodiversity
---

# 🤝 Contribution Guidelines

Thanks for your interest in contributing! This repository is the hub of the SPARROW ecosystem: it holds the overview, the project list and this documentation site. The code of each project lives in its own repository.

## How to participate

1. 💡 Find the project you want to work on in the [project list](index.md#projects).
2. 📋 Open issues and pull requests in that project's repository. Each repository has its own contribution notes and license.
3. 📝 For problems with this hub (wrong or broken links, missing projects, outdated descriptions), open an issue or a pull request against `main` in [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity).
4. 💬 For questions, join us on [Discord](https://discord.gg/TeEVxzaYtm).

## Build the docs

This site is built with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/). The pages are in `docs/`, and the navigation, plugins and redirects for old page URLs are in `mkdocs.yml`.

1. Install the dependencies (Python 3.12 is what the deploy workflow uses):
   ```bash
   pip install -r docs-requirements.txt
   ```
2. Preview the site at `http://127.0.0.1:8000/`:
   ```bash
   mkdocs serve
   ```
3. Before opening a pull request, check that the site builds without warnings:
   ```bash
   mkdocs build --strict
   ```

The `site/` directory is generated and is not committed. The site is deployed to GitHub Pages by the `deploy-docs` workflow when changes reach `main`.

---

We have adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [us](mailto:zhongqimiao@microsoft.com) with any additional questions or comments.
