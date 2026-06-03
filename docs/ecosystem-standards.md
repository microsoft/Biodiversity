---
title: "Ecosystem Documentation Standards for Microsoft Biodiversity Projects"
description: "The shared documentation standards every Microsoft Biodiversity ecosystem repo follows: site structure, metadata, structured data, cross-linking, and topic ownership."
tags:
  - documentation standards
  - open source conservation AI
  - Microsoft biodiversity AI
---

# Ecosystem Documentation Standards

Every project in the Microsoft Biodiversity ecosystem publishes its documentation the same way, so
the sites read as one family and each ranks as the canonical source for its own topic. This page is
the reference for bringing a new repo up to that standard, distilled from the work on
[MegaDetector](https://microsoft.github.io/MegaDetector/), which sets the pattern.

## Site structure

Each repo ships a MkDocs Material site under `docs/`, deployed to
`https://microsoft.github.io/<Repo>/`. Use a single `<h1>` per page that matches the page title,
keep a clear navigation tree, and give every concept its own page rather than one long README. A
docs homepage leads with the value proposition in the first sentence, then links out to the
detailed pages.

## Page metadata

Every content page carries front matter:

- a `title` that is unique and leads with the page's primary topic,
- a `description` of roughly 110 to 160 characters that reads as a search snippet,
- a `tags` list scoped to that page's topic.

`mkdocs.yml` sets `site_url` and a keyword-rich `site_description`. Descriptions stay distinct from
one repo to the next so previews and snippets do not blur together.

## Structured data and social cards

A theme override (`overrides/main.html`) injects JSON-LD, gated per page so each block only appears
where it belongs. A homepage carries a connected graph: a `WebSite` node, the project as
`SoftwareSourceCode` (the hub uses `CollectionPage` plus an `ItemList` of the projects), and the Lab
as the publishing `Organization`, cross-linked by `@id`. Add `sameAs` links to the GitHub repo and
any package, paper, or model page so search engines can resolve the project as one entity. Interior
pages carry a `BreadcrumbList`, and pages with a FAQ carry a `FAQPage`. Every page also emits Open
Graph and Twitter Card tags with an absolute share image.

## Build and freshness

The site uses the `callouts` plugin so GitHub-style admonitions render in both the README and the
docs, and `git-revision-date-localized` to show a real last-updated date. The deploy workflow checks
out full history (`fetch-depth: 0`) so those dates are accurate. A self-hosted favicon and logo live
in `docs/assets/` rather than loading from an external host.

## Cross-linking and topic ownership

The hub and the projects link to each other so the ecosystem reads as a connected graph rather than
separate sites:

- The hub links out to every project with a topic-specific anchor.
- Each project links back to the hub with an umbrella anchor.
- Each project carries a short "related projects" section, with one sentence on how each neighbor
  differs, rather than an identical block of links on every page.

Topic ownership keeps the projects from competing with each other. The hub owns the umbrella terms
for Microsoft biodiversity AI and open-source conservation AI. Each project owns its own area:

| Project | Owns |
|---|---|
| [MegaDetector](https://microsoft.github.io/MegaDetector/) | camera-trap image detection and blank-frame filtering |
| [MegaDetector-Acoustic](https://microsoft.github.io/MegaDetector-Acoustic/) | terrestrial bioacoustic classification and species identification from sound |
| [SPARROW](https://microsoft.github.io/SPARROW/) | solar-powered edge hardware, field deployment, and remote connectivity |
| [PyTorch-Wildlife](https://microsoft.github.io/Pytorch-Wildlife/) | the deep learning framework, model zoo, training, and inference |

When one project's page mentions a topic another project owns, it links to that owner rather than
trying to rank for the term itself.

## Writing

Documentation is written to be read by people. Keep prose specific and sourced to the repository,
vary sentence structure, and avoid filler. New pages add real material rather than restating the
README. Claims about accuracy, scale, or performance point to a source or are softened.
