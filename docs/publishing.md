# Preparing and publishing a prerelease

The first candidate is `phsi-models==1.0.0a1`: alpha 1 of the planned 1.0.0 release. It contains the working package; the alpha label signals that the interface and behavior may still change. Preparing files or adding a pending publisher does not publish them.

## Source permissions

Source-permission clearance is complete based on the project owner's explicit authorization on 28 October 2026 to reuse all four underlying codebases and release this package under the MIT license. The authorization is recorded in [license status](license-status.md) and [source provenance](source-provenance.json). `LICENSE` contains the package license; both supplied upstream notices are preserved in [LICENSE](../LICENSE) and in the distributions. Attribution remains in the README and [attribution notes](attribution.md), including “not yet public” for the sparse-identification repository.

There is no additional PyPI source-clearance form. Keep these records and notices with future releases; separately assess the terms of any new source material added later.

## Prepare the files locally

From the repository root, with the development environment active:

```bash
python -m pip install '.[dev]' 'twine>=6.1'
python -m pytest
python -m build --outdir dist/prerelease-1.0.0a1
python -m twine check --strict dist/prerelease-1.0.0a1/*
PHSI_DIST_DIR=dist/prerelease-1.0.0a1 python -m pytest tests/test_distribution.py
```

The upload candidates are exactly:

- `dist/prerelease-1.0.0a1/phsi_models-1.0.0a1-py3-none-any.whl` — the installable wheel.
- `dist/prerelease-1.0.0a1/phsi_models-1.0.0a1.tar.gz` — the source distribution.

Use that dedicated directory, not a wildcard over older distributions. The archive filenames normalize the distribution name `phsi-models` to `phsi_models`; Python code still uses `import phsi`. The version is recorded in `pyproject.toml`, `src/phsi/__init__.py` and `CITATION.cff`; the version assertion in `tests/test_estimator.py` must agree. No model equations or fitting settings change for this prerelease.

## Configure the PyPI account

Confirm that your email is verified and [two-factor authentication](https://docs.pypi.org/your-account/two-factor-authentication/) is enabled. Then open [account publishing](https://pypi.org/manage/account/publishing/) and add a pending **GitHub** publisher with these values:

| Field | Value |
|---|---|
| PyPI project name | `phsi-models` |
| Owner | `SINTEF` |
| Repository name | `PHSI` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

The workflow name is just the filename, without `.github/workflows/`. This connects the package to the repository's publishing workflow; no stored upload token is needed. A pending publisher **does not reserve the name**. The first successful upload creates the project. An absent project page is not proof that a name is available. See the official [pending-publisher instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) and [naming guidance](https://docs.pypi.org/project-management/choosing-a-name/).

## Prepare GitHub for the authorized publication

In the `SINTEF/PHSI` repository settings, create an environment named `pypi`. Configure a required reviewer and restrict deployments to the `main` branch. The local [release workflow](../.github/workflows/release.yml) uses this environment. Only its publishing job receives permission to obtain an upload identity; its build job tests and checks both distributions first. This follows the [Python packaging publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/).

Before uploading, run the remote test matrix on the exact release commit and inspect the built artifacts. Publish only reviewed public files and preserve upstream notices.

The workflow runs only when manually requested, on `main`, with an explicit version matching the package metadata. Pushing a commit or tag does not trigger an upload. After publication authorization, select **Actions → Publish to PyPI → Run workflow**, choose `main`, enter `1.0.0a1`, and approve the `pypi` environment job. Approving that job makes the package public. For later releases, update the version metadata and lockfile together, and use a new immutable Git tag.

After a successful upload, verify installation outside the checkout:

```bash
python -m pip install 'phsi-models==1.0.0a1'
# Or, in a uv-managed environment:
uv pip install 'phsi-models==1.0.0a1'
```

A pinned alpha version selects that prerelease explicitly. PyPI does not permit replacement of an uploaded distribution filename; corrections require a new version, such as `1.0.0a2`. See [PyPI upload troubleshooting](https://docs.pypi.org/publishing/troubleshooting/).
