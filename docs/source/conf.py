"""Sphinx configuration for the Myli documentation."""

from importlib.metadata import version as distribution_version


project = "Myli"
author = "Eight Potions"
copyright = "2026, Eight Potions"
release = distribution_version("myli")
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

root_doc = "index"
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
myst_heading_anchors = 3

html_theme = "furo"
html_title = "Myli Docs"
html_theme_options = {
    "source_repository": "https://github.com/EightPotions/Myli/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
