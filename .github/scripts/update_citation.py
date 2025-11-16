"""Update version and date in CITATION.cff based on the latest GitHub release."""

from __future__ import annotations

import requests
import yaml

# Replace with your username and repo name
USERNAME = "scottprahl"
REPO = "pyspeckle"

GITHUB_API_URL = f"https://api.github.com/repos/{USERNAME}/{REPO}/releases/latest"

# Add a header so GitHub does not apply restrictive rate limits
HEADERS = {
    "User-Agent": f"{REPO}-citation-updater",
    "Accept": "application/vnd.github+json",
}

# Fetch latest release information with timeout and user-agent
response = requests.get(
    GITHUB_API_URL,
    timeout=10,
    headers=HEADERS,
)
response.raise_for_status()

release_info = response.json()
release_date = release_info["published_at"].split("T")[0]
version = release_info["tag_name"]

# Read existing CITATION.cff
with open("CITATION.cff", "r", encoding="utf-8") as f:
    cff_data = yaml.safe_load(f)

changed = False

# Apply updates only if different
if cff_data.get("date-released") != release_date:
    cff_data["date-released"] = release_date
    changed = True

if cff_data.get("version") != version:
    cff_data["version"] = version
    changed = True

# Save only if modified
if changed:
    with open("CITATION.cff", "w", encoding="utf-8") as f:
        yaml.dump(cff_data, f)
    print(f"CITATION.cff updated → version: {version}, date: {release_date}")
else:
    print("No change in release date or version. No update needed.")
