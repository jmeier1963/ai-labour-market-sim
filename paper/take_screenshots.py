"""Capture both app screenshots for the paper with a headless browser, then
crop the app bar. Requires the Solara app running on :8765 with the temporary
screenshot defaults (fast_takeoff preset, 12 runs). Run:
  uv run --with playwright --with pillow python paper/take_screenshots.py
"""

import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

FIG_DIR = Path(__file__).parent / "figures"
BASE = "http://127.0.0.1:8765"
APP_BAR_CROP = 300  # px at device_scale_factor=2


def crop_top(path: Path, pixels: int) -> None:
    image = Image.open(path)
    image.crop((0, pixels, image.width, image.height)).save(path)


def capture_live(page) -> None:
    page.goto(BASE, wait_until="networkidle")
    page.wait_for_timeout(12_000)
    page.get_by_role("button").filter(has_text="▶").first.click()
    deadline = time.time() + 300
    while time.time() < deadline:
        page.wait_for_timeout(10_000)
    for label in ("⏸", "▶"):
        buttons = page.get_by_role("button").filter(has_text=label)
        if buttons.count():
            buttons.first.click()
            break
    page.wait_for_timeout(6_000)
    page.screenshot(path=str(FIG_DIR / "app_live.png"), full_page=False)
    print("live saved,", page.locator("text=/Time: /").first.inner_text())


def capture_research(page) -> None:
    page.goto(f"{BASE}/research", wait_until="networkidle")
    page.wait_for_timeout(6_000)
    page.get_by_role("button", name="Run Monte Carlo").click()
    page.wait_for_selector("text=/done: /", timeout=600_000)
    page.wait_for_timeout(5_000)
    page.screenshot(path=str(FIG_DIR / "app_research.png"), full_page=False)
    print("research saved")


if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1150}, device_scale_factor=2)
        capture_live(page)
        capture_research(page)
        browser.close()
    crop_top(FIG_DIR / "app_live.png", APP_BAR_CROP)
    crop_top(FIG_DIR / "app_research.png", APP_BAR_CROP)
    print("SCREENSHOTS DONE")
