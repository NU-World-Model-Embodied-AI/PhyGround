"""Render every page of the human_eval Flask app to a PNG screenshot.

Uses headless Firefox via selenium. Creates a throwaway annotator
"_screenshot_bot_<ts>" so demographics/demo/rate_group can be reached;
deletes the row at the end.

Needs Firefox and `pip install selenium`. Without --geckodriver, Selenium
Manager finds or fetches one; with Ubuntu's snap Firefox pass
--geckodriver /snap/bin/geckodriver. Run from evals/ against a running app
that shares DB_PATH:

    python -m flask --app human_eval.app:create_app run --port 5001
    python -m human_eval.capture_screenshots [--base-url http://127.0.0.1:5001]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait

from human_eval.config import DB_PATH

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "screenshots"


def make_driver(geckodriver: str | None, width: int = 1440, height: int = 900) -> webdriver.Firefox:
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument(f"--width={width}")
    opts.add_argument(f"--height={height}")
    service = Service(executable_path=geckodriver)
    drv = webdriver.Firefox(service=service, options=opts)
    drv.set_window_size(width, height)
    return drv


def full_page_screenshot(drv: webdriver.Firefox, path: Path) -> None:
    """Save a full-page PNG (geckodriver's get_full_page_screenshot)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    png = drv.get_full_page_screenshot_as_png()
    path.write_bytes(png)
    print(f"  saved {path.relative_to(HERE)}  ({len(png) // 1024} KB)")


def wait_dom(drv: webdriver.Firefox, timeout: int = 10) -> None:
    WebDriverWait(drv, timeout).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )
    time.sleep(0.6)  # let webfonts / images settle


def login(drv: webdriver.Firefox, base_url: str, username: str, cohort: str) -> None:
    drv.get(f"{base_url}/?cohort={cohort}")
    wait_dom(drv)
    drv.find_element(By.NAME, "username").send_keys(username)
    # cohort is hidden in form, already set via session
    drv.find_element(By.CSS_SELECTOR, "form button[type=submit], form input[type=submit]").click()
    wait_dom(drv)


def fill_demographics(drv: webdriver.Firefox, base_url: str) -> None:
    drv.get(f"{base_url}/demographics")
    wait_dom(drv)
    # Pick first option in each select / radio
    drv.execute_script(
        """
        function set(name, val){
            const el = document.querySelector(`[name="${name}"]`);
            if(!el) return;
            if(el.tagName==='SELECT'){
                for(const o of el.options){ if(o.value){ el.value=o.value; break; } }
            } else { el.value = val; }
            el.dispatchEvent(new Event('change', {bubbles:true}));
        }
        // gender / age / education / major typically radios or selects
        document.querySelectorAll('input[type=radio]').forEach((r, i) => {
            const n = r.name; if(!document.querySelector(`input[name="${n}"]:checked`)) r.checked = true;
        });
        set('gender','male'); set('age','20-29'); set('education','bachelor'); set('major','cs');
        """
    )
    drv.find_element(By.CSS_SELECTOR, "form button[type=submit], form input[type=submit]").click()
    wait_dom(drv)


def find_a_group_id_from_db(annotator_name: str) -> str | None:
    """Query DB for one of this annotator's pending group_ids."""
    db = sqlite3.connect(str(DB_PATH))
    try:
        row = db.execute(
            "SELECT a.group_id FROM assignments a "
            "JOIN annotators an ON a.annotator_id = an.id "
            "WHERE an.name = ? AND a.group_id IS NOT NULL "
            "ORDER BY a.id LIMIT 1",
            (annotator_name,),
        ).fetchone()
        return row[0] if row else None
    finally:
        db.close()


def cleanup_user(name: str) -> None:
    if not DB_PATH.exists():
        return
    db = sqlite3.connect(str(DB_PATH))
    try:
        row = db.execute("SELECT id FROM annotators WHERE name = ?", (name,)).fetchone()
        if row is None:
            return
        aid = row[0]
        db.execute("DELETE FROM annotations WHERE assignment_id IN "
                   "(SELECT id FROM assignments WHERE annotator_id = ?)", (aid,))
        db.execute("DELETE FROM assignments WHERE annotator_id = ?", (aid,))
        db.execute("DELETE FROM annotators WHERE id = ?", (aid,))
        db.commit()
        print(f"cleaned up annotator '{name}' (id={aid})")
    finally:
        db.close()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:5001")
    p.add_argument("--geckodriver", default=None,
                   help="Path to geckodriver (default: Selenium Manager)")
    p.add_argument("--keep-user", action="store_true",
                   help="Skip DB cleanup at the end")
    args = p.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    username = f"_screenshot_bot_{int(time.time())}"
    drv = make_driver(args.geckodriver)
    try:
        # ---- public pages ----
        for slug, url in [
            ("01_login_others", f"{args.base_url}/?cohort=others"),
            ("02_guide_en", f"{args.base_url}/guide?lang=en"),
            ("03_guide_zh", f"{args.base_url}/guide?lang=zh"),
        ]:
            print(f"-> {url}")
            drv.get(url)
            wait_dom(drv)
            full_page_screenshot(drv, OUT_DIR / f"{slug}.png")

        # ---- login (creates assignments via _auto_assign) ----
        print(f"-> login as {username}")
        login(drv, args.base_url, username, "others")

        # ---- demographics (GET, before submit) ----
        print(f"-> /demographics (blank)")
        drv.get(f"{args.base_url}/demographics")
        wait_dom(drv)
        full_page_screenshot(drv, OUT_DIR / "04_demographics.png")

        # submit demographics so /demo works
        fill_demographics(drv, args.base_url)

        # ---- demo / task list ----
        print(f"-> /demo")
        drv.get(f"{args.base_url}/demo?lang=en")
        wait_dom(drv)
        full_page_screenshot(drv, OUT_DIR / "05_demo_en.png")

        drv.get(f"{args.base_url}/demo?lang=zh")
        wait_dom(drv)
        full_page_screenshot(drv, OUT_DIR / "06_demo_zh.png")

        # ---- rate_group ----
        gid = find_a_group_id_from_db(username)
        if gid:
            print(f"-> /rate_group/{gid}")
            drv.get(f"{args.base_url}/rate_group/{gid}")
            wait_dom(drv, timeout=20)
            time.sleep(3)  # videos may take a moment to attach
            full_page_screenshot(drv, OUT_DIR / "07_rate_group.png")
        else:
            print("WARN: no assigned group found, skipping rate_group")

        # ---- thanks ----
        print(f"-> /thanks")
        drv.get(f"{args.base_url}/thanks")
        wait_dom(drv)
        full_page_screenshot(drv, OUT_DIR / "08_thanks.png")

    finally:
        drv.quit()
        if not args.keep_user:
            cleanup_user(username)

    print(f"\nAll screenshots written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
