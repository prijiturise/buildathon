
from pathlib import Path
from playwright.sync_api import sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

URL = "https://www.tnagrisnet.tn.gov.in/people_app/GoScheme"
OUTPUT = Path.cwd() / "GoScheme_All_Pages.pdf"


with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})

    try:
        page.goto(URL, wait_until="domcontentloaded", timeout=120000)
        page.locator("body").wait_for(state="visible")
        page.wait_for_timeout(3000)

        # Locate the main visible table.
        tables = page.locator("table:visible")

        if tables.count() == 0:
            raise RuntimeError(
                "No visible table found. Inspect the website structure."
            )

        table = tables.nth(0)
        collected = []
        seen = set()

        # Find pagination links/buttons labelled 1, 2, 3, etc.
        # Re-query controls after every click because the page can update.
        for page_no in range(1, 100):
            table = page.locator("table:visible").nth(0)

            # Capture the table currently displayed.
            html = table.evaluate("(el) => el.outerHTML")

            if html not in seen:
                seen.add(html)
                collected.append(
                    f"<h2>Scheme page {page_no}</h2>{html}"
                )
                print(f"Captured table {page_no}")
            else:
                print("Table already captured.")

            # Find the requested numbered page in pagination.
            number = str(page_no + 1)

            candidates = page.get_by_role(
                "link", name=number, exact=True
            )

            if candidates.count() == 0:
                candidates = page.get_by_role(
                    "button", name=number, exact=True
                )

            target = None

            for i in range(candidates.count()):
                item = candidates.nth(i)
                if item.is_visible() and item.is_enabled():
                    target = item
                    break

            if target is None:
                # No more numbered pages are currently available.
                break

            old_html = html
            target.click()
            try:
                page.wait_for_function(
                    """oldHTML => {
                        const tables = [
                            ...document.querySelectorAll('table')
                        ].filter(t => t.getClientRects().length);
                        return tables.length > 0 &&
                            tables[0].outerHTML !== oldHTML;
                    }""",
                    arg=old_html,
                    timeout=10000,
                )
            except PlaywrightTimeoutError:
                print("Table did not change; stopping pagination.")
                break

            page.wait_for_timeout(300)

        if not collected:
            raise RuntimeError("No table pages were captured.")

        # Build one PDF from the collected HTML tables.
        pdf_html = """
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <style>
        @page { size: A3 landscape; margin: 10mm; }
        body { font-family: Arial, sans-serif; font-size: 9pt; }
        table { width: 100%; border-collapse: collapse; }
        th, td {
            border: 1px solid #777;
            padding: 5px;
            overflow-wrap: anywhere;
            text-align: left;
        }
        th { background: #ddd; }
        h2 { page-break-before: always; }
        h2:first-of-type { page-break-before: auto; }
        </style>
        </head>
        <body>
        <h1>Tamil Nadu Agriculture - Go Scheme</h1>
        """ + "\n".join(collected) + """
        </body>
        </html>
        """

        pdf_page = browser.new_page()
        pdf_page.set_content(pdf_html, wait_until="load")

        pdf_page.pdf(
            path=str(OUTPUT),
            format="A3",
            landscape=True,
            print_background=True
        )

        print(f"\nPDF saved to: {OUTPUT}")
        print(f"Distinct tables captured: {len(collected)}")

    except Exception as error:
        print("Error:", error)
        page.screenshot(path="goscheme_debug.png", full_page=True)

    finally:
        browser.close()
