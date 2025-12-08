import asyncio
from playwright.async_api import async_playwright

URL = "http://127.0.0.1:9000"


async def measure(page, label):
    cards = await page.evaluate(
        "() => Array.from(document.querySelectorAll('section.card')).map(el => ({"
        " box: el.getBoundingClientRect().toJSON(),"
        " style: {height: el.style.height, maxHeight: el.style.maxHeight, minHeight: el.style.minHeight, overflow: el.style.overflow},"
        " title: el.querySelector('h2')?.textContent || '' }))"
    )
    details = await page.evaluate(
        "() => {"
        " const res = document.querySelector('section.card-scroll.results-card-flex');"
        " const hist = Array.from(document.querySelectorAll('section.card-scroll')).find(el => el !== res);"
        " const info = el => el ? { box: el.getBoundingClientRect().toJSON(), style: {height: el.style.height, maxHeight: el.style.maxHeight, minHeight: el.style.minHeight, overflow: el.style.overflow} } : null;"
        " return { results: info(res), history: info(hist) };"
        "}"
    )
    print(f"[{label}]")
    for c in cards:
        print(c)
    print("details:", details)
    print("-" * 40)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        await page.goto(URL, wait_until="networkidle", timeout=15000)
        await page.wait_for_timeout(800)
        await measure(page, "initial")

        # trigger one generation with minimal load
        await page.fill("#prompt", "test prompt")
        await page.click("button:has-text('生成图片')")
        await page.wait_for_timeout(500)  # mid-flight
        await measure(page, "during generation")
        await page.wait_for_timeout(4000)  # allow generation to finish
        await measure(page, "after generation")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
