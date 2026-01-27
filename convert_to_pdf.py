#!/usr/bin/env python3
"""
Convert HTML presentation to PDF
Uses playwright for high-quality rendering
"""

import asyncio
from playwright.async_api import async_playwright
import os

async def convert_html_to_pdf():
    """Convert reveal.js presentation to PDF"""

    html_path = os.path.abspath('TradeBridge_Presentation.html')
    pdf_path = os.path.abspath('TradeBridge_Presentation.pdf')

    print(f"📄 Converting HTML to PDF...")
    print(f"   Source: {html_path}")
    print(f"   Output: {pdf_path}")

    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={'width': 1920, 'height': 1080})

        # Load the presentation
        await page.goto(f'file://{html_path}')

        # Wait for reveal.js to initialize
        await page.wait_for_timeout(2000)

        # Get total number of slides
        total_slides = await page.evaluate('''() => {
            return Reveal.getTotalSlides();
        }''')

        print(f"   Total slides: {total_slides}")
        print(f"   Rendering slides...")

        # Export to PDF with high quality
        await page.pdf(
            path=pdf_path,
            format='A4',
            landscape=True,
            print_background=True,
            margin={
                'top': '0',
                'right': '0',
                'bottom': '0',
                'left': '0'
            },
            scale=0.95
        )

        await browser.close()

        print(f"\n✅ PDF created successfully!")
        print(f"📁 Location: {pdf_path}")

        # Get file size
        file_size = os.path.getsize(pdf_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"📊 File size: {file_size_mb:.2f} MB")

if __name__ == '__main__':
    asyncio.run(convert_html_to_pdf())
