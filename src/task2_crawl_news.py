

"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding so Vietnamese text prints without crashing
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Danh sách URL bài báo về nghệ sĩ Việt Nam liên quan ma tuý
ARTICLE_URLS = [
    # 1. Long Nhật, Sơn Ngọc Minh bị bắt — VnExpress
    "https://vnexpress.net/ca-si-long-nhat-son-ngoc-minh-bi-bat-vi-lien-quan-ma-tuy-5060857.html",
    # 2. Miu Lê bị bắt vì tổ chức sử dụng ma tuý — VnExpress
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    # 3. Châu Việt Cường nhận 13 năm tù — VnExpress
    "https://vnexpress.net/ca-si-chau-viet-cuong-nhan-13-nam-tu-vi-nhet-toi-hai-chet-co-gai-3891028.html",
    # 4. Bắt Long Nhật và Sơn Ngọc Minh vì ma tuý — Tuổi Trẻ
    "https://tuoitre.vn/bat-ca-si-long-nhat-va-ca-si-son-ngoc-minh-vi-lien-quan-ma-tuy-20260520082138943.htm",
    # 5. Showbiz Việt liên tiếp chấn động vì ma tuý — Thanh Niên
    "https://thanhnien.vn/ca-si-long-nhat-bi-bat-showbiz-viet-lien-tiep-chan-dong-vi-ma-tuy-18526052013032001.htm",
    # 6. Những nghệ sĩ Việt "ngã ngựa" vì ma tuý — Ngôi Sao (VnExpress)
    "https://ngoisao.vnexpress.net/nhung-nghe-si-viet-nga-ngua-vi-ma-tuy-4816068.html",
    # 7. Ma tuý và showbiz — Thanh Niên
    "https://thanhnien.vn/ma-tuy-va-showbiz-su-thanh-loc-can-bat-dau-tu-nghe-si-185260513123425952.htm",
]


def _slugify(text: str, max_len: int = 60) -> str:
    """Chuyển tiêu đề thành tên file an toàn."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text[:max_len]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

        title = (result.metadata or {}).get("title", "")
        if not title and result.markdown:
            # Lấy dòng đầu tiên không trống làm tiêu đề dự phòng
            for line in result.markdown.splitlines():
                line = line.strip().lstrip("#").strip()
                if line:
                    title = line
                    break

        return {
            "url": url,
            "title": title or "Unknown",
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown or "",
        }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    success = 0
    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)

            slug = _slugify(article["title"]) or f"article_{i:02d}"
            filename = f"{i:02d}_{slug}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  [OK] Saved: {filepath.name}  |  Title: {article['title'][:60]}")
            success += 1
        except Exception as exc:
            print(f"  [FAIL] {exc}")

    print(f"\nDone: {success}/{len(ARTICLE_URLS)} articles saved to {DATA_DIR}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
