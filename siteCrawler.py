import os, re
import asyncio
import random
from urllib.parse import urldefrag, urlparse, urlunparse
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher, ProxyConfig
import aiohttp

USE_PROXIES = False
MAX_RETRIES = 5
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=10)

def safe_filename(url):
    # Turn URL into a safe filename
    return re.sub(r'[^a-zA-Z0-9_-]', '_', url)

async def is_downloadable(session, url, proxy):
    # Check by URL extension
    url_lower = url.lower()
    if url_lower.endswith((".pdf", ".doc", ".docx", ".txt")):
        return True

    # Otherwise, check headers
    try:
        async with session.get(url, proxy=proxy, allow_redirects=True, timeout=10) as resp:
            content_type = resp.headers.get("Content-Type", "").lower()
            content_disp = resp.headers.get("Content-Disposition", "").lower()
            return (
                "attachment" in content_disp
                or any(mime in content_type for mime in [
                    "application/pdf", "application/msword", 
                    "application/vnd", "text/plain"
                ])
            )
    except Exception as e:
        print(f"[CHECK ERROR] {url}: {e}")
        return False



async def download_file(session, url, proxy, output_dir="output/docs", map_file="links_map.txt"):
    try:
        async with session.get(url, proxy=proxy, timeout=20) as resp:
            content_disposition = resp.headers.get("Content-Disposition", "")
            if "filename=" in content_disposition:
                filename = content_disposition.split("filename=")[1].strip('"; ')
            else:
                path = urlparse(url).path
                filename = os.path.basename(path) or "downloaded_file"

            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "wb") as f:
                while True:
                    chunk = await resp.content.read(1024)
                    if not chunk:
                        break
                    f.write(chunk)

            print(f"[DOWNLOADED] {url} → {filename}")

            with open(map_file, "a", encoding="utf-8") as map_f:
                map_f.write(f"{url},{filename}\n")
    except Exception as e:
        print(f"[DOWNLOAD ERROR] {url}: {e}")




def load_proxies(file_path="static/proxies.txt"):
    if not USE_PROXIES:
        return []
    
    with open(file_path, "r") as f:
        lines = f.read().splitlines()
    return [parse_proxy(line) for line in lines]

def parse_proxy(proxy_line):
    ip, port, user, pwd = proxy_line.strip().split(":")
    return ProxyConfig(
        server=f"http://{ip}:{port}",
        username=user,
        password=pwd
    )

proxies = load_proxies()

async def get_next_proxy():
    if not USE_PROXIES or not proxies:
        return None
    return random.choice(proxies)

async def crawl_recursive_batch(start_urls, max_depth=3, max_concurrent=10):
    browser_config = BrowserConfig(headless=True, verbose=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    visited = set()
    def normalize_url(url):
            url, _ = urldefrag(url)  # Remove fragment
            parsed = urlparse(url)
            
            # Normalize scheme, netloc, path (e.g. remove trailing slashes)
            path = parsed.path.rstrip('/')
            
            return urlunparse(parsed._replace(path=path.lower(), scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower()))

    current_urls = set(normalize_url(u) for u in start_urls)

    async with AsyncWebCrawler(config=browser_config, dispatcher=dispatcher) as crawler:
        for depth in range(max_depth):
            print(f"\n=== Crawling Depth {depth+1} ===")
            urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited]
            if not urls_to_crawl:
                break

            # Semaphore to limit concurrency to max_concurrent (optional but good)
            semaphore = asyncio.Semaphore(max_concurrent)

            async def crawl_url(url):
                async with semaphore:
                    proxy = await get_next_proxy()

                    proxy_url = None
                    if proxy:
                        proxy_url = proxy.server.replace("http://", f"http://{proxy.username}:{proxy.password}@")

                    async with aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT) as session:
                        try:
                            if await is_downloadable(session, url, proxy_url):
                                await download_file(session, url, proxy_url)
                                return None
                        except Exception as e:
                            print(f"[AIOHTTP ERROR] {url} | {e}")

                    # If not a downloadable file, run crawler with retries
                    for attempt in range(MAX_RETRIES):
                        proxy = await get_next_proxy()
                        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False).clone(proxy_config=proxy)
                        try:
                            result = await crawler.arun(url=url, config=run_config)
                            if result.success:
                                return result
                            else:
                                print(f"[Retry {attempt+1}/{MAX_RETRIES}] Failed with proxy: {proxy.server if proxy else 'None'}")
                        except Exception as e:
                             print(f"[Retry {attempt+1}/{MAX_RETRIES}] Error: {e} with proxy: {proxy.server if proxy else 'None'}")
                             await asyncio.sleep(1)  # avoid rapid retry
                    print(f"[GIVE UP] {url}")
                    return None


            tasks = [crawl_url(url) for url in urls_to_crawl]
            results = await asyncio.gather(*tasks)
            # Get the base domain (e.g., business.delaware.gov)
            base_domain = urlparse(start_urls[0]).netloc.lower()


            next_level_urls = set()
            for result in results:
                if result is None:
                    continue
                norm_url = normalize_url(result.url)
                visited.add(norm_url)
                if result.success:
                    print(f"[OK] {result.url} | Markdown length: {len(result.markdown) if result.markdown else 0}")

                    # Save the markdown to a file
                    if result.markdown:
                        filename = f"output/{safe_filename(result.url)}.md"
                        os.makedirs("output", exist_ok=True)
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write(result.markdown)

                    # Filter to only same-domain links
                    for link in result.links.get("internal", []):
                        next_url = normalize_url(link["href"])
                        if urlparse(next_url).netloc.lower() == base_domain and next_url not in visited:
                            next_level_urls.add(next_url)
                else:
                    print(f"[ERROR] {result.url}: {result.error_message}")


            current_urls = next_level_urls

if __name__ == "__main__":
    asyncio.run(crawl_recursive_batch(["https://www.royalchallengers.com/"], max_depth=1, max_concurrent=50))
