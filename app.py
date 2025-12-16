import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import anthropic
import base64
from io import BytesIO
from PIL import Image
import re
import os
import json
import hashlib
from typing import Any
from dotenv import load_dotenv

# .envファイルを読み込み
load_dotenv()

ANTHROPIC_VERSION = "2023-06-01"


def get_api_key_from_env() -> str:
    """環境変数からAPIキーを取得"""
    return os.getenv("ANTHROPIC_API_KEY", "")


def get_api_key_from_secrets() -> str:
    """Streamlit SecretsからAPIキーを取得"""
    try:
        return st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        return ""


def get_stored_api_key() -> str:
    """環境変数、Secrets、セッションの順でAPIキーを取得"""
    # まず環境変数をチェック（.envファイル）
    env_key = get_api_key_from_env()
    if env_key:
        return env_key
    # 次にSecretsをチェック（Streamlit Cloud用）
    secrets_key = get_api_key_from_secrets()
    if secrets_key:
        return secrets_key
    # 最後にセッション状態をチェック（ユーザー入力）
    return st.session_state.get("user_api_key", "")


st.set_page_config(
    page_title="漫画画像解析ツール",
    page_icon="📚",
    layout="wide"
)

# カスタムCSS
st.markdown("""
<style>
    .stCheckbox p {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 漫画画像解析ツール")
st.markdown("URLから漫画画像を抽出し、AIであらすじを解析します")


# ----------------------------
# コスト最適化の基本方針
# - 画像は縮小＋JPEG化して送信トークンを削減
# - 画像抽出は安価モデルで実施し、怪しい結果のみOpusへエスカレーション
# - 要約・整合性チェックは原則テキスト処理なので安価モデルへ
# - Streamlitのキャッシュで同一入力の再課金を抑止
# ----------------------------


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _safe_json_loads(s: str) -> dict[str, Any] | None:
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_json_block(text: str) -> str | None:
    """モデル出力からJSON部分だけを抜き出す（前後に説明文が付くことがあるため）"""
    if not text:
        return None
    # 最初の { から最後の } を雑に拾う（最小限の実装）
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


@st.cache_data(show_spinner=False, ttl=60 * 60)
def _cached_download_image(url: str, referer: str = "") -> bytes | None:
    return download_image(url, referer)


def preprocess_image_bytes(
    img_bytes: bytes,
    max_side: int = 1024,
    jpeg_quality: int = 70,
) -> bytes:
    """画像を縮小してJPEG化し、送信コスト（画像トークン）を下げる"""
    img = Image.open(BytesIO(img_bytes))
    # 透過を考慮してRGBへ
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    elif img.mode == "L":
        # 白黒はそのままでもよいが、JPEG化のためRGBへ統一
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_side:
        img.thumbnail((max_side, max_side))

    out = BytesIO()
    img.save(out, format="JPEG", quality=int(jpeg_quality), optimize=True, progressive=True)
    return out.getvalue()


def encode_image_to_base64_bytes(img_bytes: bytes) -> tuple[str, str]:
    base64_image = base64.standard_b64encode(img_bytes).decode("utf-8")
    return base64_image, "image/jpeg"


def call_claude_messages(
    api_key: str,
    model: str,
    content: list[dict],
    max_tokens: int,
    temperature: float = 0.2,
) -> str:
    """Claude API呼び出し（失敗時は例外を投げる）"""
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": content}],
    )
    # contentが複数要素になるケースはあるが、このアプリではtext先頭で十分
    return message.content[0].text


def call_claude_messages_with_usage(
    api_key: str,
    model: str,
    content: list[dict],
    max_tokens: int,
    temperature: float = 0.2,
) -> tuple[str, dict[str, Any]]:
    """Claude API呼び出し（usageも返す）"""
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": content}],
    )
    usage: dict[str, Any] = {}
    try:
        # anthropic SDKのMessageはusage属性を持つ
        u = getattr(message, "usage", None)
        if u is not None:
            usage = {
                "input_tokens": getattr(u, "input_tokens", None),
                "output_tokens": getattr(u, "output_tokens", None),
                "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", None),
                "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None),
            }
    except Exception:
        usage = {}
    return message.content[0].text, usage


@st.cache_data(show_spinner=False, ttl=60 * 10)
def get_available_anthropic_models(api_key: str) -> list[str]:
    """Anthropic APIから利用可能モデル一覧を取得（取れない場合は空リスト）"""
    if not api_key:
        return []
    try:
        resp = requests.get(
            "https://api.anthropic.com/v1/models",
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
            timeout=20,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        models = []
        for item in data.get("data", []):
            mid = item.get("id")
            if mid:
                models.append(mid)
        # ついでに "latest" があれば上に来るように
        models = sorted(set(models), key=lambda s: (0 if "latest" in s else 1, s))
        return models
    except Exception:
        return []


def get_request_headers(url: str) -> dict:
    """リクエストヘッダーを生成"""
    parsed_url = urlparse(url)
    base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Referer": base_domain,
    }


def get_pagination_urls(url: str, soup: BeautifulSoup, debug: bool = False) -> list[str]:
    """ページネーションのURLを取得"""
    urls = [url]  # 現在のページを含む

    # ページネーションのセレクタパターン
    pagination_selectors = [
        ".pagination a",
        ".page-numbers a",
        ".pager a",
        ".wp-pagenavi a",
        "nav.navigation a",
        ".post-page-numbers",
        # 数字リンク（1, 2, 3...）
        "a.page-link",
        ".pages a",
    ]

    pagination_links = []

    for selector in pagination_selectors:
        links = soup.select(selector)
        if links:
            pagination_links.extend(links)
            if debug:
                st.write(f"ページネーション検出: {selector} ({len(links)}件)")
            break

    # ページネーションが見つからない場合、数字のリンクを探す
    if not pagination_links:
        # テキストが数字のみのリンクを探す
        all_links = soup.find_all("a")
        base_path = urlparse(url).path.rstrip('/')
        for link in all_links:
            text = link.get_text(strip=True)
            href = link.get("href", "")
            if not href:
                continue
            # 数字のみのテキスト
            if text.isdigit():
                # 絶対URLまたは相対URLで同じ記事のページネーション
                full_href = urljoin(url, href)
                href_path = urlparse(full_href).path.rstrip('/')
                # 同じ記事へのリンク（/archives/823243/2 形式）
                if href_path.startswith(base_path):
                    pagination_links.append(link)
                    if debug:
                        st.write(f"数字リンク検出: {text} -> {full_href}")

    # URLを抽出
    seen = {url}
    for link in pagination_links:
        href = link.get("href")
        if href:
            full_url = urljoin(url, href)
            # 同じドメインで、まだ追加されていないURL
            if urlparse(full_url).netloc == urlparse(url).netloc and full_url not in seen:
                # 「次へ」「前へ」などのナビゲーションリンクを除外
                text = link.get_text(strip=True).lower()
                if text not in ["next", "prev", "previous", "»", "«", "›", "‹", "次へ", "前へ"]:
                    urls.append(full_url)
                    seen.add(full_url)

    # 基本URLのパスの長さを取得（ベースライン）
    base_path = urlparse(url).path.rstrip('/')

    # URLをページ番号順にソート
    def extract_page_num(u):
        path = urlparse(u).path.rstrip('/')
        # ベースURLと同じパスなら1ページ目
        if path == base_path:
            return 1
        # ベースURLの後に/数字がある場合（例: /archives/823243/2）
        if path.startswith(base_path + '/'):
            suffix = path[len(base_path)+1:]
            if suffix.isdigit():
                return int(suffix)
        return 999  # 不明な場合は最後に

    urls.sort(key=extract_page_num)

    if debug and len(urls) > 1:
        st.write(f"検出されたページ: {len(urls)}ページ")
        for u in urls:
            st.write(f"  - {u}")

    return urls


def get_page_images(url: str, debug: bool = False) -> tuple[list[dict], BeautifulSoup]:
    """ページから画像URLを抽出"""
    headers = get_request_headers(url)

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        st.error(f"ページの取得に失敗しました: {e}")
        return [], None

    soup = BeautifulSoup(response.content, "html.parser")
    images = []

    if debug:
        st.write(f"HTMLサイズ: {len(response.content)} bytes")

    # 記事本文内の画像を優先的に取得
    content_selectors = [
        "article",
        ".entry-content",
        ".post-content",
        ".article-content",
        ".content",
        ".single-content",
        ".post-body",
        ".article-body",
        "main",
        "#content",
        "#main",
        ".post",
        ".entry",
        ".ystd",
        "#ystd",
    ]

    content_area = None
    for selector in content_selectors:
        content_area = soup.select_one(selector)
        if content_area:
            if debug:
                st.write(f"コンテンツエリア検出: {selector}")
            break

    if not content_area:
        content_area = soup.body if soup.body else soup
        if debug:
            st.write("コンテンツエリア: body全体")

    img_tags = content_area.find_all("img")

    if debug:
        st.write(f"検出されたimgタグ数: {len(img_tags)}")

    skip_patterns = [
        "icon", "logo", "avatar", "emoji", "button",
        "banner", "advertisement", "widget",
        "gravatar", "favicon", "sprite", "pixel",
        "tracking", "analytics", "1x1"
    ]

    for img in img_tags:
        src = (
            img.get("src") or
            img.get("data-src") or
            img.get("data-lazy-src") or
            img.get("data-original") or
            img.get("data-full-url") or
            img.get("srcset", "").split()[0] if img.get("srcset") else None
        )

        if not src:
            continue

        if src.startswith("data:"):
            continue

        img_url = urljoin(url, src)

        img_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp"]
        has_img_ext = any(ext in img_url.lower() for ext in img_extensions)

        if any(pattern in img_url.lower() for pattern in skip_patterns):
            continue

        alt_text = img.get("alt", "")

        if has_img_ext or "/uploads/" in img_url or "/images/" in img_url:
            images.append({
                "url": img_url,
                "alt": alt_text
            })
            if debug:
                st.write(f"画像追加: {img_url[:80]}...")

    # 重複を除去
    seen_urls = set()
    unique_images = []
    for img in images:
        if img["url"] not in seen_urls:
            seen_urls.add(img["url"])
            unique_images.append(img)

    return unique_images, soup


def get_all_pages_images(url: str, debug: bool = False) -> list[dict]:
    """すべてのページから画像を取得"""
    # 最初のページを取得
    first_page_images, soup = get_page_images(url, debug)

    if not soup:
        return []

    # ページネーションを検出
    page_urls = get_pagination_urls(url, soup, debug)

    all_images = []
    seen_urls = set()

    # 最初のページの画像を追加
    for img in first_page_images:
        if img["url"] not in seen_urls:
            img["page"] = 1
            all_images.append(img)
            seen_urls.add(img["url"])

    # 追加のページがある場合
    if len(page_urls) > 1:
        for i, page_url in enumerate(page_urls[1:], start=2):
            if debug:
                st.write(f"ページ {i} を取得中: {page_url}")

            page_images, _ = get_page_images(page_url, debug)

            for img in page_images:
                if img["url"] not in seen_urls:
                    img["page"] = i
                    all_images.append(img)
                    seen_urls.add(img["url"])

    return all_images


def get_next_episode_url(soup: BeautifulSoup, base_url: str, debug: bool = False) -> str | None:
    """「次の話>>」のURLを取得"""
    # <div class="page-text-body">次の話＞＞</div> を検出
    next_episode_div = soup.find("div", class_="page-text-body", string=lambda t: t and "次の話" in t)

    if next_episode_div:
        # 親要素からリンクを探す
        parent = next_episode_div.find_parent("a")
        if parent and parent.get("href"):
            next_url = urljoin(base_url, parent["href"])
            if debug:
                st.write(f"🔗 次の話を検出: {next_url}")
            return next_url

        # 兄弟要素や近くのリンクを探す
        next_link = next_episode_div.find_next("a")
        if next_link and next_link.get("href"):
            next_url = urljoin(base_url, next_link["href"])
            if debug:
                st.write(f"🔗 次の話を検出: {next_url}")
            return next_url

    if debug:
        st.write("ℹ️ 「次の話」リンクは見つかりませんでした")

    return None


def get_episode_images(url: str, episode_num: int = 1, debug: bool = False) -> tuple[list[dict], str | None]:
    """1話分の画像を取得（ページネーションを含む）

    Returns:
        tuple: (画像リスト, 次の話のURL or None)
    """
    # 最初のページを取得
    first_page_images, soup = get_page_images(url, debug)

    if not soup:
        return [], None

    # 「次の話」のURLを取得
    next_episode_url = get_next_episode_url(soup, url, debug)

    # ページネーションを検出
    page_urls = get_pagination_urls(url, soup, debug)

    all_images = []
    seen_urls = set()

    if debug:
        st.write(f"📖 第{episode_num}話の取得開始")

    # 最初のページの画像を追加
    for img in first_page_images:
        if img["url"] not in seen_urls:
            img["page"] = 1
            img["episode"] = episode_num
            all_images.append(img)
            seen_urls.add(img["url"])

    # 追加のページがある場合
    if len(page_urls) > 1:
        for i, page_url in enumerate(page_urls[1:], start=2):
            if debug:
                st.write(f"  ページ {i} を取得中: {page_url}")

            page_images, page_soup = get_page_images(page_url, debug)

            for img in page_images:
                if img["url"] not in seen_urls:
                    img["page"] = i
                    img["episode"] = episode_num
                    all_images.append(img)
                    seen_urls.add(img["url"])

            # 各ページでも「次の話」リンクを確認（最後のページで見つかることがある）
            if page_soup and not next_episode_url:
                next_episode_url = get_next_episode_url(page_soup, page_url, debug)

    if debug:
        st.write(f"📖 第{episode_num}話: {len(all_images)}枚の画像を取得")

    return all_images, next_episode_url


def get_multiple_episodes_images(url: str, num_episodes: int, debug: bool = False) -> list[dict]:
    """複数話の画像を取得

    Args:
        url: 開始話のURL
        num_episodes: 取得する話数
        debug: デバッグモード

    Returns:
        list: 全話の画像リスト
    """
    all_images = []
    current_url = url

    for episode in range(1, num_episodes + 1):
        if not current_url:
            if debug:
                st.write(f"⚠️ 第{episode}話のURLがありません。取得を終了します。")
            break

        if debug:
            st.write(f"📚 第{episode}話を取得中: {current_url}")

        episode_images, next_url = get_episode_images(current_url, episode_num=episode, debug=debug)
        all_images.extend(episode_images)

        # 次の話へ
        current_url = next_url

        if not next_url and episode < num_episodes:
            if debug:
                st.write(f"ℹ️ 第{episode}話が最終話です。{episode}話分を取得しました。")
            break

    if debug:
        st.write(f"✅ 合計 {len(all_images)}枚の画像を取得（{min(episode, num_episodes)}話分）")

    return all_images


def download_image(url: str, referer: str = "") -> bytes | None:
    """画像をダウンロード"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Referer": referer,
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.RequestException:
        return None


def filter_manga_images(
    images: list[dict],
    min_size: int = 50000,
    referer: str = "",
    debug: bool = False,
    preprocess_max_side: int = 1024,
    preprocess_jpeg_quality: int = 70,
) -> list[dict]:
    """漫画画像をフィルタリング"""
    manga_images = []

    for img_info in images:
        img_data = _cached_download_image(img_info["url"], referer)
        if not img_data:
            if debug:
                st.write(f"ダウンロード失敗: {img_info['url'][:60]}...")
            continue

        if len(img_data) < min_size:
            if debug:
                st.write(f"サイズ不足 ({len(img_data)} bytes): {img_info['url'][:60]}...")
            continue

        try:
            img = Image.open(BytesIO(img_data))
            width, height = img.size

            aspect_ratio = width / height if height > 0 else 0

            if aspect_ratio > 3:
                if debug:
                    st.write(f"アスペクト比除外 ({aspect_ratio:.2f}): {img_info['url'][:60]}...")
                continue

            if width < 200 or height < 200:
                if debug:
                    st.write(f"サイズ除外 ({width}x{height}): {img_info['url'][:60]}...")
                continue

            # 送信コスト削減のため、LLM送信用は縮小＋JPEG化したものを保持
            send_data = preprocess_image_bytes(
                img_data,
                max_side=preprocess_max_side,
                jpeg_quality=preprocess_jpeg_quality,
            )

            manga_images.append({
                **img_info,
                "data": img_data,
                "send_data": send_data,
                "width": width,
                "height": height,
                "size": len(img_data)
            })

            if debug:
                st.write(f"✅ 漫画画像として追加: {width}x{height}, raw={len(img_data)} bytes, send={len(send_data)} bytes")

        except Exception as e:
            if debug:
                st.write(f"画像処理エラー: {e}")
            continue

    return manga_images


def encode_image_to_base64(img_info: dict) -> tuple[str, str]:
    """画像をbase64エンコード（LLM送信用の縮小JPEGを優先）"""
    img_bytes = img_info.get("send_data") or img_info.get("data") or b""
    return encode_image_to_base64_bytes(img_bytes)


def _get_llm_cache() -> dict:
    """セッション内キャッシュ（同一画像×同一モデルの再課金を抑止）"""
    if "llm_cache" not in st.session_state:
        st.session_state["llm_cache"] = {}
    return st.session_state["llm_cache"]


def _image_cache_key(img_info: dict, model: str, prompt_key: str) -> str:
    img_bytes = img_info.get("send_data") or img_info.get("data") or b""
    h = hashlib.sha256(img_bytes).hexdigest()
    meta = f"{model}|{prompt_key}|ep={img_info.get('episode',1)}|p={img_info.get('page',1)}"
    return f"{h}:{_sha256_text(meta)}"


def _validate_image_facts(facts: dict[str, Any]) -> tuple[bool, list[str]]:
    """人物・主要イベントに影響する誤りを拾うための“怪しさ”判定（画像は見ずテキストで判定）"""
    reasons: list[str] = []
    if not isinstance(facts, dict):
        return False, ["JSONではありません"]

    confidence = facts.get("confidence")
    if isinstance(confidence, (int, float)):
        if confidence < 0.55:
            reasons.append(f"confidenceが低い({confidence})")
    else:
        reasons.append("confidenceが未設定")

    characters = facts.get("characters")
    events = facts.get("events")
    if not isinstance(characters, list):
        reasons.append("charactersが配列ではない")
        characters = []
    if not isinstance(events, list):
        reasons.append("eventsが配列ではない")
        events = []

    # 人物・イベントが両方空は危険（読み取り失敗の可能性が高い）
    if len(characters) == 0 and len(events) == 0:
        reasons.append("人物/イベントが空")

    # “不明”や“読めない”が多い場合は危険
    as_text = json.dumps(facts, ensure_ascii=False)
    bad_markers = ["不明", "読めない", "判別不能", "見えない", "わからない", "?", "□", "�"]
    if sum(as_text.count(m) for m in bad_markers) >= 3:
        reasons.append("不明/文字化け/判別不能が多い")

    # テンプレっぽい出力（極端に短い）
    if len(as_text) < 120:
        reasons.append("出力が短すぎる")

    suspicious = len(reasons) > 0
    return suspicious, reasons


def _add_usage_totals(meta: dict[str, Any], model: str, usage: dict[str, Any] | None) -> None:
    """usageをモデル別に合算してmetaに保存"""
    if not model or not usage:
        return
    totals = meta.setdefault("usage_totals", {})
    m = totals.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "calls": 0})
    it = usage.get("input_tokens")
    ot = usage.get("output_tokens")
    if isinstance(it, int):
        m["input_tokens"] += it
    if isinstance(ot, int):
        m["output_tokens"] += ot
    m["calls"] += 1


def extract_image_facts_single(
    img_info: dict,
    api_key: str,
    model: str,
    title: str = "",
    max_tokens: int = 700,
) -> dict[str, Any] | None:
    """画像1枚から“人物・関係性・主要イベント”を構造化抽出（大筋あらすじ用途）"""
    cache = _get_llm_cache()
    cache_key = _image_cache_key(img_info, model=model, prompt_key="facts_v1")
    if cache_key in cache:
        return cache[cache_key]

    base64_image, media_type = encode_image_to_base64(img_info)
    header = ""
    if title:
        header = f"参考タイトル: {title}\n"
    ep = img_info.get("episode", 1)
    page = img_info.get("page", 1)
    content = [
        {"type": "text", "text": f"{header}対象: 第{ep}話 P{page}\n以下の画像から情報を抽出してください。出力はJSONのみ。"},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": base64_image},
        },
        {
            "type": "text",
            "text": """要件:
- 目的は「大筋のあらすじ」を作ること。セリフの一字一句は不要。
- ただし「登場人物/関係性」「主要イベント」は取り違えると致命的なので慎重に。
- 推測で補完しない。読めない/不明は不明と書く。

出力(JSONのみ):
{
  "episode": <int>,
  "page": <int>,
  "characters": [
    {"name_or_role": "<人物名または役割>", "relation_terms": ["義母","夫",...], "evidence": "<根拠(短い引用or描写)>"} 
  ],
  "events": ["<主要イベント1>","<主要イベント2>"],
  "key_dialogue_quotes": ["<短い引用(任意)>"],
  "confidence": <0.0-1.0>,
  "uncertainties": ["<不確かな点>"]
}""",
        },
    ]

    try:
        text, usage = call_claude_messages_with_usage(
            api_key=api_key,
            model=model,
            content=content,
            max_tokens=max_tokens,
            temperature=0.2,
        )
    except Exception:
        cache[cache_key] = None
        return None

    json_block = _extract_json_block(text) or text
    facts = _safe_json_loads(json_block)
    if isinstance(facts, dict):
        # 足りないメタを補完
        facts.setdefault("episode", ep)
        facts.setdefault("page", page)
        facts["_usage"] = usage
        facts["_model"] = model
        cache[cache_key] = facts
        return facts

    cache[cache_key] = None
    return None


def extract_panel_details(
    images: list[dict],
    api_key: str,
    title: str = "",
    primary_model: str = "claude-sonnet-4-5-20251101",
    fallback_model: str = "claude-opus-4-5-20251101",
    max_tokens_per_image: int = 700,
    suspicious_confidence_threshold: float = 0.55,
    enable_text_verifier: bool = True,
    verifier_model: str = "claude-haiku-4-5-20251101",
    debug: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Step1: 画像→事実抽出（安価モデル中心、怪しい画像だけOpusへ）

    Returns:
        panel_details_text: Step2へ渡すテキスト
        meta: エスカレーション件数などのメタ情報
    """
    extracted: list[dict[str, Any]] = []
    suspicious_indices: list[int] = []
    suspicious_reasons: dict[int, list[str]] = {}
    meta: dict[str, Any] = {"usage_totals": {}}

    for idx, img_info in enumerate(images):
        facts = extract_image_facts_single(
            img_info=img_info,
            api_key=api_key,
            model=primary_model,
            title=title,
            max_tokens=max_tokens_per_image,
        )
        if facts is None:
            suspicious_indices.append(idx)
            suspicious_reasons[idx] = ["抽出失敗(None)"]
            extracted.append({
                "episode": img_info.get("episode", 1),
                "page": img_info.get("page", 1),
                "characters": [],
                "events": [],
                "key_dialogue_quotes": [],
                "confidence": 0.0,
                "uncertainties": ["抽出失敗"],
            })
            continue

        # しきい値判定
        if isinstance(facts.get("confidence"), (int, float)) and facts["confidence"] < suspicious_confidence_threshold:
            suspicious, reasons = True, [f"confidence<{suspicious_confidence_threshold}"]
        else:
            suspicious, reasons = _validate_image_facts(facts)

        if suspicious:
            suspicious_indices.append(idx)
            suspicious_reasons[idx] = reasons

        extracted.append(facts)
        _add_usage_totals(meta, facts.get("_model", ""), facts.get("_usage"))

    # 追加の検証（テキストのみ、安価モデル）: “怪しい画像候補”を増やす
    if enable_text_verifier and extracted:
        try:
            payload = json.dumps(extracted, ensure_ascii=False)
            verifier_content = [
                {
                    "type": "text",
                    "text": f"""以下は漫画画像から抽出したJSON一覧です。
目的は「大筋のあらすじ」。ただし「登場人物/関係性」「主要イベント」の取り違えは致命的です。

このJSON一覧を読み、明らかに不足・矛盾・テンプレ臭・不自然な飛躍がある画像(配列のindex)を列挙してください。
出力はJSONのみ: {{"suspicious_indexes":[0,2,...],"reasons":{{"0":["理由1",...],...}}}}

入力JSON:
{payload}"""
                }
            ]
            verifier_text, verifier_usage = call_claude_messages_with_usage(
                api_key=api_key,
                model=verifier_model,
                content=verifier_content,
                max_tokens=700,
                temperature=0.1,
            )
            _add_usage_totals(meta, verifier_model, verifier_usage)
            verifier_json = _safe_json_loads(_extract_json_block(verifier_text) or verifier_text)
            if isinstance(verifier_json, dict):
                extra = verifier_json.get("suspicious_indexes", [])
                if isinstance(extra, list):
                    for i in extra:
                        if isinstance(i, int) and 0 <= i < len(images) and i not in suspicious_indices:
                            suspicious_indices.append(i)
                            suspicious_reasons[i] = ["テキスト検証で要再確認"]
        except Exception:
            pass

    suspicious_indices = sorted(set(suspicious_indices))

    # 怪しい画像だけOpusへ再抽出（上書き）
    escalated = 0
    if fallback_model and fallback_model != primary_model:
        for idx in suspicious_indices:
            img_info = images[idx]
            facts_opus = extract_image_facts_single(
                img_info=img_info,
                api_key=api_key,
                model=fallback_model,
                title=title,
                max_tokens=max_tokens_per_image,
            )
            if facts_opus is not None:
                extracted[idx] = facts_opus
                _add_usage_totals(meta, facts_opus.get("_model", ""), facts_opus.get("_usage"))
                escalated += 1

    # Step2へ渡す“材料”をテキスト化（JSONでもよいが、ここは見やすさ優先）
    lines: list[str] = []
    if title:
        lines.append(f"【参考タイトル】{title}")
    lines.append("【画像ごとの抽出（人物/イベント中心）】")
    for i, facts in enumerate(extracted, start=1):
        ep = facts.get("episode", 1)
        page = facts.get("page", 1)
        chars = facts.get("characters", [])
        events = facts.get("events", [])
        uq = facts.get("uncertainties", [])
        conf = facts.get("confidence", None)
        lines.append(f"\n### 画像{i}（第{ep}話 P{page}）")
        lines.append(f"- confidence: {conf}")
        lines.append(f"- characters: {json.dumps(chars, ensure_ascii=False)}")
        lines.append(f"- events: {json.dumps(events, ensure_ascii=False)}")
        if uq:
            lines.append(f"- uncertainties: {json.dumps(uq, ensure_ascii=False)}")

    meta.update({
        "total_images": len(images),
        "suspicious_images": len(suspicious_indices),
        "escalated_to_opus": escalated,
        "suspicious_indices": suspicious_indices,
        "suspicious_reasons": suspicious_reasons,
        "primary_model": primary_model,
        "fallback_model": fallback_model,
        "verifier_model": verifier_model,
    })
    return "\n".join(lines), meta


def summarize_story(
    panel_details: str,
    api_key: str,
    title: str = "",
    model: str = "claude-opus-4-5-20251101",
) -> str:
    """Step2: 抽出した情報からストーリーをまとめる"""
    try:
        prompt = f"""以下は漫画画像から抽出した「人物/関係性/主要イベント」の材料です。
これを元に「大筋のあらすじ」を作ってください。

制約:
- 推測で補完しない。不明は不明として扱う
- ただし、人物/関係性/主要イベントを取り違えない（矛盾があれば慎重に）
- 感情表現は多少の幅があってよい

"""
        if title:
            prompt += f"【参考タイトル】\n{title}\n\n"

        prompt += f"""【抽出材料】
{panel_details}

---

出力形式:
## あらすじ
(3〜6文)

## 登場人物
(箇条書き。関係性も明記)

## 主要イベント
(箇条書き。時系列が分かるように)

## 不確かな点
(材料に不明/矛盾がある場合のみ)"""

        text = call_claude_messages(
            api_key=api_key,
            model=model,
            content=[{"type": "text", "text": prompt}],
            max_tokens=1300,
            temperature=0.2,
        )
        return text
    except Exception as e:
        return f"要約エラー: {str(e)}"


def analyze_images_batch(
    images: list[dict],
    api_key: str,
    title: str = "",
    primary_model: str = "claude-opus-4-5-20251101",
    fallback_model: str = "claude-opus-4-5-20251101",
    summary_model: str = "claude-opus-4-5-20251101",
    verifier_model: str = "claude-opus-4-5-20251101",
    max_tokens_per_image: int = 700,
    suspicious_confidence_threshold: float = 0.55,
    enable_text_verifier: bool = True,
    debug: bool = False,
) -> tuple[str, dict[str, Any]]:
    """2段階解析: セリフ抽出→ストーリーまとめ"""

    # Step 1: 各画像のセリフ・状況を詳細に抽出
    panel_details, meta = extract_panel_details(
        images=images,
        api_key=api_key,
        title=title,
        primary_model=primary_model,
        fallback_model=fallback_model,
        max_tokens_per_image=max_tokens_per_image,
        suspicious_confidence_threshold=suspicious_confidence_threshold,
        enable_text_verifier=enable_text_verifier,
        verifier_model=verifier_model,
        debug=debug,
    )

    # Step 2: 抽出した情報からストーリーをまとめる（usageも収集）
    try:
        prompt = f"""以下は漫画画像から抽出した「人物/関係性/主要イベント」の材料です。
これを元に「大筋のあらすじ」を作ってください。

制約:
- 推測で補完しない。不明は不明として扱う
- ただし、人物/関係性/主要イベントを取り違えない（矛盾があれば慎重に）
- 感情表現は多少の幅があってよい

"""
        if title:
            prompt += f"【参考タイトル】\n{title}\n\n"

        prompt += f"""【抽出材料】
{panel_details}

---

出力形式:
## あらすじ
(3〜6文)

## 登場人物
(箇条書き。関係性も明記)

## 主要イベント
(箇条書き。時系列が分かるように)

## 不確かな点
(材料に不明/矛盾がある場合のみ)"""

        summary, summary_usage = call_claude_messages_with_usage(
            api_key=api_key,
            model=summary_model,
            content=[{"type": "text", "text": prompt}],
            max_tokens=1300,
            temperature=0.2,
        )
        _add_usage_totals(meta, summary_model, summary_usage)
    except Exception as e:
        summary = f"要約エラー: {str(e)}"

    # デバッグ用（長文は重いのでプレビューのみ）
    meta["summary_model"] = summary_model
    meta["panel_details_preview"] = panel_details[:4000]

    return summary, meta


def check_title_consistency(
    title: str,
    summary: str,
    api_key: str,
    model: str = "claude-opus-4-5-20251101",
) -> str:
    """タイトルとあらすじの整合性をチェック"""
    prompt = f"""以下の漫画記事の「タイトル」と「あらすじ」を比較して、整合性をチェックしてください。

【タイトル】
{title}

【あらすじ】
{summary}

---

以下の観点でチェックし、結果を報告してください：

## 整合性チェック結果

### 判定: [◯ 整合 / △ 軽微な違和感 / ✕ 不整合]

### チェック項目

1. **テーマの一致**: タイトルが示すテーマとあらすじの内容は一致していますか？
2. **登場人物**: タイトルに人物や関係性が含まれる場合、あらすじと一致していますか？
3. **結末・教訓**: タイトルが示唆する結末や教訓は、あらすじに反映されていますか？
4. **誇大表現**: タイトルが内容を誇張しすぎていませんか？

### 詳細コメント
（違和感がある場合は具体的に指摘してください）

### 改善提案
（タイトルの改善案があれば提案してください）"""

    try:
        text = call_claude_messages(
            api_key=api_key,
            model=model,
            content=[{"type": "text", "text": prompt}],
            max_tokens=700,
            temperature=0.2,
        )
        return text
    except Exception as e:
        return f"チェックエラー: {str(e)}"


# サイドバー設定
with st.sidebar:
    st.header("⚙️ 設定")

    # APIキーの取得元を確認
    env_key = get_api_key_from_env()
    secrets_key = get_api_key_from_secrets()

    if env_key:
        # .envに設定済みの場合
        st.success("APIキー設定済み（.env）")
        api_key = env_key
    elif secrets_key:
        # Secretsに設定済みの場合
        st.success("APIキー設定済み（Secrets）")
        api_key = secrets_key
    else:
        # ユーザー入力モード
        api_key_input = st.text_input(
            "Anthropic API Key",
            type="password",
            value=st.session_state.get("user_api_key", ""),
            help="Claude APIキーを入力してください",
            key="api_key_input"
        )

        if st.button("🔐 APIキーを設定", use_container_width=True):
            if api_key_input:
                st.session_state["user_api_key"] = api_key_input
                st.success("セッションに保存しました")
                st.rerun()
            else:
                st.warning("APIキーを入力してください")

        if st.session_state.get("user_api_key"):
            st.info("APIキー設定済み（セッション）")
            if st.button("🗑️ クリア", use_container_width=True):
                del st.session_state["user_api_key"]
                st.rerun()

        api_key = st.session_state.get("user_api_key", "")

    st.divider()

    st.subheader("画像フィルタ設定")
    min_image_size = st.slider(
        "最小画像サイズ (KB)",
        min_value=1,
        max_value=500,
        value=30,
        help="この値より小さい画像は除外されます"
    )

    debug_mode = st.checkbox("デバッグモード", value=True, help="画像検出の詳細を表示")

    st.divider()
    st.subheader("💰 コスト最適化（重要）")

    st.markdown("**方針**: 画像は縮小して送信し、抽出は安価モデル中心→怪しい画像だけOpusへ再抽出します。")

    preprocess_max_side = st.slider(
        "LLM送信用 画像最大辺(px)",
        min_value=512,
        max_value=1600,
        value=1024,
        step=64,
        help="小さいほど安くなりやすい（ただし文字が潰れると精度低下）"
    )
    preprocess_jpeg_quality = st.slider(
        "LLM送信用 JPEG品質",
        min_value=40,
        max_value=90,
        value=70,
        step=5,
        help="小さいほど安くなりやすい（ただし文字が潰れると精度低下）"
    )
    max_images_total = st.slider(
        "解析に使う最大画像枚数（上限）",
        min_value=5,
        max_value=120,
        value=45,
        step=5,
        help="多いほど精度は上がり得ますが、コストが直線的に増えます"
    )

    st.subheader("🤖 モデル設定")
    available_models = get_available_anthropic_models(api_key) if api_key else []
    if available_models:
        st.caption("✅ APIキーで利用可能なモデル一覧を取得しました（404を避けるため、ここから選ぶのがおすすめです）")
    else:
        st.caption("ℹ️ 利用可能モデル一覧を取得できませんでした。モデル名は手入力してください（404が出る場合はモデル名が違います）。")

    default_opus = "claude-opus-4-5-20251101"

    if available_models:
        primary_model = st.selectbox(
            "抽出（一次）モデル",
            options=available_models,
            index=0 if default_opus not in available_models else available_models.index(default_opus),
            help="画像→人物/イベント抽出の一次モデル（基本はここを安価に）"
        )
    else:
        primary_model = st.text_input(
            "抽出（一次）モデル",
            value=default_opus,
            help="画像→人物/イベント抽出の一次モデル（基本はここを安価に）"
        )
    enable_fallback_opus = st.checkbox("怪しい画像だけ高精度モデルへ再抽出（推奨）", value=True)
    if available_models:
        fallback_model = st.selectbox(
            "抽出（再抽出）モデル",
            options=available_models,
            index=0 if default_opus not in available_models else available_models.index(default_opus),
            help="一次抽出が怪しい時だけ使うモデル（Opusなど）"
        )
    else:
        fallback_model = st.text_input(
            "抽出（再抽出）モデル",
            value=default_opus,
            help="一次抽出が怪しい時だけ使うモデル（Opusなど）"
        )

    if available_models:
        summary_model = st.selectbox(
            "要約モデル",
            options=available_models,
            index=0 if default_opus not in available_models else available_models.index(default_opus),
            help="画像抽出後のテキスト要約なので、基本は安価モデルでOK（安いモデルが使えるなら切替推奨）"
        )
        verifier_model = st.selectbox(
            "テキスト検証モデル（任意）",
            options=available_models,
            index=0 if default_opus not in available_models else available_models.index(default_opus),
            help="抽出JSONの不足/矛盾をテキストだけで検知（安いモデル推奨）"
        )
        consistency_model = st.selectbox(
            "タイトル整合性チェックモデル",
            options=available_models,
            index=0 if default_opus not in available_models else available_models.index(default_opus),
            help="テキストのみのチェック。安いモデルで十分"
        )
    else:
        summary_model = st.text_input(
            "要約モデル",
            value=default_opus,
            help="画像抽出後のテキスト要約なので、基本は安価モデルでOK（安いモデルが使えるなら切替推奨）"
        )
        verifier_model = st.text_input(
            "テキスト検証モデル（任意）",
            value=default_opus,
            help="抽出JSONの不足/矛盾をテキストだけで検知（安いモデル推奨）"
        )
        consistency_model = st.text_input(
            "タイトル整合性チェックモデル",
            value=default_opus,
            help="テキストのみのチェック。安いモデルで十分"
        )

    st.subheader("🔎 検知パラメータ")
    max_tokens_per_image = st.slider(
        "画像1枚あたりの最大出力トークン",
        min_value=200,
        max_value=1400,
        value=700,
        step=50,
        help="大きいほど情報が増える可能性はあるが、コストも増える"
    )
    suspicious_confidence_threshold = st.slider(
        "confidenceしきい値（これ未満は再抽出候補）",
        min_value=0.30,
        max_value=0.80,
        value=0.55,
        step=0.05,
    )
    enable_text_verifier = st.checkbox(
        "テキスト検証で“怪しい画像”候補を追加（推奨）",
        value=True,
        help="画像は見ず、抽出結果(JSON)だけを安価モデルでチェックします"
    )

# メインコンテンツ
st.subheader("📖 漫画タイプを選択してURLを入力")

manga_type_col1, manga_type_col2 = st.columns(2)

with manga_type_col1:
    st.markdown("**📚 連載漫画**（3話分読み込み）")
    serial_url = st.text_input(
        "連載漫画URL",
        placeholder="https://example.com/serial-manga",
        help="連載漫画のURLを入力（3話分の画像を読み込みます）",
        label_visibility="collapsed"
    )

with manga_type_col2:
    st.markdown("**📄 エピ漫画**（1話分読み込み）")
    episode_url = st.text_input(
        "エピ漫画URL",
        placeholder="https://example.com/episode-manga",
        help="エピソード漫画のURLを入力（1話分の画像を読み込みます）",
        label_visibility="collapsed"
    )

# どちらのURLが入力されたか判定
url = ""
num_episodes = 1
manga_type_label = ""

if serial_url and episode_url:
    st.warning("⚠️ どちらか一方のURLのみ入力してください")
elif serial_url:
    url = serial_url
    num_episodes = 3
    manga_type_label = "連載漫画"
elif episode_url:
    url = episode_url
    num_episodes = 1
    manga_type_label = "エピ漫画"

article_title = st.text_input(
    "📰 記事タイトル（任意）",
    placeholder="漫画記事のタイトルを入力",
    help="タイトルを入力すると、あらすじとの整合性をチェックします"
)

col1, col2 = st.columns([1, 4])
with col1:
    analyze_button = st.button("🔍 解析開始", type="primary", use_container_width=True)

if analyze_button:
    if not url:
        st.error("URLを入力してください（連載漫画またはエピ漫画のどちらか）")
    elif serial_url and episode_url:
        st.error("どちらか一方のURLのみ入力してください")
    elif not api_key:
        st.error("APIキーを設定してください")
    else:
        st.info(f"📖 **{manga_type_label}**として解析します（{num_episodes}話分）")

        with st.spinner("ページから画像を取得中..."):
            # 新しいロジック: 話数単位で取得
            images = get_multiple_episodes_images(url, num_episodes=num_episodes, debug=debug_mode)

        if not images:
            st.warning("画像が見つかりませんでした。デバッグモードをONにして詳細を確認してください。")
        else:
            st.info(f"📷 {len(images)}件の画像を検出しました。漫画画像をフィルタリング中...")

            with st.spinner("漫画画像をフィルタリング中..."):
                manga_images = filter_manga_images(
                    images,
                    min_size=min_image_size * 1000,
                    referer=url,
                    debug=debug_mode,
                    preprocess_max_side=preprocess_max_side,
                    preprocess_jpeg_quality=preprocess_jpeg_quality,
                )

            if not manga_images:
                st.warning("漫画画像が見つかりませんでした。フィルタ設定を調整してみてください。")

                if debug_mode and images:
                    st.subheader("検出された画像URL一覧")
                    for img in images:
                        st.text(img["url"])
            else:
                # コスト暴発防止: 最大枚数でカット
                if len(manga_images) > max_images_total:
                    st.warning(f"⚠️ 画像が{len(manga_images)}枚あります。コスト抑制のため先頭{max_images_total}枚だけで解析します。")
                    manga_images = manga_images[:max_images_total]

                # 話数ごとの画像数を集計
                episode_counts = {}
                for img in manga_images:
                    ep = img.get("episode", 1)
                    episode_counts[ep] = episode_counts.get(ep, 0) + 1

                episode_summary = "、".join([f"第{ep}話: {count}枚" for ep, count in sorted(episode_counts.items())])
                st.success(f"📚 {len(manga_images)}件の漫画画像を検出しました（{episode_summary}）")

                # 画像を表示
                st.header("🖼️ 検出された漫画画像")

                # グリッド表示（話数ごとにグループ化）
                cols_per_row = 3
                for i in range(0, len(manga_images), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(manga_images):
                            img_info = manga_images[idx]
                            with col:
                                page_num = img_info.get("page", 1)
                                episode_num = img_info.get("episode", 1)
                                st.image(
                                    img_info["data"],
                                    caption=f"第{episode_num}話 P{page_num}",
                                    use_container_width=True
                                )

                # あらすじ解析
                st.divider()
                st.header("📝 あらすじ解析")

                with st.spinner("AIがあらすじを解析中..."):
                    used_fallback_model = fallback_model if enable_fallback_opus else primary_model
                    summary, meta = analyze_images_batch(
                        manga_images,
                        api_key,
                        title=article_title,
                        primary_model=primary_model,
                        fallback_model=used_fallback_model,
                        summary_model=summary_model,
                        verifier_model=verifier_model,
                        max_tokens_per_image=max_tokens_per_image,
                        suspicious_confidence_threshold=suspicious_confidence_threshold,
                        enable_text_verifier=enable_text_verifier,
                        debug=debug_mode,
                    )

                st.markdown(summary)

                with st.expander("🔧 解析メタ（コスト/品質の参考）", expanded=False):
                    st.write(f"総画像: {meta.get('total_images')} / 怪しい判定: {meta.get('suspicious_images')} / 再抽出: {meta.get('escalated_to_opus')}")
                    st.write(f"一次モデル: {meta.get('primary_model')}")
                    st.write(f"再抽出モデル: {meta.get('fallback_model')}")
                    st.write(f"検証モデル: {meta.get('verifier_model')}")
                    st.write(f"要約モデル: {meta.get('summary_model')}")
                    idxs = meta.get("suspicious_indices", [])
                    if idxs:
                        st.write("怪しい画像index（0始まり）:", idxs)
                        st.json(meta.get("suspicious_reasons", {}))
                    preview = meta.get("panel_details_preview")
                    if preview:
                        st.divider()
                        st.caption("抽出材料プレビュー（先頭のみ）: ここが薄い/不明だと、あらすじも薄くなります")
                        st.text_area("panel_details_preview", value=preview, height=240)

                    totals = meta.get("usage_totals")
                    if isinstance(totals, dict) and totals:
                        st.divider()
                        st.caption("usage集計（モデル別）: ここからコストを推定できます")
                        st.json(totals)

                        st.subheader("💵 コスト推定（任意）")
                        st.caption("単価はあなたの契約/請求単価に合わせて入力してください（$ / 1M tokens）。")
                        usd_jpy = st.number_input("換算レート（USD→JPY）", min_value=50.0, max_value=300.0, value=150.0, step=1.0)
                        default_in = st.number_input("入力単価（$/1M tokens）(共通)", min_value=0.0, value=0.0, step=0.5)
                        default_out = st.number_input("出力単価（$/1M tokens）(共通)", min_value=0.0, value=0.0, step=0.5)

                        est_usd = 0.0
                        for mname, v in totals.items():
                            it = v.get("input_tokens", 0) or 0
                            ot = v.get("output_tokens", 0) or 0
                            if not isinstance(it, int):
                                it = 0
                            if not isinstance(ot, int):
                                ot = 0
                            est_usd += (it / 1_000_000.0) * float(default_in) + (ot / 1_000_000.0) * float(default_out)
                        st.write(f"推定コスト: **約 ${est_usd:.4f}（約 ¥{est_usd * float(usd_jpy):.1f}）**")

                # タイトルとの整合性チェック
                if article_title:
                    st.divider()
                    st.header("🔍 タイトル整合性チェック")

                    with st.spinner("タイトルとあらすじの整合性をチェック中..."):
                        consistency_result = check_title_consistency(
                            article_title,
                            summary,
                            api_key,
                            model=consistency_model,
                        )

                    st.markdown(consistency_result)

# フッター
st.divider()
st.caption("💡 ヒント: 記事タイトルを入力すると、あらすじとの整合性を自動チェックします")
