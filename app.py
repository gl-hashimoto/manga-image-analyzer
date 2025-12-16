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
from dotenv import load_dotenv

# .envファイルを読み込み
load_dotenv()


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


def filter_manga_images(images: list[dict], min_size: int = 50000, referer: str = "", debug: bool = False) -> list[dict]:
    """漫画画像をフィルタリング"""
    manga_images = []

    for img_info in images:
        img_data = download_image(img_info["url"], referer)
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

            manga_images.append({
                **img_info,
                "data": img_data,
                "width": width,
                "height": height,
                "size": len(img_data)
            })

            if debug:
                st.write(f"✅ 漫画画像として追加: {width}x{height}, {len(img_data)} bytes")

        except Exception as e:
            if debug:
                st.write(f"画像処理エラー: {e}")
            continue

    return manga_images


def encode_image_to_base64(img_info: dict) -> tuple[str, str]:
    """画像をbase64エンコード"""
    img = Image.open(BytesIO(img_info["data"]))
    format_map = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "GIF": "image/gif",
        "WEBP": "image/webp"
    }
    media_type = format_map.get(img.format, "image/jpeg")
    base64_image = base64.standard_b64encode(img_info["data"]).decode("utf-8")
    return base64_image, media_type


def extract_panel_details(images: list[dict], api_key: str, title: str = "") -> str:
    """Step1: 各画像のセリフ・状況を詳細に抽出"""
    client = anthropic.Anthropic(api_key=api_key)

    content = []

    # タイトルがある場合は参考情報として先に追加
    if title:
        content.append({
            "type": "text",
            "text": f"【参考：記事タイトル】\n{title}\n\n"
        })

    for i, img_info in enumerate(images):
        base64_image, media_type = encode_image_to_base64(img_info)

        content.append({
            "type": "text",
            "text": f"【画像 {i+1}】"
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64_image,
            },
        })

    content.append({
        "type": "text",
        "text": """各画像について、以下の情報を正確に抽出してください。

【重要】
- 吹き出し内のセリフは一字一句正確に書き起こしてください
- 誰が誰に話しているか明確にしてください
- 登場人物の呼び方（「お義母さん」「ママ」「お母さん」等）に注目し、関係性を判断してください
- キャラクターの外見の特徴（年齢層、髪型、服装）を記録してください

各画像について以下の形式で出力：

### 画像X
- **セリフ**: （吹き出し内のセリフを全て書き起こし。誰の発言か明記）
- **状況**: （何が起きているか）
- **登場人物**: （この画像に登場する人物と特徴）
- **感情/表情**: （キャラクターの感情状態）"""
    })

    try:
        message = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=4096,
            messages=[{"role": "user", "content": content}],
        )
        return message.content[0].text
    except Exception as e:
        return f"抽出エラー: {str(e)}"


def summarize_story(panel_details: str, api_key: str, title: str = "") -> str:
    """Step2: 抽出した情報からストーリーをまとめる"""
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""以下は漫画の各コマから抽出したセリフと状況の詳細です。
これを元に、ストーリーを正確にまとめてください。

"""
    if title:
        prompt += f"【記事タイトル】\n{title}\n\n"

    prompt += f"""【抽出された詳細情報】
{panel_details}

---

上記の情報を元に、以下の形式でまとめてください：

## あらすじ
（ストーリーの流れを3〜5文で説明。セリフの内容を反映し、登場人物の関係性を明確に記載）

## 登場人物
（主要キャラクターを箇条書きで。関係性も明記。例：「義母（主人公の夫の母）」）

## ポイント
（この漫画の見どころや教訓を1〜2文で）

## 主要なセリフ
（ストーリーを理解する上で重要なセリフを2〜3個引用）"""

    try:
        message = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return f"要約エラー: {str(e)}"


def analyze_images_batch(images: list[dict], api_key: str, title: str = "") -> str:
    """2段階解析: セリフ抽出→ストーリーまとめ"""

    # Step 1: 各画像のセリフ・状況を詳細に抽出
    panel_details = extract_panel_details(images, api_key, title)

    if panel_details.startswith("抽出エラー"):
        return panel_details

    # Step 2: 抽出した情報からストーリーをまとめる
    summary = summarize_story(panel_details, api_key, title)

    return summary


def check_title_consistency(title: str, summary: str, api_key: str) -> str:
    """タイトルとあらすじの整合性をチェック"""
    client = anthropic.Anthropic(api_key=api_key)

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
        message = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return message.content[0].text
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
                    debug=debug_mode
                )

            if not manga_images:
                st.warning("漫画画像が見つかりませんでした。フィルタ設定を調整してみてください。")

                if debug_mode and images:
                    st.subheader("検出された画像URL一覧")
                    for img in images:
                        st.text(img["url"])
            else:
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
                    summary = analyze_images_batch(manga_images, api_key, title=article_title)

                st.markdown(summary)

                # タイトルとの整合性チェック
                if article_title:
                    st.divider()
                    st.header("🔍 タイトル整合性チェック")

                    with st.spinner("タイトルとあらすじの整合性をチェック中..."):
                        consistency_result = check_title_consistency(article_title, summary, api_key)

                    st.markdown(consistency_result)

# フッター
st.divider()
st.caption("💡 ヒント: 記事タイトルを入力すると、あらすじとの整合性を自動チェックします")
