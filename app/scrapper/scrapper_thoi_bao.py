import requests
from bs4 import BeautifulSoup
import csv
from time import sleep
from common import Data

# URL trang chủ
base_url = "https://thoibaotaichinhvietnam.vn/chung-khoan&s_cond=&BRSR="

# Giả lập trình duyệt để tránh bị chặn
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}


def get_article_links(url):
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("Không thể truy cập trang chủ:", response.status_code)
        return []

    soup = BeautifulSoup(response.content, "html.parser")

    article_links = []
    listRes = soup.find(
        "div",
        class_="cat-listing bg-dots mt20 pt20 article-bdt-20 thumb-w250 title-22 no-catname",
    )
    res = listRes.find("div", class_="cat-content")
    # print(listRes)

    all_hrefs = res.find_all("article", class_="article")
    for href in all_hrefs:
        article_links.append(href.find("a", class_="article-thumb")["href"])

    return article_links


# Hàm cào dữ liệu từ một bài viết
def scrape_article(url):
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Không thể truy cập {url}: {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, "html.parser")

    # Lấy tiêu đề (thẻ h1 thường là tiêu đề bài viết)
    # title = soup.find('h1', class_='post-title')  # Điều chỉnh class nếu cần
    # title_text = title.text.strip() if title else "Không tìm thấy tiêu đề"

    # Lấy ngày đăng (thường trong thẻ time hoặc div meta)
    date = soup.find("span", class_="format_date")  # Hoặc class như 'article-date'
    date_text = date.text.strip() if date else "Không tìm thấy ngày"
    div = soup.find("div", class_="post-content __MASTERCMS_CONTENT")

    if div:
        paragraphs = div.find_all("p")
        content_text = "\n".join([p.text.strip() for p in paragraphs])

    else:
        print("Không tìm thấy thẻ div với class mong muốn.")
    return Data(date=date_text, content=content_text)


def scrape_thoi_bao():
    # Lấy danh sách bài viết
    article_links = []
    numbers = list(range(0, 200, 15))

    result = []
    for i in numbers:
        curr_url = base_url + str(i)
        article_link = get_article_links(curr_url)
        article_links = article_links + article_link
    for i, link in enumerate(article_links, 1):
        print(f"Đang cào bài {i}/{len(article_links)}: {link}")
        result.append(scrape_article(link))
        # sleep(1)  # Nghỉ 1 giây để tránh bị chặn
    print(f"Ngày muộn nhất cào được: {result[-1].date}")
    return result
