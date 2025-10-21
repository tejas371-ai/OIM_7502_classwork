import scrapy


class Sp500Spider(scrapy.Spider):
    name = "sp500"
    allowed_domains = ["slickcharts.com"]
    start_urls = ["https://www.slickcharts.com/sp500/performance"]

    def parse(self, response):
        rows = response.xpath('//table[contains(@class, "table")]/tbody/tr')

        for row in rows:
            ytd_raw = row.xpath('normalize-space(td[last()]/text())').get()
            # Clean up non-breaking spaces and extra characters
            if ytd_raw:
                ytd_clean = ytd_raw.replace('\xa0', '').replace(' ', '')
            else:
                ytd_clean = None

            yield {
                'Number': row.xpath('normalize-space(td[1]/text())').get(),
                'Company': row.xpath('normalize-space(td[2]/a/text())').get(),
                'Symbol': row.xpath('normalize-space(td[3]/a/text())').get(),
                'YTD Return': ytd_clean
            }
