class Review:

    def __init__(
        self,
        text,
        rating,
        username,
        product_name,
        source_url
    ):
        self.text = text
        self.rating = rating
        self.username = username
        self.product_name = product_name
        self.source_url = source_url

    def to_dict(self):
        return {
            "noi_dung": self.text,
            "so_sao": self.rating,
            "nguoi_danh_gia": self.username,
            "san_pham": self.product_name,
            "link": self.source_url
        }