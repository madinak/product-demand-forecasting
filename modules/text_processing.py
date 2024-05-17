import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday
from pandas.tseries.offsets import CustomBusinessDay

class TextProcessor:
    def __init__(self, df, max_features=5000, num_clusters=20):
        self.df = df
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        self.stop_words = set(['red', 'blue', 'green', 'yellow', 'black', 'white', 'pink',
                               'large', 'small', 'medium', 'mini', 'maxi'])  # Stop words to be removed

    def preprocess_text(self, text):
        # Convert text to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove stop words
        words = [word for word in text.split() if word not in self.stop_words]
        # Extract last two words
        keywords = words[-2:]
        return ' '.join(keywords) if len(keywords) > 0 else text

    def fit_transform(self):
        # Preprocess text data in the Description column
        self.df['Preprocessed_Description'] = self.df['Description'].apply(self.preprocess_text)
        # Convert preprocessed text data into numerical features
        X = self.vectorizer.fit_transform(self.df['Preprocessed_Description'])
        return X

    def fit_predict(self):
        # Fit clustering algorithm on preprocessed text data
        clusters = self.kmeans.fit_predict(self.vectorizer.fit_transform(self.df['Preprocessed_Description']))
        # Assign cluster labels to the DataFrame
        self.df['Cluster'] = clusters
        return self.df
    

class ProductCategorizer:
    def __init__(self, df):
        self.df = df
        self.keywords = {
            'Home Accessories': ['CANDLE', 'CANDLE PLATE', 'CANDLEHOLDER', 'DOOR', 'MAT', 'DRAWER', 'LANTERN', 'LIGHT',
                     'LAMP', 'TORCH', 'HANGER', 'RACK', 'STORAGE BOX', 'HOOK', 'WICKER', 'JUG', 'CLOTHES PEGS',
                     'ALARM CLOCK', 'QUILT', 'CUSHION', 'CUSHION COVER', 'MIRROR', 'CRATE', 'BASKET', 'WALL CLOCK',
                     'WALL', 'CLOCK', 'SIGN', 'METAL SIGN', 'DOOR SIGN'],
            'Kitchenware': ['COFFEE', 'COFFEE GRINDER', 'CUTLERY', 'MUG', 'RECIPE', 'JAR', 'CAKE', 'CAKE CASES',
                            'CAKE STAND', 'POT', 'TEASPOON', 'SPOON', 'FORK', 'PLATE', 'PLATTER', 'CUP', 'SAUCER',
                            'CAKE TIN', 'BUCKET', 'BREAD', 'FRIDGE', 'BOTTLE', 'TRAY', 'MATCHES', 'KITCHEN', 'TISSUES',
                            'STRAWS', 'FOOD COVER', 'TEA TOWEL', 'PANNETONE', 'BOWL', 'RECIPE', 'OVEN MITT',
                            'BAKING SET', 'LADLE', 'WATER BOTTLE', 'LUNCH BOX', 'SNACK BOX'],
            'Garden': ['WATERING CAN', 'PICNIC', 'LADDER', 'GARDEN', 'GARDENING', 'HERB MAKER', 'PLANT POT', 'PLANTER',
                       'BIRD FEEDER', 'WREATH'],
            'Accessories': ['EARRINGS', 'BRACELET', 'PARASOL', 'UMBRELLA', 'HAND WARMER', 'HAT', 'BANGLE', 'HAIRBAND',
                             'HAIR GRIP', 'HAIRCLIP', 'FIRST AID TIN', 'RING', 'KEY RING', 'NECKLACE', 'UMBRELLA',
                             'SHOES', 'SLIPPERS'],
            'Office and School': ['BOOK', 'NOTEBOOK', 'PEN', 'PENCIL', 'ERASER', 'PHOTO FRAME', 'FRAME', 'STICKER SHEET',
                                   'STICKERS', 'WASTEPAPER BIN', 'MONEY BANK', 'PIGGY BANK', 'BLACK BOARD', 'BOARD', 'CHALKBOARD',
                                   'CALENDAR', 'ENVELOP', 'CARD', 'POSTCARD', 'GREETING CARD', 'BIRTHDAY CARD', 'CARD KIT',
                                   'CRAYON', 'GLOBE'],
            'Bags': ['BAG', 'JUMBO BAG', 'PURSE', 'CARD HOLDER', 'LUGGAGE', 'SHOPPER', 'HANDBAG', 'PASSPORT COVER',
                     'LUNCH BAG'],
            'Children Toys': ['DOLL', 'BABUSHKA', 'PLAYHOUSE', 'BLOCKS', 'BLOCK WORDS', 'PUZZLE', 'SPACEBOY', 'TOY', 'DOMINOES',
                      'SOLDIER', 'SKIPPING ROPE', 'CONES', 'MAGIC', 'ENCHANTED', 'GAME', 'SCISSORS', 'PLAY', 'HELICOPTER',
                      'TEDDY', 'CHILDREN', 'BOY', 'GIRL'],
            'Arts, Crafts, and Decorations': ['PICTURE FRAME', 'FELT', 'FLOWER', 'DECORATION', 'COLOURED',
                                                      'SEWING KIT', 'SEWING BOX', 'REPAIR', 'CRAFT', 'DECORATION', 'RIBBONS',
                                                      'DOILIES', 'DECORATION', 'WRAP', 'SPINNING', 'BUNTING', 'NAPKIN',
                                                      'CONFETTI', 'ORNAMENT', 'PAPER CHAIN KIT', 'STRIPES', 'LACE', 'FLANNEL',
                                                      'PAISLEY', 'HEART', 'FELT HEART', 'FABRIC', 'T-LIGHTS', 'STRING', 'GIFT',
                                                      'CHRISTMAS', 'SNOWMEN', 'SANTA', 'XMAS']
        }
        self.cluster_mapping = {
            0: 'Arts, Crafts, and Decorations',
            1: 'Kitchenware',
            2: 'Home Accessories',
            3: 'Home Accessories',
            4: 'Home Accessories',
            5: 'Arts, Crafts, and Decorations',
            6: 'Kitchenware',
            7: 'Kitchenware',
            8: 'Home Accessories',
            9: 'Home Accessories',
            10: 'Arts, Crafts, and Decorations',
            11: 'Accessories',
            12: 'Accessories',
            13: 'Arts, Crafts, and Decorations',
            14: 'Arts, Crafts, and Decorations',
            15: 'Home Accessories',
            16: 'Bags',
            17: 'Home Accessories',
            18: 'Home Accessories',
            19: 'Office and School'
        }

    def assign_category(self, description):
        for category, words in self.keywords.items():
            for word in words:
                if word in description:
                    return category
        return None

    def assign_category_cluster(self, cluster):
        return self.cluster_mapping.get(cluster, None)

    def predict_categories(self):
        categories = []

        for index, row in self.df.iterrows():
            category = self.assign_category(row['Description'])

            if category is None:
                category = self.assign_category_cluster(row['Cluster'])

            categories.append(category)

        return categories