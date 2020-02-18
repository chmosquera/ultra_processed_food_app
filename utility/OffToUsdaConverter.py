import fdc
import requests
import pickle
from fdc import FdcClient
from os import environ
from sys import stderr
from datetime import date

DISCARD_CACHE = ".~discard_cache.pkl"
CACHE_SAVE_INTERVAL = 500
MATCH_LOG = "ff_to_usda_matching.log"
ENABLE_MATCH_LOG = True

class OffToUsdaConverter:
    def __init__(self, header: list, minimumScore = 100.0):
        self.client = FdcClient(environ['USDA_API_KEY'])
        self.minimumScore = minimumScore
        self.totalItr = 0
        self.matchLog = None
        self.discardCount = 0
        self.header = header
        self.outputHeader = ["code", "product_name", "nova_group", "ingredients"]
        self.data = []
        self.init_discard_cache(DISCARD_CACHE)
        self.codeidx, self.countryidx, self.novaidx, self.nameidx, self.brandsidx = get_header_indices(header)

    def convert(self, csvprods:list):
        if(ENABLE_MATCH_LOG):
            self.matchLog = open(MATCH_LOG, 'a')
            self.matchLog.write(f"Starting {date.today().ctime()}, with minimum score {self.minimumScore}\n")

        try:
            for product in csvprods:
                self.match_product(product)
        except KeyboardInterrupt as e:
            print("Caught Interupt. Exiting gracefully...")
            pass
        except requests.exceptions.HTTPError as e:
            print("HTTP error: You probably need to give the API a break", file=stderr)
            print(e)

        finalReport = self.format_progress()
        print("Finish:", finalReport)

        print("Saving discard cache...")
        self.write_discard_cache(DISCARD_CACHE)

        if(ENABLE_MATCH_LOG):
            finish = f"Finished {date.today().ctime()}, " + finalReport
            self.matchLog.write("\n\n" + finish + "\n")
            self.matchLog.close()

        print("\n")
        return(self.outputHeader,self.data)

    def match_product(self, product):
        self.totalItr += 1
        print(f"Found {self.totalItr - self.discardCount} matches and discarded {self.discardCount} after {self.totalItr} entries", end = '\r')

        if(self.totalItr % CACHE_SAVE_INTERVAL == 0):
            self.write_discard_cache(DISCARD_CACHE)
            if(ENABLE_MATCH_LOG):
                self.matchLog.flush()

        if(product[self.codeidx] in self.discardCache):
            self.discardCount += 1
            return
        elif(product[self.novaidx] == ''):
            self.discard_product(product[self.codeidx])
            return

        # Look for an exact match first, then a best match if no exact match is found
        try:
            self.attempt_exact_match(product) # Look for exact match by barcode id
        except (StopIteration, KeyError) as e:
            try:
                self.attempt_best_match(product)
            except (StopIteration, KeyError):
                self.discard_product(product[self.codeidx])

    def attempt_exact_match(self, product):
        exactMatch = next(self.client.search(product[self.codeidx], productLimit=1))
        self.data.append([product[self.codeidx], product[self.nameidx], product[self.novaidx], exactMatch['ingredients']])

    def attempt_best_match(self, product):
        bestMatch = next(self.client.search(product[self.nameidx].upper(), productLimit=1, brandOwner=product[self.brandsidx].upper()))
        if(ENABLE_MATCH_LOG):
            entry = self.format_log_entry(product, bestMatch)
            self.matchLog.write(entry)
        if(float(bestMatch['score']) >= self.minimumScore):
            self.data.append([product[self.codeidx], product[self.nameidx], product[self.novaidx], bestMatch['ingredients']])
        else:
            self.discard_product(product[self.codeidx])

    def discard_product(self, code: str):
        self.discardCache.add(code)
        self.discardCount += 1

    def init_discard_cache(self, cachePath: str):
        try:
            pickleJar = open(DISCARD_CACHE, 'rb');
            self.discardCache = pickle.load(pickleJar)
            pickleJar.close()
        except Exception as e:
            self.discardCache = set()
            pass

    def write_discard_cache(self, cachePath: str):
        pickleJar = open(cachePath, 'wb')
        pickle.dump(self.discardCache,  pickleJar)
        pickleJar.close()

    def format_progress(self):
        return(f"Found {self.totalItr - self.discardCount} matches and discarded {self.discardCount} after {self.totalItr} entries")

    def format_log_entry(self, product, bestMatch):
        goodEnough = float(bestMatch['score']) >= self.minimumScore
        productPart = f"Best match for '{product[self.nameidx].upper()}' from '{product[self.brandsidx].upper()}' is"
        matchPart = f"'{bestMatch['description']}' from '{bestMatch['brandOwner']}'"
        scorePart = f"| score: {bestMatch['score']} -> {'ACCEPTED' if goodEnough else 'REJECTED'}" 

        entry = ' '.join([productPart, matchPart, scorePart]) + '\n'
        return(entry)


def get_header_indices(header: list):
    codeidx = header.index('code')
    countryidx = header.index('countries')
    novaidx = header.index('nova_group')
    nameidx = header.index('product_name')
    brandsidx = header.index('brands')
    
    return(codeidx, countryidx, novaidx, nameidx, brandsidx)