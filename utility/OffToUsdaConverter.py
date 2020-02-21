import fdc
import requests
import pickle
from fdc import FdcClient
from os import environ
from sys import stderr
from datetime import date
import time

DISCARD_CACHE = ".~discard_cache.pkl"
CACHE_SAVE_INTERVAL = 500
MATCH_LOG = "ff_to_usda_matching.log"
ENABLE_MATCH_LOG = True

SHOULD_RETRY = True
RETRY_ATTEMPTS = 5
RETRY_DELAY = 60*3 # 3 minutes

class OffToUsdaConverter:
    def __init__(self, header: list, minimumScore = 100.0, picklePath = None):
        if(picklePath != None):
            self.load_state(picklePath)
        else:
            self.minimumScore = minimumScore
            self.totalItr = 0
            self.matchLog = None
            self.discardCount = 0
            self.data = []
            self.init_discard_cache(DISCARD_CACHE)
            self.header = header
            self.outputHeader = ["code", "product_name", "nova_group", "ingredients"]
            self.codeidx, self.countryidx, self.novaidx, self.nameidx, self.brandsidx = get_header_indices(header)
            self.client = FdcClient(environ['USDA_API_KEY'])

    @classmethod
    def load_from_pickle(cls, path="./.~OffToUsdaConverter.state.pkl"):
        pickleJar = open(path, 'rb');
        converter = pickle.load(pickleJar)
        pickleJar.close()
        return converter

    def save_state(self, path="./.~OffToUsdaConverter.state.pkl"):
        pickleJar = open(path, 'wb')
        matchLog = self.matchLog
        self.matchLog = None
        pickle.dump(self, pickleJar)
        pickleJar.close()
        self.matchLog = matchLog
    
    def convert(self, csvprods:list):
        if(self.totalItr != 0):
            for _ in range(self.totalItr): next(csvprods);
        if(ENABLE_MATCH_LOG):
            self.matchLog = open(MATCH_LOG, 'a')
            self.matchLog.write(f"Starting {date.today().ctime()}, with minimum score {self.minimumScore}\n")

        try:
            for product in csvprods:
                try:
                    self.match_product(product)
                except requests.exceptions.HTTPError as e:
                    print("\n", e, sep='', file=stderr)
                    tryCounter = 0
                    if(e.response.raw.status == 500):
                        print(f"Server Error on entry {self.totalItr} ({product[self.codeidx]}), skipping entry...", file=stderr)
                        continue
                    elif(e.response.raw.status == 429):
                        print("Too many requests. Waiting for an hour...", file=stderr)
                        time.sleep(60*61)
                        self.match_product(product)
                    if(SHOULD_RETRY and e.response.raw.status != 500):
                        while(tryCounter < RETRY_ATTEMPTS):
                            tryCounter+=1
                            print(f"Failed to contact FDC retrying in 3 minutes, retry #{tryCounter}", file = stderr)
                            time.sleep(RETRY_DELAY)
                            try:
                                self.match_product(product)
                                break
                            except requests.exceptions.HTTPError as e:
                                print(e)
                                pass
                        continue
                    raise e

        except KeyboardInterrupt as e:
            print("Caught Interupt. Exiting gracefully...")
            pass
        except requests.exceptions.HTTPError as e:
            print("HTTP error: You might need to give the API a break", file=stderr)
            print(e)

        prodsFinished = False
        try:
            next(csvprods)
        except StopIteration:
            prodsFinished = True

        finalReport = self.format_progress()
        print("Finish:", finalReport)

        if(not prodsFinished):
            print("Saving state...")
            self.save_state()

        if(ENABLE_MATCH_LOG):
            finish = f"Finished {date.today().ctime()}, " + finalReport
            self.matchLog.write("\n\n" + finish + "\n")
            self.matchLog.close()

        print("\n")
        return(self.outputHeader,self.data)

    def match_product(self, product):
        self.totalItr += 1
        print(self.format_progress(), end = '\r')

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
        return(f"Found {self.totalItr - self.discardCount} matches and discarded {self.discardCount} of {self.totalItr} entries")

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