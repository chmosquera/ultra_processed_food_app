import csv
from os import environ
from sys import stderr
from requests import get as GET
from usda import UsdaClient
from fdc import FdcClient
import pickle
import requests

DISCARD_CACHE = ".~discard_cache.pkl"
CACHE_SAVE_INTERVAL = 500

def main():
    header, data = (None, None)
    with open('en.openfoodfacts.org.products.csv', 'r') as csvin:
        reader = csv.reader(csvin, delimiter='\t')
        header = next(reader)
        countryidx = header.index('countries')

        filterFunc = lambda pentry: pentry[countryidx] == "United States"
        us_foods = filter(filterFunc, reader)

        header, data = convert_openff_to_usda(header, us_foods)

        del reader
        csvin.close()
    
    assert(header != None and data != None)
    with open('usda_products.csv', 'w') as csvout:
        writer = csv.writer(csvout, delimiter='\t')
        writer.writerow(header)
        writer.writerows(data)
        del writer
        csvout.close()
    print("Completed successful export to usda_products.csv")

def write_discard_cache(discardCache: set):
    pickleJar = open(DISCARD_CACHE, 'wb')
    pickle.dump(discardCache,  pickleJar)
    pickleJar.close()


def convert_openff_to_usda(header: list, csvprods):
    client = FdcClient(environ['USDA_API_KEY'])
    codeidx = header.index('code')
    countryidx = header.index('countries')
    novaidx = header.index('nova_group')
    nameidx = header.index('product_name')

    discardCache = None
    try:
        pickleJar = open(DISCARD_CACHE, 'rb');
        discardCache = pickle.load(pickleJar)
        pickleJar.close()
    except Exception as e:
        discardCache = set()

    header = ["code", "product_name", "nova_group", "ingredients"]
    data = []

    totalItr = 0
    discardCount = 0;
    try:
        for product in csvprods:
            totalItr += 1
            print(f"Found {totalItr - discardCount} matches and discarded {discardCount} after {totalItr} entries", end = '\r')

            if(totalItr % CACHE_SAVE_INTERVAL == 0):
                write_discard_cache(discardCache)

            if(product[codeidx] in discardCache):
                discardCount += 1
                continue
            elif(product[novaidx] == ''):
                discardCache.add(product[codeidx])
                discardCount += 1
                continue

            try:
                exactMatch = next(client.search(product[codeidx], productLimit=1))
                data.append([product[codeidx], product[nameidx], product[novaidx], exactMatch['ingredients']])
            except StopIteration as e:
                discardCache.add(product[codeidx])
                discardCount+=1
                continue
        
        print(f"Found {totalItr - discardCount} matches and discarded {discardCount} after {totalItr} entries")
    except KeyboardInterrupt as e:
        print("Caught Interupt. Exiting gracefully...")
        pass
    except requests.exceptions.HTTPError as e:
        print("HTTP error: You probably need to give the API a break", file=stderr)
        print(e)

    print(f"Final: {totalItr - discardCount} matches and discarded {discardCount} after {totalItr} entries")
    print("Saving discard cache...")
    write_discard_cache(discardCache)

    print("\n")
    return(header,data)


if __name__ == "__main__":
    main()