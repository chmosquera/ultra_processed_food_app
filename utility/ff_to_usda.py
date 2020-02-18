import csv
from OffToUsdaConverter import OffToUsdaConverter

SCORE_MINIMUM = 100

def main():
    header, data = (None, None)
    with open('en.openfoodfacts.org.products.csv', 'r') as csvin:
        reader = csv.reader(csvin, delimiter='\t')
        header = next(reader)
        countryidx = header.index('countries')

        filterFunc = lambda pentry: pentry[countryidx] == "United States"
        us_foods = filter(filterFunc, reader)

        converter = OffToUsdaConverter(header, SCORE_MINIMUM)
        header, data = converter.convert(us_foods)

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





if __name__ == "__main__":
    main()