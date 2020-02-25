"""

    datafilter.py

    takes in the openfoodfacts.csv file and filters out data that are only from the US,
    contain the full ingredient list, and its NOVA classification score

    data attributes include url, name, origin, countries, ingredients, and nova score

"""

import csv
import sys


# Dictionary for each header and its column position
header_dict = dict()

# Read the file
with open("openfoodfacts.csv", newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter="\t")
    header_list = next(reader)

    for index, header in enumerate(header_list):
        header_dict[header] = index

    # Get the indices of each corresponding attribute header
    url_index = header_dict["url"]
    name_index = header_dict["product_name"]
    origins_index = header_dict["origins"]
    countries_index = header_dict["countries"]
    ingredients_index = header_dict["ingredients_text"]
    nova_index = header_dict["nova_group"]

    for key, item in header_dict.items():
        print(key, item)

    """

    # Write to file
    with open("us_openfoodfacts.csv", "w+", newline="") as outfile:
        writer = csv.writer(outfile)

        for i, row in enumerate(reader):
            print(i)
            if row[ingredients_index] != "" and row[nova_index] != "" and row[countries_index] == "United States":
                outrow = [row[url_index], row[name_index], row[origins_index], row[countries_index], row[ingredients_index],  row[nova_index]]
                writer.writerow(outrow)
    """

