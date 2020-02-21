from csv import reader as CsvReader
from csv import writer as CsvWriter
import pickle
from sys import stderr
from argparse import ArgumentParser, FileType
from os import environ
from random import shuffle
from hashlib import md5

argParser = ArgumentParser()
argParser.add_argument("inputCSV", type=FileType('r'), help="CSV file containing products and ingredients as input")
argParser.add_argument("-r", "--randomize", action="store_true", help="Flags that entries should given in randomized order")


def main(args):
    print("Welcome!")
    print("Please give the following products a NOVA score.")
    print("Read about the NOVA score system here:")
    print("   https://archive.wphna.org/wp-content/uploads/2016/01/WN-2016-7-1-3-28-38-Monteiro-Cannon-Levy-et-al-NOVA.pdf")
    print("   https://world.openfoodfacts.org/nova")
    print("\n")

    inputReader = CsvReader(args.inputCSV, delimiter='\t')
    header = next(inputReader)
    if(header != ["code", "product_name", "nova_group", "ingredients"]):
        print("CSV format is not as expected! Aborting...", file=stderr)
        exit(1)

    outputName = args.inputCSV.name.replace(".csv", "_human.csv")
    outputFile = open(outputName, 'w')

    collector = None
    try:
        collector = ScoreCollector.load(outputFile)
        inputData = list(inputReader)
        args.inputCSV.seek(0)
        next(inputReader)
        collectorMatch = collector.data_matches(inputData) and collector.randomize == args.randomize
        if(not collectorMatch or collector.finished):
            if(not collectorMatch):
                print("Current config does not match saved state.")
            else:
                print("Last save was of a completed scoring.")
            try:
                response = input("Overwrite and start over?[Y/n]").upper()
                if(response == "Y"):
                    collector = ScoreCollector(inputReader, outputFile, randomize = args.randomize)
                else:
                    raise RuntimeError()
            except:
                print("Not overwriting. Aborting...")
                exit(1)
        else:
            print("Resuming saved session...")
    except Exception as e:
        collector = ScoreCollector(inputReader, outputFile, randomize = args.randomize)

    print(f"Saving responses to '{outputName}'")

    try:
        collector.run()
    except StopIteration:
        pass
    except Exception as e:
        print("Unknown Exception. Exiting...", file=stderr)
        raise e
    
    collector.save()
    outputFile.close()

class ScoreCollector:
    def __init__(self, inputReader: CsvReader, outputFile, randomize = False):
        self.inputData = list(inputReader)
        self.dataHash = md5(pickle.dumps(self.inputData)).digest()
        self.inputIdx = None
        self.indices = None
        self.outputFile = outputFile
        self.outputWriter = CsvWriter(outputFile, delimiter='\t')
        self.outputHeader = ["code", "product_name", "nova_group", "ingredients", "hu_nova_score"]
        self.randomize = randomize
        self.outputWriter.writerow(self.outputHeader)
        self.finished = False

    def run(self):
        if(self.randomize):
            if(self.indices == None or self.inputIdx == None):
                self.indices = list(range(len(self.inputData)))
                shuffle(self.indices)
            while(len(self.indices) > 0 or self.inputIdx != None):
                self.inputIdx = self.indices.pop() if self.inputIdx == None else self.inputIdx
                rowdata = self.inputData[self.inputIdx]
                score = ScoreCollector._collect_score(rowdata)
                self.inputIdx = None
                rowdata.append(score)
                self.outputWriter.writerow(rowdata)
        else:
            if(self.inputIdx == None): self.inputIdx = 0;
            for self.inputIdx in range(self.inputIdx, len(self.inputData)):
                rowdata = self.inputData[self.inputIdx]
                score = ScoreCollector._collect_score(rowdata)
                rowdata.append(score)
                self.outputWriter.writerow(rowdata)
        self.finished = True

    @staticmethod
    def _collect_score(data: list):
        print("================================================================================\n")
        print("Product: ", data[1])
        print("Ingredients: ", data[3])
        print("--------------------")
        score = None
        while(score == None):
            try: 
                score = int(input("Please input a NOVA score (1-4):"))
                if(score < 1 or score > 4): raise ValueError; 
            except ValueError:
                print("Invalid entry. Please enter a number between 1 and 4.")
                score = None
            except (KeyboardInterrupt, EOFError):
                print("\n\nThank you for your help. Exiting...")
                raise StopIteration()
        return(score)

    def save(self, path=".~ScoreCollector.pkl"):
        print("Saving ScoreCollector...")
        jar = open(path, 'wb')
        outputWriter = self.outputWriter
        outputFile = self.outputFile
        self.outputWriter = None
        self.outputFile = None
        outputFile.flush()
        pickle.dump(self, jar)
        jar.close()
        self.outputWriter = outputWriter
        self.outputFile = outputFile

    @classmethod
    def load(cls, outputFile, path=".~ScoreCollector.pkl"):
        jar = open(path, 'rb')
        collector = pickle.load(jar)
        jar.close()
        collector.outputFile = outputFile
        collector.outputWriter = CsvWriter(outputFile, delimiter='\t')
        collector.outputWriter.writerow(collector.outputHeader)
        return collector

    def data_matches(self, inputData: list):
        other = md5(pickle.dumps(inputData)).digest()
        return(other == self.dataHash)
    
if __name__ == "__main__":
    args = argParser.parse_args()
    main(args)