import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import time
import os
import sys
import json
import cifar10_epoch_error_report
import csv

# Script will iterate over h5 models and run error report for each
def main():
    args = sys.argv[0:]

    if len(sys.argv) == 1:
        print(' Example: input_folder error_folder report.csv')
    else:
        input = args[1].replace("\\","/")
        report = args[2].replace("\\","/")
        reportlog = report + "/report.csv"
        if len(args) == 4:
            reportlog = args[3].replace("\\","/")
        print(' validation report for H5 models in folder: ', input)
        #os.makedirs(report)
        filelist = os.listdir(input)
        filelist = sorted(filelist)
        start = time.time()
        with open(reportlog, 'w', newline='') as logfile:
            header = ["filename", "total", "correct", "error"]
            csvwriter = csv.DictWriter(logfile, fieldnames=header)
            csvwriter.writeheader()
            for f in filelist:
                if f.endswith('h5'):
                    epoch = parseFilename(f)
                    inputfile = input + '/' + f
                    outputfile = report + '/epoch_' + str(epoch) + '_errors.csv'
                    resp = cifar10_epoch_error_report.run(inputfile, outputfile)
                    csvwriter.writerow({"filename":f, "total":resp[0], "correct": resp[1], "error": resp[2]})
        end = time.time()
        elapsed = (end - start)/60
        print(' total time: ', elapsed)
# the expected file format should have the epoch after weights
# weights.01-0.363-1.097-0.597-1.050.h5
def parseFilename(file:str):
    tokens = file.split(".")
    wtokens = tokens[1].split("-")
    return wtokens[0]

# this is the recommended approach of handling main function
if __name__ == "__main__":
    main()
