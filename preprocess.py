import argparse
import sys
import os
from pytorch_pretrained_bert import BertTokenizer, BasicTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=argparse.FileType('r'), default=sys.stdin, metavar='PATH',
                        help="The input file to be tokenized (default: STDIN")
    parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdin, metavar='PATH',
                        help="The output file containing the tokenized version of the input file (default: STDIN")
    parser.add_argument('-segment', action="store_true",
                        help="Segment the words after preprocessing?")

    args = parser.parse_args()
    if args.segment:
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    else:
        tokenizer = BasicTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    for line in args.input:
        args.output.write("%s\n" % (' '.join(tokenizer.tokenize(line.strip()))))

    args.input.close()
    args.output.close()
