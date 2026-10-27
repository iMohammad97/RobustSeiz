if __name__ == "__main__":
    import argparse

    from eegnet_seizure.main import main

    parser = argparse.ArgumentParser(description="EEGNet seizure detection: EDF -> BIDS events TSV.")
    parser.add_argument("input", help="Path to the input EDF EEG file.")
    parser.add_argument("output", help="Path to the output TSV annotation file.")
    args = parser.parse_args()
    main(args.input, args.output)
