# Keyboard-Based Annotation Tool
### ncurses FTW

Something I threw together in a weekend to help with rapid annotation of hundreds of sentences multiple times, each with a small set of tags/slots.  Annotating each word is a single hotkey keypress, allowing you to do all annotating without leaving the homekeys.  Once you have good mental mapping of the hotkeys for each tag, you can annotate each sentence in a few seconds. 

Put all your sentences into a single, newline-separated file, update the config.json to have your desired tags under a specific layer, then run it, like the following:

```
python ncurses_annotate_main.py \
    sentences.txt \
    -c config.json \
    -l directionality_tags # needs to match one of the "layers" in config.json
```

Every time you commit a sentence it will overwrite the input file with the updated tags in .jsonl format.  So if you need to do multiple layers, make multiple initial copies of your file, and update the -l (layer) argument and input filename for each run.

This has quirks.  It's not wildly robust, but it's not wildly buggy, either.  It was written in a weekend so don't expect it to be too magical.  
