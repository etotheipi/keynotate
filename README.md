# Keyboard-Based Annotation Tool
### ncurses FTW

Something I threw together in a weekend to help with rapid annotation of hundreds of sentences multiple times, each with a small set of tags/slots.  Annotating each word is a single hotkey keypress, allowing you to do all annotating without leaving the homekeys.  Once you have good mental mapping of the hotkeys for each tag, you can annotate each sentence in a few seconds. 

Put all your sentences into a single, newline-separated file, update the config.json with all the labels for each layer and then run it:

```
python ncurses_annotate_main.py \
    sentences.txt \
    -c config.json \
```

(It's a feature, not a bug) Every time you commit a sentence it will overwrite the input file with the updated tags in .jsonl format and then reread it from scratch.  This sounds wildly inefficient, but only if you're annotating 100k examples with this tool (please don't).  On the other hand, it adds immediate robustness to the whole application since everything is written to disk immediately, and read/load methods are tested constantly.  If the app crashes you are pretty much guaranteed to have a valid annotation file that you can just load again.  This was a nice efficiency boost for something I was able to put together over a weekend.


