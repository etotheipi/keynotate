import argparse
import curses
import json


def end_application(stdscr):
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()


class AnnotationList:
    MAX_TAG_SIZE = 4

    def __init__(self, tag_descr_pairs, hotkeys=None):
        """

        :param tag_descr_pairs: List of pairs like [['IP4', 'IPv4Address'], ['MAC', 'Hardware ID']]
        :param hotkeys:
        """
        self.tag_descr_pairs = tag_descr_pairs
        assert max([len(td[0]) for td in tag_descr_pairs]) <= AnnotationList.MAX_TAG_SIZE
        self.hotkeys = hotkeys

        if hotkeys is None:
            self.hotkeys = [td[0][0].lower() for td in self.tag_descr_pairs]
            if len(self.hotkeys) != len(set(self.hotkeys)):
                raise IOError('Auto hotkey assignment requires all short tags start with different letters')

        assert len(self.tag_descr_pairs) == len(self.hotkeys)

    def get_hotkeys(self):
        out_map = {key: td_pair for key, td_pair in zip(self.hotkeys, self.tag_descr_pairs)}
        return out_map

    def draw_annotate_legend(self, topleft_y, topleft_x, scr):
        builtin_keys = {
            '<SPACE>': ('', 'Use original tag'),
            '.': ('', 'Repeat last tag assignment'),
            'w': ('', 'Go to next _W_ord (no changes)'),
            'b': ('', 'Go _B_ack one word (no changes)'),
            '<': ('', 'Previous Sentence'),
            '>': ('', 'Next Sentence'),
            'c': ('', 'Commit sentence to disk (only when at end of sentence)'),
        }
        x_offset = topleft_x
        for key_list in [self.get_hotkeys(), builtin_keys]:
            y_offset = topleft_y
            for key, (tag, descr) in key_list.items():
                disp_str = f'({key}) {tag.ljust(5)} [{descr}]'
                scr.addstr(y_offset, x_offset, disp_str)
                y_offset += 1
            x_offset += 50


class AnnotateConfig:
    """
    Config should be a json blob with the different layers:
    """
    def __init__(self, config_file=None, config_map=None):
        if config_map is not None:
            self.config_map = config_map
        elif config_file is not None:
            with open(config_file, 'r') as f:
                self.config_map = json.load(f)
        else:
            raise IOError('Need to supply either a file or map')

        self.validate()

    def validate(self):
        assert isinstance(self.config_map, dict)
        assert 'layers' in self.config_map
        for layer_map in self.config_map['layers']:
            assert 'name' in layer_map
            assert 'tags' in layer_map
            for pairs in layer_map['tags']:
                assert len(pairs) == 2

    def get_layer_tags(self, layer_name):
        tags_list = list(filter(lambda m: m['name'] == layer_name, self.config_map['layers']))
        assert len(tags_list) == 1
        return tags_list[0]['tags']

    @staticmethod
    def create_sample_config():
        sample_config = {
            'layers': [
                {'name': 'coarse_tags',
                 'tags': [
                     ['SRC', 'Source'],
                     ['DEST', 'Destination'],
                     ['EITH', 'Either Direction (src or dst)'],
                     ['TO', 'To-Field'],
                     ['FROM', 'From-Field'],
                 ]},
                {'name': 'incl_excl',
                 'tags': [
                     ['INCL', 'Inclusion'],
                     ['EXCL', 'Exclusion'],
                 ]},
                {'name': 'and_or_single',
                 'tags': [
                     ['AND', 'Part of AND boolean subgroup'],
                     ['OR', 'Part of OR boolean subgroup'],
                     ['SNGL', 'Token is part of slot that is not a boolean group (single item)'],
                 ]},
            ]
        }

        with open('sample_config.json', 'w') as f:
            json.dump(sample_config, f, indent=2)


# We need to generate a sample config file
AnnotateConfig.create_sample_config()


class SentenceState:
    MAX_LINE_LENGTH = 120

    def __init__(self, orig_sentence, token_list, orig_tag_list='O'):
        self.orig_sentence = orig_sentence  # we need this to write out exactly what was read from file
        self.token_list = token_list

        self.orig_tag_list = orig_tag_list
        self.updated_tag_list = [''] * len(token_list)
        self.pointer_index = 0
        self.last_tag_update = 'O'
        if isinstance(orig_tag_list, str):
            self.orig_tag_list = [orig_tag_list] * len(token_list)
            self.updated_tag_list = [''] * len(token_list)

    @staticmethod
    def from_raw_sentence(sentence, tokenizer=lambda s: s.split()):
        return SentenceState(sentence, tokenizer(sentence))

    def update_tag(self, new_tag, increment_pointer=True):
        self.updated_tag_list[self.pointer_index] = new_tag
        self.last_tag_update = new_tag
        self.increment_pointer()

    def skip_tag(self):
        update_tag = self.orig_tag_list[self.pointer_index]
        self.updated_tag_list[self.pointer_index] = update_tag
        self.last_tag_update = update_tag
        self.increment_pointer()

    def decrement_pointer(self, n=1):
        self.pointer_index = max(0, self.pointer_index - n)

    def increment_pointer(self, n=1):
        self.pointer_index = min(len(self.token_list), self.pointer_index + n)

    def is_pointer_at_end(self):
        return self.pointer_index >= len(self.token_list)

    def is_complete(self):
        return ('' not in self.updated_tag_list)

    def get_centered_tags(self, tag_list, filler='_'):
        assert len(self.token_list) == len(tag_list)
        accum_tags = []
        for i in range(len(self.token_list)):
            toklen = max(AnnotationList.MAX_TAG_SIZE, len(self.token_list[i]))
            accum_tags.append(tag_list[i].center(toklen, filler))
        with open('log.txt', 'a') as f:
            f.write(str((self.token_list, self.updated_tag_list, tag_list, accum_tags)) + '\n')
        return accum_tags

    def highlight_token(self, draw_char, token_idx):
        tag_list = [''] * len(self.token_list)
        if token_idx >= len(self.token_list):
            return self.get_centered_tags(tag_list, filler=' ')

        tag_list[token_idx] = draw_char
        return self.get_centered_tags(tag_list, filler=' ')

    def draw_current_state(self, scr, spacer=' '):

        curr_display_row = 3
        all_tokens = [tok.center(AnnotationList.MAX_TAG_SIZE, ' ') for tok in self.token_list]
        char_spans = SentenceState.token_list_to_char_spans(self.orig_sentence, self.token_list, allow_empty=True)
        while True:
            # We have to complicate the heck out of this method to handle wrapping...
            curr_row_first_token_index = 0
            curr_row_last_token_index = 0
            for chspan in char_spans:
                curr_row_last_token_index += 1
                if chspan[1] > SentenceState.MAX_LINE_LENGTH:
                    break

            toks = all_tokens[curr_row_first_token_index:curr_row_last_token_index]

            orig_tags = self.get_centered_tags(self.orig_tag_list)
            arrow = self.highlight_token('V', self.pointer_index)
            upd_tags = self.get_centered_tags(self.updated_tag_list)
            if self.pointer_index == len(self.token_list):
                upd_tags.append('  [Press C to commit]')
            scr.addstr(curr_display_row, 20, spacer.join(toks))
            scr.addstr(curr_display_row+2, 2, "Original Tags")
            scr.addstr(curr_display_row+2, 20, spacer.join(orig_tags))
            scr.addstr(curr_display_row+4, 20, spacer.join(arrow))
            scr.addstr(curr_display_row+5, 2, "Updated Tags")
            scr.addstr(curr_display_row+5, 20, spacer.join(upd_tags))
            scr.refresh()

            if curr_row_last_token_index == len(self.token_list):
                break

            curr_display_row += 9
            curr_row_first_token_index = curr_row_last_token_index


    def to_json(self):
        char_spans = SentenceState.token_list_to_char_spans(
            self.orig_sentence,
            self.token_list,
            self.updated_tag_list,
            allow_empty=False)
        return json.dumps({'text': self.orig_sentence, 'labels': char_spans})

    def append_to_file(self, filename):
        with open(filename, 'a') as f:
            f.write('\n' + self.to_json())

    # This is a variety of static methods that might warrant their own class...
    @staticmethod
    def token_list_to_char_spans(sentence, token_list, tag_list=None, allow_empty=True):
        if tag_list is None:
            tag_list = [''] * len(token_list)

        offset = 0
        tok_char_spans = []
        for tok,tag in zip(token_list, tag_list):
            start = sentence.find(tok, offset)
            if tag == 'O' or (tag == '' and not allow_empty):
                continue
            tok_char_spans.append([start, start + len(tok), tag])
            offset += len(tok)

        return tok_char_spans

    @staticmethod
    def find_matching_slot(token_start, token_end, tag_span_triplets, no_find_tag='O'):
        """
        token_start - int
        token_end - int
        tag_span_triplets is a list of triplets [[start, end, tag], [start, end, tag], ...]
        """
        for slot_start, slot_end, slot_tag in tag_span_triplets:
            if slot_start <= token_start < slot_end or slot_start <= token_end < slot_end:
                overlap_start = max(token_start, slot_start)
                overlap_end = min(token_end, slot_end)
                if (overlap_end - overlap_start) > 0.5 * (token_end - token_start):
                    return slot_tag

        # If not found, return 'O'
        return no_find_tag

    @staticmethod
    def from_json(str_json, tokenizer=lambda s: s.split()):
        sent_map = json.loads(str_json)
        orig_sentence = sent_map['text']
        span_triplets = sent_map['labels']
        token_list = tokenizer(orig_sentence)
        token_spans = SentenceState.token_list_to_char_spans(orig_sentence, token_list)

        initial_tag_list = []
        for tok_start, tok_end, tag in token_spans:
            initial_tag_list.append(SentenceState.find_matching_slot(tok_start, tok_end, span_triplets))

        return SentenceState(orig_sentence, token_list, initial_tag_list)

    @staticmethod
    def write_entire_file(ss_list, filename):
        """
        This seems wildly inefficient to rewrite all entries every time, but this is just NLP.  Unless you have 100k+
        sentences, the speed will hardly be impacted and we don't have to deal with things like changed already-
        committed entries
        """
        with open(filename, 'w') as f:
            for ss in ss_list:
                f.write(ss.to_json() + '\n')


class MessageLine:
    def __init__(self, y, x, scr):
        self.y = y
        self.x = x
        self.scr = scr

    def display(self, msg):
        self.scr.addstr(self.y, self.x, msg)

    def clear(self):
        self.scr.addstr(self.y, self.x, ' '*100)


def main(stdscr):
    parser = argparse.ArgumentParser(prog='Hotkey-based Annotation Tool')
    parser.add_argument(dest='filename', type=str, help='File with existing sentences, raw or json (will be modified)')
    #parser.add_argument('-i', '--file-in', dest='file_in', type=str,
                        #help='File with existing sentences, w/ or w/o annotations')
    #parser.add_argument('-o', '--file-out', dest='file_out', type=str,
                        #help='File to put updated annotations')
    parser.add_argument('-l', '--layer-name', dest='layer_name', default=None, type=str,
                        help='Use layer name from config file.  Can be left out if only one layer defined')
    parser.add_argument('-c', '--config', dest='config', type=str, default='config.json',
                        help='Config file (see AnnotateConfig cls)')
    parser.add_argument('-s', '--skip-lines', dest='skip_lines', type=int, default=0,
                        help='Start at specific line number')
    args = parser.parse_args()

    cfg = AnnotateConfig(config_file=args.config)
    if args.layer_name is None:
        anno_list = cfg.config_map['layers'][0]['tags']
    else:
        anno_list = AnnotationList(cfg.get_layer_tags(args.layer_name))

    def read_in_whole_file(filename):
        raw_lines = [line.strip('\'" \t') for line in open(filename, 'r').read().split('\n')]
        ss_list = []
        for line in raw_lines:
            if len(line.strip()) == 0:
                continue

            if line.startswith('{'):
                ss_list.append(SentenceState.from_json(line))
            else:
                ss_list.append(SentenceState.from_raw_sentence(line))

        return ss_list


    try:
        stdscr.keypad(True)
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        print(curses.LINES, curses.COLS)
        #assert curses.LINES >= 40
        #assert curses.COLS >= 120
        msg_lines = [
            MessageLine( 1, 2, stdscr),
            MessageLine(38, 3, stdscr),
            MessageLine(39, 3, stdscr),
            MessageLine(40, 3, stdscr),
        ]

        hotkey_map = {ord(k): tag[0] for k,tag in anno_list.get_hotkeys().items()}

        sentence_index = 0
        while True:
            # Read in whole list every time so that we only ever see what has been committed to file when switching
            ss_list = read_in_whole_file(args.filename)
            ss = ss_list[sentence_index]

            while True:  # Iterate of tokens within sentence
                stdscr.clear()
                ss.draw_current_state(stdscr)
                anno_list.draw_annotate_legend(curses.LINES-10, 3, stdscr)
                msg_lines[0].display(f'Sentence {sentence_index+1} of {len(ss_list)}')
                if ss.is_pointer_at_end():
                    msg_lines[1].display('Press C to commit to file')
                stdscr.refresh()
                ch = stdscr.getch()

                if ch == ord('<'):
                    sentence_index = max(0, sentence_index - 1)
                    break
                elif ch == ord('>'):
                    sentence_index = min(len(ss_list) - 1, sentence_index + 1)
                    break
                elif ch == ord(' '):
                    if ss.is_pointer_at_end(): continue
                    ss.skip_tag()
                elif ch == ord('.'):
                    if ss.is_pointer_at_end(): continue
                    ss.update_tag(ss.last_tag_update)
                elif ch == ord('c'):
                    if ss.is_pointer_at_end():
                        if ss.is_complete():
                            SentenceState.write_entire_file(ss_list, args.filename)
                            sentence_index = min(len(ss_list) - 1, sentence_index + 1) # increment sentence index
                            break  # This skips to the next sentence
                        else:
                            msg_lines[1].display('Not all tags have been set!')
                elif ch in [ord('w'), curses.KEY_RIGHT]:
                    ss.increment_pointer()
                elif ch in [ord('b'), curses.KEY_LEFT]:
                    ss.decrement_pointer()
                elif ch == ord('q'):
                    msg_lines[1].display('Use Ctrl-C to quit!')
                elif ch in hotkey_map.keys():
                    new_tag = anno_list
                    ss.update_tag(hotkey_map[ch])

                stdscr.refresh()

    except KeyboardInterrupt:
        msg_lines[3].display('')
        stdscr.refresh()
        input('Exiting application.  Sorry, nothing you can do but press ctrl-C again to close this window')
        raise KeyboardInterrupt
    finally:
        end_application(stdscr)


curses.wrapper(main)