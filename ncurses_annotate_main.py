import argparse
import curses
import json
from collections import defaultdict

import logging
logging.basicConfig(filename='anno_log.txt', level=logging.DEBUG)

class LayerTagset:
    MAX_TAG_SIZE = 4
    RESERVED_HOTKEYS = 'wbo.<> '

    def __init__(self, tag_info_list, all_layer_names=None):
        """
        :param key_tag_descrs: List of pairs like [['i', 'IP4', 'IPv4Address'], ['m', 'MAC', 'Hardware ID']]
        :param key_tag_descrs: List of maps like [{'key': 'i', 'tag': 'IP4', 'description': 'IPv4Address'}
        :param hotkeys:
        """
        assert max([len(ti['tag']) for ti in tag_info_list]) <= LayerTagset.MAX_TAG_SIZE
        self.tag_info_list = tag_info_list
        self.all_layer_names = all_layer_names
        self.hotkey_map = {}
        for ti in self.tag_info_list:
            key = ti.get('key', ti['tag'][0].lower())  # get first letter of tag if no 'key' specified

            if key in self.hotkey_map:
                raise IOError(f'Key "{key}" is assigned to multiple tags')
            if key in LayerTagset.RESERVED_HOTKEYS:
                raise IOError(f'Key "{key}" is a reserved key, cannot bei n "{LayerTagset.RESERVED_HOTKEYS}')
            self.hotkey_map[key] = (ti['tag'], ti['description'])

    def get_hotkeys(self):
        return self.hotkey_map

    def draw_annotate_legend(self, topleft_y, topleft_x, scr, is_review_mode=False):
        builtin_keys = {
            'SPACE': ('', 'Use original tag'),
            'ENTER': ('', 'Commit/save (at end of sentence)'),
            'o': ('', 'Use O-tag'),
            '.': ('', 'Repeat last tag assignment'),
            'w': ('', 'Go to next _W_ord (no changes)'),
            'b': ('', 'Go _B_ack one word (no changes)'),
            '<': ('', 'Previous Sentence'),
            '>': ('', 'Next Sentence'),
        }

        layer_legend = {}
        for i,name in enumerate(self.all_layer_names):
            layer_legend[str(i+1)] = ('', f'Edit {name}')
        layer_legend[0] = ('', 'Review All Layers')

        left_menu = {} if is_review_mode else self.get_hotkeys()

        x_offset = topleft_x
        for key_list in [left_menu, builtin_keys, layer_legend]:
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

        self.map_tags_short_to_full = defaultdict(dict)
        self.map_tags_full_to_short = defaultdict(dict)

        for layer_info in self.config_map['layers']:
            for tag_info in layer_info['tags']:
                if 'full_tag' not in tag_info:
                    tag_info['full_tag'] = tag_info['tag']
                self.map_tags_short_to_full[layer_info['name']][tag_info['tag']] = tag_info['full_tag']
                self.map_tags_full_to_short[layer_info['name']][tag_info['full_tag']] = tag_info['tag']

        self.validate()

    def validate(self):
        assert isinstance(self.config_map, dict)
        assert 'layers' in self.config_map
        for layer_map in self.config_map['layers']:
            assert 'name' in layer_map
            assert 'tags' in layer_map
            for tag_info in layer_map['tags']:
                assert 'tag' in tag_info
                assert 'full_tag' in tag_info
                assert 'description' in tag_info

    def get_layer_tags(self, layer_name):
        tags_list = list(filter(lambda m: m['name'] == layer_name, self.config_map['layers']))
        assert len(tags_list) == 1
        return tags_list[0]['tags']

    def get_layer_names(self):
        return [submap['name'] for submap in self.config_map['layers']]

    def convert_tag_short_to_full(self, layer_name, short_tag, dne_return_orig=True):
        short_tag = short_tag.upper()
        if short_tag == 'O':
            return 'O'

        full_tag = self.map_tags_short_to_full[layer_name].get(short_tag, None)
        if dne_return_orig and full_tag is None:
            return short_tag
        else:
            return full_tag

    def convert_tag_full_to_short(self, layer_name, full_tag, dne_return_orig=True):
        full_tag = full_tag.upper()
        if full_tag == 'O':
            return 'O'

        short_tag = self.map_tags_full_to_short[layer_name].get(full_tag, None)

        if dne_return_orig and short_tag is None:
            return full_tag
        else:
            return short_tag

    @staticmethod
    def create_sample_config():
        sample_config = {
            'layers': [
                {'name': 'coarse_tags',
                 'type': 'slots',
                 'tags': [
                     {'tag': 'SRC',  'description': 'Source'},
                     {'tag': 'DEST', 'description': 'Destination'},
                     {'tag': 'EITH', 'description': 'Either Dir (src or dst)'},
                     {'tag': 'TO',   'description': 'To-Field'},
                     {'tag': 'FROM', 'description': 'From-Field'},
                 ]},
                {'name': 'incl_excl',
                 'type': 'slots',
                 'tags': [
                     {'tag': 'INCL', 'description': 'Inclusion'},
                     {'tag': 'EXCL', 'description': 'Exclusion'},
                 ]},
                {'name': 'and_or_single',
                 'type': 'slots',
                 'tags': [
                     {'tag': 'AND',  'description': 'Part of AND boolean subgroup'},
                     {'tag': 'OR',   'description': 'Part of OR boolean subgroup', 'key': 'r'},  # override hotkey
                     {'tag': 'SNGL', 'description': 'Slot is not part of a boolean group'},
                 ]},
            ]
        }

        with open('sample_config.json', 'w') as f:
            json.dump(sample_config, f, indent=2)


# We need to generate a sample config file
AnnotateConfig.create_sample_config()


class SentenceState:
    MAX_LINE_LENGTH = 120

    def __init__(self, orig_sentence, token_list, config_obj, orig_tag_lists=None, other_kv_pairs=None):
        self.orig_sentence = orig_sentence  # We need the exact input sentence for re-writing it later
        self.token_list = token_list
        self.layer_names = config_obj.get_layer_names()
        self.curr_token_index = 0
        self.last_tag_update = 'O'
        self.curr_layer_name = self.layer_names[0]
        self.config_obj = config_obj
        self.other_kv_pairs = other_kv_pairs if other_kv_pairs is not None else {}

        self.updated_tag_lists = {lname: [''] * len(token_list) for lname in self.layer_names}
        self.orig_tag_lists = orig_tag_lists
        if orig_tag_lists is None:
            self.orig_tag_lists = {lname: ['O'] * len(token_list) for lname in self.layer_names}

    @staticmethod
    def default_tokenizer(sentence):
        for punc in ',"\';':
            sentence = sentence.replace(punc, f' {punc} ')
        return sentence.split()

    @staticmethod
    def from_raw_sentence(sentence, config_obj, tokenizer=None):
        if tokenizer is None:
            tokenizer = SentenceState.default_tokenizer

        return SentenceState(sentence, tokenizer(sentence), config_obj)

    def switch_to_layer(self, layer_name):
        self.curr_layer_name = layer_name

    def increment_layer(self):
        layer_idx = self.layer_names.index(self.curr_layer_name)
        next_layer = min(len(self.layer_names)-1, layer_idx + 1)
        self.curr_layer_name = self.layer_names[next_layer]

    def update_tag(self, new_tag):
        self.updated_tag_lists[self.curr_layer_name][self.curr_token_index] = new_tag
        self.last_tag_update = new_tag
        self.increment_pointer()

    def skip_tag(self):
        update_tag = self.orig_tag_lists[self.curr_layer_name][self.curr_token_index]
        self.updated_tag_lists[self.curr_layer_name][self.curr_token_index] = update_tag
        self.last_tag_update = update_tag
        self.increment_pointer()

    def decrement_pointer(self, n=1):
        self.curr_token_index = max(0, self.curr_token_index - n)

    def increment_pointer(self, n=1):
        self.curr_token_index = min(len(self.token_list), self.curr_token_index + n)

    def is_pointer_at_end(self):
        return self.curr_token_index >= len(self.token_list)

    def is_complete(self):
        return ('' not in self.updated_tag_lists[self.curr_layer_name])

    def get_centered_tags(self, tag_list, filler='_'):
        assert len(self.token_list) == len(tag_list)
        accum_tags = []
        for i in range(len(self.token_list)):
            toklen = max(LayerTagset.MAX_TAG_SIZE, len(self.token_list[i]))
            accum_tags.append(tag_list[i].center(toklen, filler))
        return accum_tags

    def highlight_token(self, draw_char, token_idx):
        tag_list = [''] * len(self.token_list)
        if token_idx >= len(self.token_list):
            return self.get_centered_tags(tag_list, filler=' ')

        tag_list[token_idx] = draw_char
        return self.get_centered_tags(tag_list, filler=' ')

    def draw_edit_mode_state(self, scr, spacer=' '):

        curr_display_row = 4
        all_tokens = [tok.center(LayerTagset.MAX_TAG_SIZE, ' ') for tok in self.token_list]
        char_spans = SentenceState.token_list_to_char_spans(self.orig_sentence, self.token_list, allow_empty=True)
        all_orig_tags = self.get_centered_tags(self.orig_tag_lists[self.curr_layer_name])

        row_start_idx = 0
        row_last_idx = 0
        last_line_length = 0
        while True:
            # We have to complicate the heck out of this method to handle wrapping...
            for chspan in char_spans:
                row_last_idx += 1
                if chspan[1] - last_line_length > SentenceState.MAX_LINE_LENGTH:
                    last_line_length = chspan[1]
                    break

            toks = all_tokens[row_start_idx:row_last_idx]
            orig_tags = all_orig_tags[row_start_idx:row_last_idx]
            arrow1 = self.highlight_token('|', self.curr_token_index)[row_start_idx:row_last_idx]
            arrow2 = self.highlight_token('V', self.curr_token_index)[row_start_idx:row_last_idx]
            upd_tags = self.get_centered_tags(self.updated_tag_lists[self.curr_layer_name])[row_start_idx:row_last_idx]
            if self.curr_token_index == len(self.token_list):
                upd_tags.append('  [Press <Enter> to commit]')
            scr.addstr(curr_display_row, 20, spacer.join(toks))
            scr.addstr(curr_display_row+2, 2, "Original Tags")
            scr.addstr(curr_display_row+2, 20, spacer.join(orig_tags))
            scr.addstr(curr_display_row+4, 20, spacer.join(arrow1))
            scr.addstr(curr_display_row+5, 20, spacer.join(arrow2))
            scr.addstr(curr_display_row+6, 2, "Updated Tags")
            scr.addstr(curr_display_row+6, 20, spacer.join(upd_tags))
            scr.refresh()

            if row_last_idx >= len(self.token_list):
                break

            curr_display_row += 10
            row_start_idx = row_last_idx

    def draw_review_mode_state(self, scr, spacer=' '):
        curr_display_row = 4
        all_tokens = [tok.center(LayerTagset.MAX_TAG_SIZE, ' ') for tok in self.token_list]
        char_spans = SentenceState.token_list_to_char_spans(self.orig_sentence, self.token_list, allow_empty=True)
        all_orig_tags = self.get_centered_tags(self.orig_tag_lists[self.curr_layer_name])
        all_layer_tags = {l: self.get_centered_tags(self.orig_tag_lists[l]) for l in self.layer_names}

        num_layers = len(self.layer_names)

        row_start_idx = 0
        row_last_idx = 0
        last_line_length = 0
        while True:
            # We have to complicate the heck out of this method to handle wrapping...
            for chspan in char_spans:
                row_last_idx += 1
                if chspan[1] - last_line_length > SentenceState.MAX_LINE_LENGTH:
                    last_line_length = chspan[1]
                    break

            toks = all_tokens[row_start_idx:row_last_idx]
            scr.addstr(curr_display_row + 0, 2, 'Sentence:')
            scr.addstr(curr_display_row + 0, 20, spacer.join(toks))

            for i,layer_name in enumerate(self.layer_names):
                scr.addstr(curr_display_row + 2 + i,  2, f'{layer_name[:16]}')
                layer_tags = all_layer_tags[layer_name][row_start_idx:row_last_idx]
                scr.addstr(curr_display_row + 2 + i, 20, spacer.join(layer_tags))

            scr.refresh()

            if row_last_idx >= len(self.token_list):
                break

            curr_display_row += 6 + num_layers
            row_start_idx = row_last_idx

    def to_json(self):
        layer_spans = {}
        for layer_name in self.layer_names:
            layer_tags = self.updated_tag_lists[layer_name]
            if layer_tags[0] == '':
                layer_tags = self.orig_tag_lists[layer_name]

            out_name = f'label_spans_{layer_name}'
            layer_spans[out_name] = SentenceState.token_list_to_char_spans(
                self.orig_sentence,
                self.token_list,
                [self.config_obj.convert_tag_short_to_full(layer_name, t) for t in layer_tags],
                allow_empty=False)

        return json.dumps({'text': self.orig_sentence, **layer_spans, **self.other_kv_pairs})

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
        for tok, tag in zip(token_list, tag_list):
            start = sentence.find(tok, offset)
            if tag == 'O' or (tag == '' and not allow_empty):
                continue
            tok_char_spans.append([start, start + len(tok), tag])
            offset = start + len(tok)

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
    def from_json(str_json, config_obj, tokenizer=None):
        if tokenizer is None:
            tokenizer = SentenceState.default_tokenizer

        sent_map = json.loads(str_json)
        orig_sentence = sent_map['text']
        prefix = 'label_spans_'
        names_in_file = [k[len(prefix):] for k in sent_map.keys() if k.startswith(prefix)]

        for name in names_in_file:
            if name not in config_obj.get_layer_names():
                print(f'Existing annotation layer ({name}) not in config file')

        token_list = tokenizer(orig_sentence)
        token_spans = SentenceState.token_list_to_char_spans(orig_sentence, token_list)

        all_init_tag_list = {}
        for layer in config_obj.get_layer_names():
            full_layer_name = f'{prefix}{layer}'
            layer_tag_list = []
            for tok_start, tok_end, tag in token_spans:
                if full_layer_name in sent_map:
                    slot = SentenceState.find_matching_slot(tok_start, tok_end, sent_map[full_layer_name])
                else:
                    slot = 'O'

                short_tag = config_obj.convert_tag_full_to_short(layer, slot, dne_return_orig=True)
                layer_tag_list.append(short_tag)

            all_init_tag_list[layer] = layer_tag_list

        all_keys = set(sent_map.keys())
        relevant_keys = set(['text'] + [f'{prefix}{layer}' for layer in config_obj.get_layer_names()])
        other_data = {k: sent_map[k] for k in all_keys - relevant_keys}

        return SentenceState(orig_sentence, token_list, config_obj, all_init_tag_list, other_data)

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
    parser.add_argument('-c', '--config', dest='config', type=str, default='config.json',
                        help='Config file (see AnnotateConfig cls)')
    #parser.add_argument('-m', '--merge', dest='merge_file', type=str, default=None,
                        #help='Merge annotations from existing .jsonl into the file then exit')
    args = parser.parse_args()

    cfg = AnnotateConfig(config_file=args.config)
    with open('new_config.json', 'w') as f:
        json.dump(cfg.config_map, f, indent=2)

    layer_names = cfg.get_layer_names()

    def read_in_whole_file(filename, config):
        raw_lines = [line.strip() for line in open(filename, 'r').read().split('\n')]
        ss_list = []
        for line in raw_lines:
            if len(line.strip()) == 0:
                continue

            if line.strip().startswith('{'):
                ss_list.append(SentenceState.from_json(line, config))
            else:
                ss_list.append(SentenceState.from_raw_sentence(line, config))

        return ss_list


    try:
        stdscr.keypad(True)
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        print(curses.LINES, curses.COLS)
        assert curses.LINES >= 40
        assert curses.COLS >= 120
        msg_lines = [
            MessageLine( 1, 2, stdscr),
            MessageLine( 2, 2, stdscr),
            MessageLine(38, 3, stdscr),
            MessageLine(39, 3, stdscr),
            MessageLine(40, 3, stdscr),
            MessageLine(curses.LINES-1, 3, stdscr),
        ]

        sentence_index = 0
        selected_layer = cfg.get_layer_names()[0]
        is_review_mode = False
        while True:
            # Read in whole list every time so that we only ever see what has been committed to file when switching
            tagset = LayerTagset(cfg.get_layer_tags(selected_layer), layer_names)
            ss_list = read_in_whole_file(args.filename, cfg)
            ss = ss_list[sentence_index]
            ss.switch_to_layer(selected_layer)
            hotkey_map = {ord(k): tag[0] for k, tag in tagset.get_hotkeys().items()}

            while True:  # Iterate of tokens within sentence
                stdscr.clear()
                if is_review_mode:
                    ss.draw_review_mode_state(stdscr)
                else:
                    ss.draw_edit_mode_state(stdscr)
                    msg_lines[1].display(f'LAYER "{selected_layer}"')

                tagset.draw_annotate_legend(curses.LINES-10, 3, stdscr, is_review_mode)
                msg_lines[0].display(f'Sentence {sentence_index+1} of {len(ss_list)}')
                if ss.is_pointer_at_end():
                    msg_lines[-1].display('Press <Enter> to save and move on')
                stdscr.refresh()
                ch = stdscr.getch()

                if ch in [ord(str(d)) for d in range(1, len(layer_names)+1)]:
                    # Selected a different layer
                    selected_layer = cfg.get_layer_names()[int(chr(ch)) - 1]
                    is_review_mode = False
                    break
                elif ch == ord('0'):
                    is_review_mode = not is_review_mode
                    break
                elif ch in [ord('<'), curses.KEY_UP]:
                    sentence_index = max(0, sentence_index - 1)
                    break
                elif ch in [ord('>'), curses.KEY_DOWN]:
                    sentence_index = min(len(ss_list) - 1, sentence_index + 1)
                    break
                elif ch == ord(' '):
                    if ss.is_pointer_at_end(): continue
                    ss.skip_tag()
                elif ch == ord('.'):
                    if ss.is_pointer_at_end(): continue
                    ss.update_tag(ss.last_tag_update)
                elif ch in [ord('o')]:
                    ss.update_tag('O')
                elif ch in [ord('\n')]:
                    logging.error("Hit enter")
                    if not ss.is_pointer_at_end():
                        logging.error("Not at end")
                        while not ss.is_pointer_at_end():
                            ss.skip_tag()
                        logging.error("should be at end now")
                    elif ss.is_pointer_at_end():
                        if not ss.is_complete():
                            msg_lines[2].display('Not all tags have been set!')
                        else:
                            SentenceState.write_entire_file(ss_list, args.filename)
                            sentence_index = min(len(ss_list) - 1, sentence_index + 1) # increment sentence index
                            break
                elif ch in [ord('w'), curses.KEY_RIGHT]:
                    ss.increment_pointer()
                elif ch in [ord('b'), curses.KEY_LEFT, curses.KEY_BACKSPACE]:
                    ss.decrement_pointer()
                elif ch == ord('q'):
                    msg_lines[2].display('Use Ctrl-C to quit!')
                elif ch in hotkey_map.keys():
                    if not ss.is_pointer_at_end():
                        ss.update_tag(hotkey_map[ch])

                stdscr.refresh()

    except KeyboardInterrupt:
        msg_lines[-1].display('')
        stdscr.refresh()
        input('Exiting application.  Sorry, nothing you can do but press ctrl-C again to close this window')
        raise KeyboardInterrupt


curses.wrapper(main)