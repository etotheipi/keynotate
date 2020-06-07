import curses
import json


def end_application(stdscr):
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()


class AnnotationList:

    def __init__(self, short_tags, tag_descrs, hotkeys=None):
        self.short_tags = short_tags
        self.tag_descrs = tag_descrs
        self.hotkeys = hotkeys
        assert len(self.short_tags) == len(self.tag_descrs)

        if hotkeys is None:
            self.hotkeys = [t[0].lower() for t in short_tags]
            if 'o' in self.hotkeys:
                raise IOError('Cannot have an auto-hotkey tag that starts with "O"')
            if len(self.hotkeys) != len(set(self.hotkeys)):
                raise IOError('Auto hotkey assignment requires all short tags start with different letters')

        assert len(self.short_tags) == len(self.hotkeys)

    def get_hotkeys(self):
        out_map = {key: (tag, descr) for key, tag, descr in zip(self.hotkeys, self.short_tags, self.tag_descrs)}
        out_map['o'] = ('O', 'O')
        return out_map

    def draw_annotate_legend(self, topleft_y, topleft_x, scr):
        y_offset = topleft_y
        for key, (tag, descr) in self.get_hotkeys().items():
            disp_str = f'({key}) {tag.ljust(5)} [{descr}]'
            scr.addstr(y_offset, topleft_x, disp_str)
            y_offset += 1



class SentenceState:
    def __init__(self, token_list, orig_tag_list='O'):
        self.token_list = token_list
        self.orig_tag_list = orig_tag_list
        self.updated_tag_list = [''] * len(token_list)
        self.pointer_index = 0
        self.last_tag_update = 'O'
        if isinstance(orig_tag_list, str):
            self.orig_tag_list = [orig_tag_list] * len(token_list)
            self.updated_tag_list = [''] * len(token_list)

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
            toklen = max(3, len(self.token_list[i]))
            accum_tags.append(tag_list[i].center(toklen, filler))
        return accum_tags

    def highlight_token(self, draw_char, token_idx):
        tag_list = [''] * len(self.token_list)
        if token_idx >= len(self.token_list):
            return self.get_centered_tags(tag_list, filler=' ')

        tag_list[token_idx] = draw_char
        return self.get_centered_tags(tag_list, filler=' ')

    def draw_current_state(self, scr, spacer='  '):
        toks = [tok.center(3, ' ') for tok in self.token_list]
        orig_tags = self.get_centered_tags(self.orig_tag_list)
        upd_tags = self.get_centered_tags(self.updated_tag_list)
        hi1 = self.highlight_token('|', self.pointer_index)
        hi2 = self.highlight_token('V', self.pointer_index)
        scr.addstr(3, 20, spacer.join(toks))
        scr.addstr(5, 2, "Original Tags")
        scr.addstr(5, 20, spacer.join(orig_tags))
        scr.addstr(7, 20, spacer.join(hi1))
        scr.addstr(8, 20, spacer.join(hi2))
        scr.addstr(10, 2, "Updated Tags")
        scr.addstr(10, 20, spacer.join(upd_tags))
        scr.refresh()

    def parse_json(self, str_json):
        json.loads(str_json)
        raise NotImplementedError # Need to do the conversion from char spans to tokens

    def to_json(self):
        raise NotImplementedError

    def append_to_file(self, filename):
        with open(filename, 'a') as f:
            f.write('\n' + self.to_json())



def main(stdscr):
    try:
        stdscr.keypad(False)
        curses.noecho()
        curses.cbreak()
        print(curses.LINES, curses.COLS)
        assert curses.LINES >= 40
        assert curses.COLS >= 120
        MSG_ROW = 39
        MSG_COL = 3

        sentences = [
            "I want to ask marie@power.com to talk to me over IP addr 18.38.1.255",
            "Give me everything sent by joe@abc.com or 3f:11:91:a7:22:2e",
        ]
        anno_list = AnnotationList(['SRC', 'DST', 'TO', 'FRM', 'BTH'], ['Source', 'Dest', 'Sent-To', 'Rcvd-From', 'Src and Dest'])
        hotkey_map = {ord(k): tag[0] for k,tag in anno_list.get_hotkeys().items()}

        for sent in sentences:
            ss = SentenceState(sent.split(), 'O')
            while True:
                ss.draw_current_state(stdscr)
                anno_list.draw_annotate_legend(20, 3, stdscr)
                stdscr.refresh()
                ch = stdscr.getch()

                if ch == ord(' '):
                    ss.skip_tag()
                elif ch == ord('.'):
                    ss.update_tag(ss.last_tag_update)
                elif ch == ord('c'):
                    if ss.is_pointer_at_end():
                        if ss.is_complete():
                            ss.append_to_file(output_file)
                            break  # This skips to the next sentence
                        else:
                            stdscr.addstr(MSG_ROW, MSG_COL, 'Not all tags have been set!')
                elif ch in [ord('w'), curses.KEY_RIGHT]:
                    ss.increment_pointer()
                elif ch in [ord('b'), curses.KEY_LEFT]:
                    ss.decrement_pointer()
                elif ch == ord('q'):
                    stdscr.addstr(30, 10, 'Use Ctrl-C to quit!')
                elif ch in hotkey_map.keys():
                    new_tag = anno_list
                    ss.update_tag(hotkey_map[ch])

                stdscr.refresh()

    finally:
        end_application(stdscr)


curses.wrapper(main)