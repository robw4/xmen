"""Various helper functions for xmen list"""
import re


def args_to_filters(args):
    """Convert args namespace object to filters."""
    # set default to display all parameters if -p flag is set without an argument
    if len(args.filter_params) == 0:
        args.filter_params = [".*"]
    filters = [
        'root',
        'name',
        args.display_date,
        args.display_status,
        args.display_purpose,
        args.display_messages,
        args.display_version,
        args.display_meta,
        *args.filters,
        *args.filter_params]
    filters = [split_operators(f) for f in filters]
    n = len(filters) - len(args.filter_params)
    filters = [[(r'(?!_)(' if i > n - 1 else '_(') + f[0] + ')', *f[1:]] for i, f in enumerate(filters)]
    return filters


def split_operators(m):
    """Convert a string of the form 'regex op y' or 'regex' to a list of parts. Useful for generating
    filters for the visualise_params function."""
    splits = re.split(r'(<=|==|>=|<|>|!=)', m.replace(' ', ''))
    if len(splits) in [1, 3]:
        return splits
    else:
        raise RuntimeError(f'invalid parameter config passed {m}')


def visualise_params(dics, *filters):
    """Visualise a set of experiment parameters as a data-frame.

    Args:
        dics: A list of dictionaries loaded from param.yml files. Do not need to contain the same keys
        *filters: If filter is of length 1 it is treated as a regex and keys from each experiment matching the regex
            will be displayed in the data_frame. If filter is a triplet ``reg``, ``op``, ``y``,
            then experiments (rows) will only be added to the table if __all__ the keys matching
            reg satisfy the condition exec({k} {op} {y}). If more than one filter is passed the result will be the
            combination of each of the filters applied individually.

    Returns:
        df: A pandas data frame of the experiments matching the filters. Rows are experiments and columns are
            parameters.
    """
    import numpy as np
    from xmen.utils import dics_to_pandas
    # convert parameters to pandas data-frame
    df = dics_to_pandas(dics, '|'.join(f[0] for f in filters))
    keys = df.columns
    # filter parameters if filters are a tuple of (reg, op, y)
    for k in keys:
        for f in filters:
            if len(f) == 3 and re.match(f[0], k):
                reg, op, y = f
                _ = {}
                exec(f'df = df.loc[df[k] {op} {y}]', {'df': df, 'k': k}, _)
                df = _['df']

    # fix table order
    display_keys = ['_root', '_name', '_status', '_created', '_purpose']
    display_keys += [k for k in keys if not k.startswith('_')]
    display_keys += [k for k in keys if k.startswith('_messages')]
    display_keys += [k for k in keys if k.startswith('_meta')]
    display_keys += [k for k in keys if k not in display_keys]
    df = df.filter(items=display_keys)
    # update the names of the table
    names = {
        '_root': 'root', '_name': 'name', '_status': 'status', '_purpose': 'purpose',
        '_created': 'created', '_messages_': '', '_version_': '', '_meta_': ''}
    updates = {}
    for k in keys:
        for s, n in names.items():
            if k.startswith(s):
                updates[k] = k.replace(s, n)
    df = df.rename(updates, axis='columns')
    df = df.replace(np.nan, '', regex=True)
    # shorten paths for visualisation
    import os
    roots = [v["root"] for v in df.transpose().to_dict().values()]
    prefix = os.path.dirname(os.path.commonprefix(roots))
    if prefix != '/':
        roots = [r[len(prefix) + 1:] for r in roots]
    df.update({'root': roots})
    return df, prefix


def notebook_display(results):
    """Display experiments in notebook mode"""
    import textwrap
    from xmen.utils import recursive_print_lines
    notes = []
    for i, (r, e, p, n, d, t, v) in enumerate(
            zip(*[results[j] for j in ('_root', '_experiments', '_purpose',
                                       '_notes', '_created', '_type', '_version')])):
        k = 5
        i = str(i)
        note = ' ' * (k // 2 - len(str(i))) + str(i) + ' ' * (k // 2 - 1) + r + '\n' + ' ' * k
        if len(e) > 0:
            note += ('\n' + ' ' * k).join(['|- ' + ee[len(r) + 1:] for ee in e]) + '\n' + ' ' * k
        note += 'purpose: ' + p + '\n' + ' ' * k
        note += 'created: ' + d + '\n' + ' ' * k
        if len(v) > 0:
            note += 'version: ' + '\n' + ' ' * (k + 2) + ('\n' + ' ' * (k + 2)).join(
                [l for l in recursive_print_lines(v)])
        if len(n) > 0:
            note += '\n' + ' ' * k + 'notes: ' + '\n' + ' ' * (k + 2)
            note += ('\n' + ' ' * (k + 2)).join(
                ['\n'.join(textwrap.wrap(nn, width=1000, subsequent_indent=' ' * (k + 3))) for i, nn in
                 enumerate(n)])
        notes += [note]
    print('\n'.join(notes))


class NotEnoughRows(Exception):
    pass


def manage_backspace(x):
    """A validator function to manage backspacing in curses textbox. For some reason my backspace maps to 127
    (key delete)"""
    import curses
    if x in [127]:
        x = curses.KEY_BACKSPACE
    return x


def interactive_display(stdscr, results, args):
    """interactively display results with various search queries"""
    import curses
    import multiprocessing as mp
    import queue

    global root
    stdscr.refresh()

    rows, cols = stdscr.getmaxyx()

    if rows < 9:
        raise NotEnoughRows

    pos_x, pos_y = 0, 0
    last_time = None
    meta, message = [], []

    # initialise colours
    if curses.has_colors():
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(6, curses.COLOR_MAGENTA, curses.COLOR_BLACK)

    WHITE = curses.color_pair(1)
    RED = curses.color_pair(2)
    CYAN = curses.color_pair(3)
    YELLOW = curses.color_pair(4)
    GREEN = curses.color_pair(5)
    MAGNETA = curses.color_pair(6)

    global window
    window = curses.newwin(rows, cols, 0, 0)

    def generate_table(results, args):
        data_frame, root = visualise_params(results, *args_to_filters(args))
        return data_frame, root

    def toggle(args, name, update=None):
        val = getattr(args, name)
        if name == 'display_meta':
            if update in meta:
                meta.remove(update)
            else:
                meta.append(update)
            update = '|'.join(meta)
            setattr(args, name, update if update else '^$')
        elif name == 'display_messages':
            if update in message:
                message.remove(update)
            else:
                message.append(update)
            update = '|'.join(message)
            setattr(args, name, update if update else '^$')
        elif name == 'filters':
            if update == '':
                setattr(args, name, [args.filters[0]])
            else:
                setattr(args, name, getattr(args, name) + [update])
        elif name == 'filter_params':
            if update == '':
                setattr(args, name, [args.filter_params[0]])
            else:
                setattr(args, name, getattr(args, name) + [update])
        else:
            if update is None:
                update = DEFAULTS[name]
            if val == '^$':
                setattr(args, name, update)
            else:
                setattr(args, name, '^$')
        return args

    def display_row(i, pad, table=None, x=None):
        from tabulate import tabulate
        if x is None:
            x = tabulate(table, headers='keys', tablefmt='github').split('\n')
        x0 = x[0]
        xi = x[i]
        # stdscr.addstr(i + 2, 1, xx)
        status = [ii for ii, hh in enumerate(x0.split('|')) if hh.strip() == 'status']
        if i == 0:
            xi = xi.replace('|', '')
            pad.addstr(i, 1, xi, curses.A_BOLD)
        else:
            componenents = xi.split('|')
            if status:
                for j, c in enumerate(componenents):
                    if j == status[0]:
                        col = {
                            'registered': WHITE, 'timeout': CYAN,
                            'stopped': MAGNETA,
                            'running': YELLOW, 'error': RED,
                            'finished': GREEN}
                        cc = col.get(c.strip(), None)
                        if cc is None:
                            offset = sum(len(_) for _ in componenents[:j]) + 1
                            pad.addstr(i, offset, c)
                        else:
                            pad.addstr(i, sum(len(_) for _ in componenents[:j]) + 1, c, cc)
                    else:
                        pad.addstr(i, sum(len(_) for _ in componenents[:j]) + 1, c)
            else:
                pad.addstr(i, 1, ''.join(componenents))

    def display_table(table, root):
        global pad, window
        rows, cols = stdscr.getmaxyx()
        window.erase()

        from tabulate import tabulate
        x = tabulate(table, headers='keys', tablefmt='github').split('\n')
        x.pop(1)

        # if rows is None:
        import curses
        window.addstr(0, 0, f'Experiments matching {args.pattern}')
        window.addstr(1, 0, f'Roots relative to {root}')
        if last_time is not None:
            window.addstr(2, 0, f'Last message recieved @ {time.time() - last_time:.3f} seconds ago', curses.A_DIM)
        else:
            window.addstr(2, 0, f'No running experiments', curses.A_DIM)

        import curses
        window.addstr(rows - 4, 0, 'Toggles:', curses.A_BOLD)
        window.addstr(rows - 3, 0, f'date = {args.display_date} meta = {args.display_meta}, messages={args.display_messages}, version={args.display_version}')
        window.addstr(rows - 2, 0, f'param_filters={args.filter_params}, filters={args.filters}')
        window.addstr(rows - 1, 0,
                      'd=date s=status v=version p=purpose m=specify-message t=monitor-message M=meta G=gpu S=slurm c=cpu n=network V=virtual w=swap o=os D=disks')
        if len(x) > rows - 6:
            filler = '|'.join([' ...'.ljust(len(xx), ' ') if xx else ''
                                 for ii, xx in enumerate(x[1].split('|'))])
            filler = filler.replace('...', '   ', 1)
            x = [x[0], filler] + x[-(rows - 4):]

        pad = curses.newpad(len(x), len(x[0]))
        for i, xx in enumerate(x):
            display_row(i, pad, x=x)
        # for y in range(0, n * rows - 1):
        #     for x in range(0, n * cols - 1):
        #         pad.addch(y, x, ord('a') + (x * x + y * y) % 26)
        window.noutrefresh()
        pad.noutrefresh(pos_x, 0, 3, 0, rows - 5, cols - 1)
        curses.doupdate()

    def visualise_results(results, args):
        data_frame, root = generate_table(results, args)
        display_table(data_frame, root)

    visualise_results(results, args)
    rows, cols = stdscr.getmaxyx()

    try:
        import multiprocessing
        q = mp.Queue(maxsize=1)
        p = multiprocessing.Process(target=updates_client, args=(q, args))
        p.start()

        while True:
            try:
                updates, last_time = q.get(False)
                if updates is None:
                    raise RunTimeError(last_time)
                import os
                paths = [os.path.join(results[i]['_root'], results[i]['_name'])
                         for i in range(len(results))]
                idx = [(i, n) for i, n in enumerate(paths) if n in updates]
                if idx:
                    for i, n in idx:
                        results[i].update(updates[n])
                    # for i, n in idx:
                    #     for k, v in updates[n].items():
                    #         results[k][i] = v
                    visualise_results(results, args)

            except queue.Empty:
                pass

            import sys
            import time

            if stdscr is not None:
                stdscr.timeout(1000)
                c = stdscr.getch()
                if c == ord('d'):
                    toggle(args, 'display_date')
                    # args.display_date = not args.display_date
                    visualise_results(results, args)
                if c == ord('v'):
                    toggle(args, 'display_version')
                    visualise_results(results, args)
                if c == ord('s'):
                    toggle(args, 'display_status')
                    visualise_results(results, args)
                if c == ord('p'):
                    toggle(args, 'display_purpose')
                    visualise_results(results, args)
                if c == ord('M'):
                    toggle(args, 'display_meta', 'meta_(slurm_job|root$|name$|mac$|host$|user$|home$)')
                    visualise_results(results, args)
                if c == ord('V'):
                    toggle(args, 'display_meta', 'meta_virtual.*')
                    visualise_results(results, args)
                if c == ord('w'):
                    toggle(args, 'display_meta', 'meta_swap.*')
                    visualise_results(results, args)
                if c == ord('o'):
                    toggle(args, 'display_meta', 'meta_system.*')
                    visualise_results(results, args)
                if c == ord('D'):
                    toggle(args, 'display_meta', 'meta_disks.*')
                    visualise_results(results, args)
                if c == ord('c'):
                    toggle(args, 'display_meta', 'meta_cpu.*')
                    visualise_results(results, args)
                if c == ord('G'):
                    toggle(args, 'display_meta', 'meta_gpu.*')
                    visualise_results(results, args)
                if c == ord('n'):
                    toggle(args, 'display_meta', 'meta_network.*')
                    visualise_results(results, args)
                if c == ord('S'):
                    args = toggle(args, 'display_meta', 'meta_slurm.*')
                    visualise_results(results, args)
                if c == ord('t'):
                    args = toggle(args, 'display_messages', 'messages_(last|e$|s$|wall$|end$|next$|m_step$|m_load$)')
                    visualise_results(results, args)
                if c == ord('f'):
                    from curses.textpad import Textbox, rectangle
                    rows, cols = stdscr.getmaxyx()
                    window.move(rows - 4, 0)
                    window.clrtoeol()
                    window.move(rows - 3, 0)
                    window.clrtoeol()
                    window.move(rows - 2, 0)
                    window.clrtoeol()
                    window.move(rows - 1, 0)
                    window.clrtoeol()
                    window.addstr(rows - 3, 0,
                                  'Filter:',
                                  curses.A_BOLD)
                    # window.addstr(rows - 2, 0,'')
                    # window.addstr(rows - 1, 0, ' ' * (cols - 1))
                    window.refresh()
                    win = curses.newwin(1, cols, rows - 1, 0)
                    text_box = Textbox(win)
                    text = text_box.edit(validate=manage_backspace).strip()
                    args = toggle(args, 'filters', text)
                    visualise_results(results, args)
                if c == ord('P'):
                    from curses.textpad import Textbox, rectangle
                    rows, cols = stdscr.getmaxyx()
                    window.move(rows - 4, 0)
                    window.clrtoeol()
                    window.move(rows - 3, 0)
                    window.clrtoeol()
                    window.move(rows - 2, 0)
                    window.clrtoeol()
                    window.move(rows - 1, 0)
                    window.clrtoeol()
                    window.addstr(rows - 3, 0,
                                  'Filter params:',
                                  curses.A_BOLD)
                    # window.addstr(rows - 2, 0,
                    #               '(ctrl-h=backspace, "" to delete filter)')
                    # window.addstr(rows - 1, 0, ' ' * (cols - 1))
                    window.refresh()
                    win = curses.newwin(1, cols, rows - 1, 0)
                    text_box = Textbox(win)
                    text = text_box.edit(validate=manage_backspace).strip()
                    toggle(args, 'filter_params', text)
                    visualise_results(results, args)
    except KeyboardInterrupt:
        p.terminate()
        raise KeyboardInterrupt


def updates_client(q, args):
    from xmen.utils import commented_to_py
    import os
    import socket
    import time
    # get the hostname
    import struct
    import ruamel.yaml
    from xmen.experiment import IncompatibleYmlException, HOST, PORT

    server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    server_socket.bind((HOST, PORT))  # bind host address and port together

    # configure how many client the server can listen simultaneously
    server_socket.listen(100)

    while True:
        try:
            updates = {}
            last = time.time()
            while time.time() - last < args.interval:
                conn, address = server_socket.accept()
                # print("Connection from: " + str(address))
                length = conn.recv(struct.calcsize('Q'))
                if length:
                    length, = struct.unpack('Q', length)
                    # print('length = ', length)
                    params = conn.recv(length, socket.MSG_WAITALL).decode()
                    yaml = ruamel.yaml.YAML()
                    try:
                        params = yaml.load(params)
                    except:
                        raise IncompatibleYmlException
                    params = {k: commented_to_py(v) for k, v in params.items()}
                    # print(params)
                    updates[os.path.join(params['_root'], params['_name'])] = params
                conn.close()  # close the connection
            if updates:
                q.put((updates, last))
        except Exception as m:
            q.put((None, m))
            with open('/data/engs-robot-learning/kebl4674/usup/tmp/xmen-error-log.txt', 'w') as f:
                f.write(m)
            break